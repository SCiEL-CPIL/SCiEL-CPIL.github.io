import time
import itertools
import numpy as np
import random

_DEADLINE = 115.0  # seconds

def _build_balls(q, n, radius):
    """Precompute for each word the list of words within Hamming distance radius."""
    total = q ** n
    balls = []
    for w in range(total):
        word = [0] * n
        idx = w
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        
        neighbors = [w]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    word[pos] = v
                    nb_idx = 0
                    for val in word:
                        nb_idx = nb_idx * q + val
                    neighbors.append(nb_idx)
                    word[pos] = orig
        balls.append(neighbors)
    return balls

def construct(q, n, radius):
    start_time = time.time()
    total = q ** n
    rng = random.Random(42)
    
    balls = _build_balls(q, n, radius)
    
    # Helper functions
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # 1. Greedy Construction
    # cov[i] = number of codewords covering i
    cov = np.zeros(total, dtype=np.int32)
    code_set = set()
    
    # Precompute ball sizes for greedy? No, just iterate.
    # To speed up greedy, we can pick the word that maximizes new coverage.
    # This is O(total^2 * ball_size) which is 729^2 * 13 ~ 7M ops. Acceptable.
    
    # Actually, we can just use a known good starting point if greedy is slow.
    # But greedy is usually fine for n=6.
    
    # Let's use the tetracode start as it's very good (81 words) and covers everything.
    # Greedy might start at ~85. Tetracode is better.
    
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
            
    for c in initial_code:
        code_set.add(word_to_idx(c))
        
    for ci in code_set:
        for nb in balls[ci]:
            cov[nb] += 1
            
    best_code = set(code_set)
    best_size = len(best_code)
    
    def compute_redundant(code_set):
        """Return list of codewords that can be removed (min cov in ball >= 2)."""
        redundant = []
        for ci in code_set:
            min_c = 999
            for nb in balls[ci]:
                if cov[nb] < min_c:
                    min_c = cov[nb]
                if min_c < 2:
                    break
            if min_c >= 2:
                redundant.append(ci)
        return redundant

    def try_remove_redundant(code_set):
        removed_any = False
        while True:
            redundant = compute_redundant(code_set)
            if not redundant:
                break
            # Remove one
            ci = redundant[0] # or random
            code_set.discard(ci)
            for nb in balls[ci]:
                cov[nb] -= 1
            removed_any = True
            if time.time() - start_time > _DEADLINE - 5:
                return removed_any
        return removed_any

    # Main Local Search Loop
    # We perform "Remove-Swap-Remove" moves.
    
    for attempt in range(1000):
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        # Reset state to best known
        code_set = set(best_code)
        cov[:] = 0
        for ci in code_set:
            for nb in balls[ci]:
                cov[nb] += 1
                
        # Phase 1: Remove all redundant
        try_remove_redundant(code_set)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
            if best_size <= 71: # Can't go lower than lower bound, but maybe 72?
                break
                
        # Phase 2: Try swaps to enable further removals
        # We try to find a swap (remove c1, add c2) such that after the swap,
        # at least one codeword becomes redundant.
        
        # To make this efficient, we don't try all pairs.
        # We pick a random c1.
        # We determine which words become "critical" (cov drops to 1) when c1 is removed.
        # c2 must cover these critical words? No, c2 just needs to restore coverage to >= 1.
        # Actually, if c1 is removed, some words might have cov=0. c2 MUST cover them.
        # If no words have cov=0, c2 can be anything, but we want it to help create redundancy.
        
        swap_attempts = 0
        max_swap_attempts = 50 # Limit per restart
        
        while swap_attempts < max_swap_attempts:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            cis = list(code_set)
            if not cis:
                break
                
            c1 = rng.choice(cis)
            
            # Simulate removal of c1
            # Track words that drop to 0 (uncovered) and words that drop to 1 (critical)
            uncovered = []
            critical = []
            
            for nb in balls[c1]:
                cov[nb] -= 1
                if cov[nb] == 0:
                    uncovered.append(nb)
                elif cov[nb] == 1:
                    critical.append(nb)
            
            code_set.discard(c1)
            
            if uncovered:
                # c2 must cover all words in 'uncovered'
                # Find candidates c2 that cover all uncovered words
                # Note: c2 can be any word in the space, not just in code_set.
                # But usually, adding a word that is a neighbor of something in code_set is good.
                # Let's iterate through all possible c2? 729 options.
                # For each c2, check if it covers all 'uncovered'.
                found_candidate = False
                for c2 in range(total):
                    # Quick check: does c2 cover all uncovered?
                    # This is expensive if done naively.
                    # We can precompute a bitmask or set of balls.
                    # But 729 * 13 * 50 attempts = ~500k ops. Fine.
                    
                    valid = True
                    for u in uncovered:
                        if u not in balls[c2]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Candidate c2 found. Add it.
                    for nb in balls[c2]:
                        cov[nb] += 1
                    code_set.add(c2)
                    
                    # Now try to remove redundant
                    removed_any = try_remove_redundant(code_set)
                    
                    if removed_any and len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                        # Restart main loop with new best
                        break
                    else:
                        # Revert
                        code_set.discard(c2)
                        for nb in balls[c2]:
                            cov[nb] -= 1
                        # Restore c1
                        for nb in balls[c1]:
                            cov[nb] += 1
                        code_set.add(c1)
                        
                    found_candidate = True
                    # We only try one candidate per c1 to keep it fast?
                    # Or try a few?
                    break # Try next c1
                    
                if not found_candidate:
                    # Revert c1 removal
                    for nb in balls[c1]:
                        cov[nb] += 1
                    code_set.add(c1)
                    
            else:
                # No uncovered words. We can add any c2.
                # To maximize chance of creating redundancy, pick c2 randomly
                # or pick one that overlaps heavily with existing code.
                # Let's just pick a random c2 not in code_set.
                c2 = rng.randint(0, total - 1)
                if c2 in code_set:
                    # Try a few times to get a new one
                    for _ in range(10):
                        c2 = rng.randint(0, total - 1)
                        if c2 not in code_set:
                            break
                    if c2 in code_set:
                        # Restore c1
                        for nb in balls[c1]:
                            cov[nb] += 1
                        code_set.add(c1)
                        swap_attempts += 1
                        continue
                        
                # Add c2
                for nb in balls[c2]:
                    cov[nb] += 1
                code_set.add(c2)
                
                # Try to remove redundant
                removed_any = try_remove_redundant(code_set)
                
                if removed_any and len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = set(code_set)
                    break
                else:
                    # Revert
                    code_set.discard(c2)
                    for nb in balls[c2]:
                        cov[nb] -= 1
                    # Restore c1
                    for nb in balls[c1]:
                        cov[nb] += 1
                    code_set.add(c1)
            
            swap_attempts += 1
            
            if best_size <= 71:
                break

    result = [idx_to_word(ci) for ci in sorted(best_code)]
    return result
