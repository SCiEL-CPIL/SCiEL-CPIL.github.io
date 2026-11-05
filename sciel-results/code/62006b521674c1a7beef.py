"""Football pool problem: ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

_DEADLINE = 115.0  # seconds, safety margin before 120s kill

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
    
    # Precompute balls
    balls = _build_balls(q, n, radius)
    ball_size = len(balls[0])
    
    # Initial code: tetracode sum (81 words)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Convert to set of integer indices
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

    initial_indices = set(word_to_idx(c) for c in initial_code)
    best_code = set(initial_indices)
    best_size = len(best_code)
    
    # Coverage array: cov[i] is the number of codewords covering word i
    cov = np.zeros(total, dtype=np.int32)
    
    def update_cov(ci, delta):
        for nb in balls[ci]:
            cov[nb] += delta

    # Initialize coverage for initial code
    for ci in initial_indices:
        update_cov(ci, 1)
        
    def compute_slack(code_set):
        """
        Compute slack for each codeword in code_set.
        Slack = sum(coverage[nb] - 1 for nb in ball(ci))
        Also returns min_cov for each ci.
        A codeword is redundant if min_cov >= 2.
        """
        # We only care about codewords in the set
        cis = list(code_set)
        if not cis:
            return {}, {}
            
        # For each ci, we need min(cov[nb] for nb in balls[ci]) and sum(cov[nb]-1)
        # Doing this in a loop is okay if the set is small, but let's try to be efficient.
        # Since we are iterating over ~70-80 codewords, and ball size is 1+6*2=13,
        # 80*13 = 1040 operations. This is fast in Python.
        
        slacks = {}
        min_covs = {}
        for ci in cis:
            min_c = 999
            slack = 0
            for nb in balls[ci]:
                c_val = cov[nb]
                if c_val < min_c:
                    min_c = c_val
                slack += c_val - 1
            slacks[ci] = slack
            min_covs[ci] = min_c
        return slacks, min_covs

    def try_remove_redundant(code_set):
        """
        Repeatedly remove redundant codewords.
        Returns True if any were removed.
        """
        removed_any = False
        while True:
            slacks, min_covs = compute_slack(code_set)
            # Find redundant: min_cov >= 2
            redundant = [ci for ci, mc in min_covs.items() if mc >= 2]
            if not redundant:
                break
            
            # Remove the one with highest slack
            best_ci = max(redundant, key=lambda x: slacks[x])
            
            # Remove it
            code_set.discard(best_ci)
            update_cov(best_ci, -1)
            removed_any = True
            
            if time.time() - start_time > _DEADLINE - 5:
                break
                
        return removed_any

    # Main search loop
    for attempt in range(50):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        # Start from best found so far
        code_set = set(best_code)
        
        # Reset coverage to match code_set
        cov[:] = 0
        for ci in code_set:
            update_cov(ci, 1)
            
        # Phase 1: Remove redundant
        try_remove_redundant(code_set)
        
        # Phase 2: Local Search (Swaps)
        # If we are still above target (73), try swaps
        if len(code_set) > 73:
            # We try to swap a codeword with a neighbor to improve the "profile"
            # We prioritize swapping codewords that have low slack (critical) 
            # with neighbors that cover more "tight" words.
            
            # To make this faster, we limit the number of swap attempts per attempt
            swap_attempts = 0
            max_swap_attempts = 200
            
            while swap_attempts < max_swap_attempts:
                if time.time() - start_time > _DEADLINE - 5:
                    break
                
                slacks, min_covs = compute_slack(code_set)
                cis = list(code_set)
                if not cis:
                    break
                
                # Pick a codeword to try removing/swapping
                # Prefer those with higher slack? Or lower min_cov?
                # Actually, we want to find a swap that allows removal.
                # Let's pick a random codeword.
                ci = rng.choice(cis)
                
                # Get neighbors of ci not in code
                neighbors = balls[ci]
                candidates = [nb for nb in neighbors if nb not in code_set]
                
                if not candidates:
                    swap_attempts += 1
                    continue
                
                # Try a random candidate
                new_ci = rng.choice(candidates)
                
                # Perform swap
                old_cov_snapshot = cov.copy() # This is expensive (729 ints). 
                # Alternative: just update and revert.
                
                # Remove ci
                for nb in balls[ci]:
                    cov[nb] -= 1
                code_set.discard(ci)
                
                # Add new_ci
                for nb in balls[new_ci]:
                    cov[nb] += 1
                code_set.add(new_ci)
                
                # Try to remove redundant
                removed_any = try_remove_redundant(code_set)
                
                if removed_any and len(code_set) < best_size:
                    # Accept the swap and the removals
                    best_size = len(code_set)
                    best_code = set(code_set)
                    # Update global cov to match new best_code
                    # Actually, we are already in a state where cov matches code_set
                    # But we need to be careful if we continue the loop.
                    # Since we accepted, we can break out of this attempt's swap loop 
                    # or continue to see if we can remove more.
                    # Let's break to restart the attempt with the new best.
                    break
                else:
                    # Revert
                    # Restore cov
                    cov[:] = old_cov_snapshot
                    # Restore code_set
                    code_set.discard(new_ci)
                    code_set.add(ci)
                
                swap_attempts += 1
                
        # Update best if improved (even if no swaps were made, just in case)
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
            
        if best_size <= 73:
            break

    # Final result
    result = [idx_to_word(ci) for ci in sorted(best_code)]
    return result
# EVOLVE-BLOCK-END
