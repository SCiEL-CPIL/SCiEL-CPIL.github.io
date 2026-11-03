"""Superpermutation search: TSP-based construction + simulated annealing local search."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import permutations


def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _build_perm_index(n):
    """Build mapping from permutation tuple to ID, and list of all perms."""
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    return all_perms, perm_to_id


def _overlap_length(a, b, n):
    """Compute max overlap: longest suffix of a that is prefix of b."""
    max_ov = min(n - 1, len(a), len(b))
    for ov in range(max_ov, 0, -1):
        if a[-ov:] == b[:ov]:
            return ov
    return 0


def _classic_superperm(n):
    """Classic recursive construction."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new
    return word


def _tsp_construct(n, time_limit=60):
    """
    TSP-based construction: treat each permutation as a node, find a path
    visiting all nodes with minimum total transition cost (cost = n - overlap).
    Uses greedy nearest-neighbor with 2-opt refinement.
    """
    deadline = time.time() + time_limit
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = _factorial(n)
    
    # Precompute overlap for each pair is O(N^2 * n) which is too expensive for n=7
    # Instead, use a smarter approach: for each permutation, find its best successor
    # by checking all permutations and computing overlap on the fly.
    
    # For n=7, N=5040. Computing all pairs is 5040^2 * 7 ≈ 178M ops - too slow in Python.
    # Instead, use a greedy construction with limited lookahead.
    
    # Strategy: Start with a random permutation, then greedily add the unvisited
    # permutation with maximum overlap (minimum cost).
    
    # Precompute for each permutation, the set of permutations that can follow it
    # with overlap >= 1 (i.e., last char of current = first char of next, etc.)
    
    # Actually, let's use a different approach: build the word directly.
    # Start with a random permutation, then repeatedly append the character
    # that maximizes the number of new permutations covered.
    
    # Better: Use the "greedy extension" approach.
    # Start with a random permutation of length n.
    # At each step, try appending each symbol 1..n and pick the one that
    # creates the most new permutation windows.
    
    random.seed(42)
    
    best_word = None
    best_len = float('inf')
    
    # Try multiple random starting permutations
    start_perms = list(permutations(range(1, n + 1)))
    random.shuffle(start_perms)
    num_starts = min(20, len(start_perms))
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        start_perm = list(start_perms[start_idx])
        word = list(start_perm)
        
        # Track which permutations are covered
        covered = set()
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if len(set(p)) == n:
                covered.add(p)
        
        # Greedily extend
        max_extensions = len(start_perm) * _factorial(n)  # upper bound
        stuck_count = 0
        
        for step in range(max_extensions):
            if time.time() > deadline - 10:
                break
            
            # Try each symbol
            best_sym = -1
            best_new_count = 0
            best_word_after = None
            
            for sym in range(1, n + 1):
                candidate = word + [sym]
                # Only check the last window (the new one created)
                if len(candidate) >= n:
                    new_window = tuple(candidate[-n:])
                    if len(set(new_window)) == n and new_window not in covered:
                        # Count how many new permutations this creates
                        # (just this one, since we only add one char)
                        new_count = 1
                    else:
                        new_count = 0
                    
                    if new_count > best_new_count or (new_count == best_new_count and best_sym == -1):
                        best_new_count = new_count
                        best_sym = sym
                        best_word_after = candidate
            
            if best_sym == -1 or best_new_count == 0:
                # No new permutation can be created by any extension
                # Try a different strategy: look ahead 2 steps
                found = False
                for sym1 in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    for sym2 in range(1, n + 1):
                        candidate = word + [sym1, sym2]
                        new_count = 0
                        for i in range(len(candidate) - n + 1):
                            w = tuple(candidate[i:i+n])
                            if len(set(w)) == n and w not in covered:
                                new_count += 1
                        if new_count > 0:
                            # Extend by sym1, sym2
                            word = candidate
                            for i in range(len(word) - n + 1):
                                w = tuple(word[i:i+n])
                                if len(set(w)) == n:
                                    covered.add(w)
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    # Truly stuck - try inserting a symbol in the middle
                    inserted = False
                    for pos in range(len(word) + 1):
                        for sym in range(1, n + 1):
                            candidate = word[:pos] + [sym] + word[pos:]
                            new_count = 0
                            for i in range(len(candidate) - n + 1):
                                w = tuple(candidate[i:i+n])
                                if len(set(w)) == n and w not in covered:
                                    new_count += 1
                            if new_count > 0:
                                word = candidate
                                for i in range(len(word) - n + 1):
                                    w = tuple(word[i:i+n])
                                    if len(set(w)) == n:
                                        covered.add(w)
                                inserted = True
                                break
                        if inserted:
                            break
                    
                    if not inserted:
                        break
            else:
                word = best_word_after
                # Add the new window to covered
                if len(word) >= n:
                    new_window = tuple(word[-n:])
                    if len(set(new_window)) == n:
                        covered.add(new_window)
        
        # Check if we've covered all permutations
        if len(covered) >= _factorial(n) and len(word) < best_len:
            best_len = len(word)
            best_word = list(word)
    
    if best_word is not None:
        return best_word
    
    # Fallback to classic construction
    return _classic_superperm(n)


def _simulated_annealing(word, n, time_limit=30):
    """
    Simulated annealing local search on the word.
    Moves: delete a character, swap two adjacent characters, insert a character.
    Uses count tracking for fast validity checks.
    """
    deadline = time.time() + time_limit
    word = list(word)
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = _factorial(n)
    
    # Initialize counts
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
    
    zero_count = sum(1 for c in counts if c == 0)
    
    if zero_count > 0:
        return word
    
    best_word = list(word)
    best_len = len(word)
    current_temp = 1.0
    cooling_rate = 0.999
    
    L = len(word)
    
    while time.time() < deadline - 5:
        # Random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Delete a random position
            if len(word) <= n:
                continue
            pos = random.randint(0, len(word) - 1)
            
            # Try deletion
            old_counts = []
            L = len(word)
            k_start = max(0, pos - n + 1)
            k_end = min(pos, L - n)
            
            for k in range(k_start, k_end + 1):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_counts.append((k, pid))
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
            
            new_k_start = max(0, pos - n + 1)
            new_k_end = min(pos - 1, L - 1 - n)
            new_pids = []
            for k in range(new_k_start, new_k_end + 1):
                if k < pos:
                    w1 = word[k:pos]
                    w2 = word[pos+1:k+n+1]
                    p = tuple(w1 + w2)
                else:
                    p = tuple(word[k+1:k+1+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
                    new_pids.append(pid)
            
            if zero_count == 0:
                # Accept
                del word[pos]
                if len(word) < best_len:
                    best_len = len(word)
                    best_word = list(word)
            else:
                # Rollback
                for k, pid in old_counts:
                    counts[pid] += 1
                    if counts[pid] == 1:
                        zero_count -= 1
                for pid in new_pids:
                    counts[pid] -= 1
                    if counts[pid] == 0:
                        zero_count += 1
        
        elif move_type < 0.8:
            # Swap two adjacent characters
            if len(word) < 2:
                continue
            pos = random.randint(0, len(word) - 2)
            
            # Save state
            old_a = word[pos]
            old_b = word[pos+1]
            
            # Perform swap
            word[pos] = old_b
            word[pos+1] = old_a
            
            # Recompute counts (expensive but necessary)
            old_counts = list(counts)
            new_counts = [0] * num_perms
            for i in range(len(word) - n + 1):
                p = tuple(word[i:i+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    new_counts[pid] += 1
            
            new_zero = sum(1 for c in new_counts if c == 0)
            
            if new_zero == 0:
                # Valid, accept
                counts = new_counts
                zero_count = new_zero
                if len(word) < best_len:
                    best_len = len(word)
                    best_word = list(word)
            else:
                # Rollback
                word[pos] = old_a
                word[pos+1] = old_b
                counts = old_counts
        
        else:
            # Insert a random character at a random position
            pos = random.randint(0, len(word))
            sym = random.randint(1, n)
            
            # Save state
            old_counts = list(counts)
            word.insert(pos, sym)
            
            # Recompute counts
            new_counts = [0] * num_perms
            for i in range(len(word) - n + 1):
                p = tuple(word[i:i+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    new_counts[pid] += 1
            
            new_zero = sum(1 for c in new_counts if c == 0)
            
            if new_zero == 0:
                # Valid, but we got longer - only accept with probability
                if random.random() < current_temp:
                    counts = new_counts
                    zero_count = new_zero
                else:
                    # Rollback
                    del word[pos]
                    counts = old_counts
            else:
                # Invalid, rollback
                del word[pos]
                counts = old_counts
        
        current_temp *= cooling_rate
    
    return best_word


def _fast_delete_pass(word, n, time_limit=20):
    """
    Fast iterative deletion pass with count tracking.
    Multiple passes in alternating directions.
    """
    deadline = time.time() + time_limit
    word = list(word)
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = _factorial(n)
    
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
    
    zero_count = sum(1 for c in counts if c == 0)
    
    if zero_count > 0:
        return word
    
    best_len = len(word)
    
    for direction in range(4):  # 2 backward, 2 forward
        if time.time() > deadline - 2:
            break
        
        improved = True
        while improved:
            if time.time() > deadline - 2:
                break
            improved = False
            
            if direction % 2 == 0:
                positions = range(len(word) - 1, -1, -1)
            else:
                positions = range(len(word))
            
            for i in positions:
                if time.time() > deadline - 2:
                    break
                if len(word) - 1 < n:
                    break
                
                L = len(word)
                old_counts = []
                
                k_start = max(0, i - n + 1)
                k_end = min(i, L - n)
                
                for k in range(k_start, k_end + 1):
                    p = tuple(word[k:k+n])
                    pid = perm_to_id.get(p)
                    if pid is not None:
                        old_counts.append((k, pid))
                        counts[pid] -= 1
                        if counts[pid] == 0:
                            zero_count += 1
                
                new_k_start = max(0, i - n + 1)
                new_k_end = min(i - 1, L - 1 - n)
                new_pids = []
                for k in range(new_k_start, new_k_end + 1):
                    if k < i:
                        w1 = word[k:i]
                        w2 = word[i+1:k+n+1]
                        p = tuple(w1 + w2)
                    else:
                        p = tuple(word[k+1:k+1+n])
                    pid = perm_to_id.get(p)
                    if pid is not None:
                        if counts[pid] == 0:
                            zero_count -= 1
                        counts[pid] += 1
                        new_pids.append(pid)
                
                if zero_count == 0:
                    del word[i]
                    improved = True
                    best_len = len(word)
                else:
                    for k, pid in old_counts:
                        counts[pid] += 1
                        if counts[pid] == 1:
                            zero_count -= 1
                    for pid in new_pids:
                        counts[pid] -= 1
                        if counts[pid] == 0:
                            zero_count += 1
    
    return word


def construct(n):
    """
    Construct a superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    time_limit = 120  # evaluator limit
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Strategy: Try TSP-based construction first, then refine with local search.
    # If TSP fails or is too slow, fall back to classic + deletion.
    
    # Phase 1: Try TSP construction with limited time
    tsp_time = 40
    word = _tsp_construct(n, time_limit=tsp_time)
    
    # Verify it's a valid superpermutation
    all_perms, perm_to_id = _build_perm_index(n)
    covered = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if len(set(p)) == n:
            covered.add(p)
    
    if len(covered) < _factorial(n):
        # TSP construction failed, fall back to classic
        word = _classic_superperm(n)
    
    # Phase 2: Fast deletion passes
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 15
    
    if remaining > 10:
        word = _fast_delete_pass(word, n, remaining // 2)
    
    # Phase 3: Simulated annealing
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 10
    
    if remaining > 10:
        word = _simulated_annealing(word, n, remaining - 5)
    
    # Phase 4: Final deletion pass
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 5
    
    if remaining > 5:
        word = _fast_delete_pass(word, n, remaining)
    
    # Final verification
    all_perms, perm_to_id = _build_perm_index(n)
    covered = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if len(set(p)) == n:
            covered.add(p)
    
    if len(covered) < _factorial(n):
        # Shouldn't happen, but fallback
        word = _classic_superperm(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
