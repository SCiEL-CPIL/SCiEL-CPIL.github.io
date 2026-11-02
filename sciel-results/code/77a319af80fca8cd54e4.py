"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    
    # General fallback
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    """Optimized construction for K_3(7,1) using Greedy + Local Search (Removal & Swap)."""
    
    N = 2187
    all_indices = list(range(N))
    
    # Precompute covers as bitsets (Python ints)
    # covers[i] is a bitmask of all words within Hamming distance 1 of word i
    covers = [0] * N
    # Also precompute the list of indices for each cover for faster iteration in local search
    cover_indices = [None] * N
    
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    for i in range(N):
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        indices = [i]
        
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * powers[pos]
                    mask |= (1 << new_i)
                    indices.append(new_i)
        
        covers[i] = mask
        cover_indices[i] = tuple(indices)
    
    full_mask = (1 << N) - 1
    
    # Precompute NumPy adjacency matrix for fast greedy candidate scoring
    # adj[i, j] = 1 if j is within Hamming distance 1 of i
    adj = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in cover_indices[i]:
            adj[i, j] = 1
    
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        covs = covers  # local ref
        N_local = N
        adj_local = adj
        
        while uncovered:
            # Convert uncovered bitmask to boolean numpy array
            # uncovered is a Python int with N bits
            unc_arr = np.zeros(N_local, dtype=np.uint8)
            mask = uncovered
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                unc_arr[idx] = 1
                mask ^= lsb
            
            # Vectorized: count how many uncovered words each candidate covers
            # scores[i] = sum over j of adj[i,j] * unc_arr[j]
            scores = adj_local.dot(unc_arr)
            
            best_count = int(scores.max())
            if best_count == 0:
                break
            
            # Find all candidates with best score
            best_mask = (scores == best_count)
            best_indices = np.where(best_mask)[0]
            
            if randomize and len(best_indices) > 1:
                idx = int(random.choice(best_indices))
            else:
                idx = int(best_indices[0])
            
            selected.append(idx)
            uncovered &= ~covs[idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected, uncovered

    def local_search(code, deadline):
        """
        Local search with Removal, Swap, and 2-for-1 moves.
        Returns the reduced code.
        """
        if not code:
            return code
        
        code_set = set(code)
        ci = cover_indices
        covs = covers
        N_local = N
        
        # Initialize counts
        counts = [0] * N_local
        for c in code_set:
            for j in ci[c]:
                counts[j] += 1
        
        best_code = list(code_set)
        best_size = len(best_code)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Removals
            current_list = list(code_set)
            random.shuffle(current_list)
            
            for c in current_list:
                if c not in code_set:
                    continue
                safe = True
                for j in ci[c]:
                    if counts[j] <= 1:
                        safe = False
                        break
                if safe:
                    for j in ci[c]:
                        counts[j] -= 1
                    code_set.remove(c)
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
            
            # Phase 2: Swaps
            if time.time() < deadline - 5.0:
                code_list = list(code_set)
                if code_list:
                    swap_attempts = 80
                    for _ in range(swap_attempts):
                        if time.time() > deadline - 5.0:
                            break
                        c_old = random.choice(code_list)
                        if c_old not in code_set:
                            continue
                        c_new = random.randint(0, N_local - 1)
                        if c_new in code_set:
                            continue
                        valid = True
                        for j in ci[c_old]:
                            if counts[j] == 1:
                                if not ((covs[c_new] >> j) & 1):
                                    valid = False
                                    break
                        if valid:
                            for j in ci[c_old]:
                                counts[j] -= 1
                            for j in ci[c_new]:
                                counts[j] += 1
                            code_set.remove(c_old)
                            code_set.add(c_new)
                            improved = True
                            code_list = list(code_set)
            
            # Phase 3: 2-for-1 moves (remove two, add one)
            if time.time() < deadline - 5.0:
                code_list = list(code_set)
                if len(code_list) >= 2:
                    two_for_one_attempts = 60
                    for _ in range(two_for_one_attempts):
                        if time.time() > deadline - 5.0:
                            break
                        c1, c2 = random.sample(code_list, 2)
                        if c1 not in code_set or c2 not in code_set:
                            continue
                        
                        # Find candidate new codeword: prefer one that covers the union of
                        # exclusively-covered words of c1 and c2
                        # Compute words that would become uncovered if we remove c1 and c2
                        # without replacement
                        exclusive_words = set()
                        for j in ci[c1]:
                            if counts[j] == 1:
                                exclusive_words.add(j)
                        for j in ci[c2]:
                            if counts[j] == 1:
                                exclusive_words.add(j)
                        
                        if not exclusive_words:
                            # Both are redundant together; just remove one
                            # Actually this means we can just remove both if all their words
                            # are covered by others
                            # But that's a 2-for-0 move, not 2-for-1. Skip.
                            continue
                        
                        # Find best new codeword: one that covers the most exclusive_words
                        best_new = -1
                        best_coverage = -1
                        # Sample a subset of candidates for speed
                        sample_size = min(200, N_local)
                        candidates = random.sample(range(N_local), sample_size)
                        for c_new in candidates:
                            if c_new in code_set:
                                continue
                            cov_count = 0
                            for j in exclusive_words:
                                if (covs[c_new] >> j) & 1:
                                    cov_count += 1
                            if cov_count > best_coverage:
                                best_coverage = cov_count
                                best_new = c_new
                        
                        if best_new == -1 or best_coverage < len(exclusive_words):
                            continue
                        
                        # Full validation
                        c_new = best_new
                        valid = True
                        for j in ci[c1]:
                            cnt = counts[j]
                            if j in ci[c2]:
                                if cnt <= 2 and not ((covs[c_new] >> j) & 1):
                                    valid = False
                                    break
                            else:
                                if cnt <= 1 and not ((covs[c_new] >> j) & 1):
                                    valid = False
                                    break
                        if valid:
                            for j in ci[c2]:
                                if j in ci[c1]:
                                    continue
                                cnt = counts[j]
                                if cnt <= 1 and not ((covs[c_new] >> j) & 1):
                                    valid = False
                                    break
                        
                        if valid:
                            for j in ci[c1]:
                                counts[j] -= 1
                            for j in ci[c2]:
                                counts[j] -= 1
                            for j in ci[c_new]:
                                counts[j] += 1
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(c_new)
                            improved = True
                            if len(code_set) < best_size:
                                best_size = len(code_set)
                                best_code = list(code_set)
                            code_list = list(code_set)

        return best_code

    best_code = None
    best_size = N
    
    # 1. Deterministic Greedy + Local Search
    code, unc = greedy_cover(randomize=False)
    if not unc:
        code = local_search(code, deadline)
        if code:
            best_code = code
            best_size = len(code)
    
    # 2. Randomized Greedy + Local Search
    attempts = 0
    while time.time() < deadline - 8.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts * 7919)
        if not unc:
            code = local_search(code, deadline)
            if code and len(code) < best_size:
                best_size = len(code)
                best_code = code
        # Early stop if we've hit the known best
        if best_size <= 186:
            break
    
    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_cover(randomize=False)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(best_code)
    
    # Convert indices to words
    powers = [1, 3, 9, 27, 81, 243, 729]
    result = []
    for idx in best_code:
        word = [0]*7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)
    
    return result

def _construct_general(q, n, radius, deadline):
    """General fallback for other q, n, radius."""
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute covers
    covers = [set() for _ in range(total)]
    for i, w in enumerate(all_words):
        # Generate all words within radius
        # This is expensive for large n, but for small n it's fine.
        # For general case, we might not have time.
        # Let's just do a simple greedy without precomputation for now.
        pass
    
    # Simple greedy
    uncovered = set(range(total))
    selected = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            # Count how many uncovered words are within radius of i
            cnt = 0
            for j in uncovered:
                # Hamming distance
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        # Remove covered words
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)
            
    # Ensure validity
    if uncovered:
        # Add all uncovered
        for j in uncovered:
            selected.append(j)
            
    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
