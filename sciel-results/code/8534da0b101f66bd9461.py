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
    
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        covs = covers  # local ref
        all_idx = all_indices
        N_local = N
        
        while uncovered:
            # Find best candidate: one that covers the most uncovered words
            best_count = -1
            best_idx = -1
            best_tie_list = None
            
            for i in all_idx:
                cnt = (uncovered & covs[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    best_tie_list = None
                elif cnt == best_count and cnt > 0:
                    if best_tie_list is None:
                        best_tie_list = [best_idx]
                    best_tie_list.append(i)
            
            if best_count == 0:
                break
                
            if randomize and best_tie_list and len(best_tie_list) > 1:
                idx = random.choice(best_tie_list)
            elif randomize and best_tie_list:
                idx = best_tie_list[0]
            else:
                idx = best_idx
            
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
                    two_for_one_attempts = 40
                    for _ in range(two_for_one_attempts):
                        if time.time() > deadline - 5.0:
                            break
                        c1, c2 = random.sample(code_list, 2)
                        if c1 not in code_set or c2 not in code_set:
                            continue
                        c_new = random.randint(0, N_local - 1)
                        if c_new in code_set:
                            continue
                        # Check: after removing c1 and c2, all words they exclusively cover must be in c_new
                        valid = True
                        for j in ci[c1]:
                            cnt = counts[j]
                            # If j is also in c2's cover, removing both reduces by 2
                            if j in ci[c2]:
                                if cnt <= 2 and not ((covs[c_new] >> j) & 1):
                                    valid = False
                                    break
                            else:
                                if cnt <= 1 and not ((covs[c_new] >> j) & 1):
                                    valid = False
                                    break
                        if valid:
                            # Also check c2's exclusive words
                            for j in ci[c2]:
                                if j in ci[c1]:
                                    continue  # already checked
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
