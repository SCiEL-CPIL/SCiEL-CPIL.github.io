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
    """
    Optimized construction for K_3(7,1) using NumPy vectorized Greedy + Local Search.
    """
    N = 2187
    # Precompute the cover matrix: cover[i] is a boolean array of length N,
    # where cover[i][j] is True if word j is within distance 1 of word i.
    
    # Generate all words as indices 0..2186
    indices = np.arange(N)
    
    # Convert indices to base-3 digits
    # digits[i] is the i-th word as a tuple of 7 digits
    digits = np.zeros((N, 7), dtype=np.int8)
    tmp = indices.copy()
    for pos in range(7):
        digits[:, pos] = tmp % 3
        tmp //= 3
    
    # Precompute cover matrix
    # cover[i] covers i and all words at distance 1
    # A word j is at distance 1 from i if they differ in exactly one position.
    # We can compute this efficiently.
    
    # For each position p, and each digit d != digits[i][p], 
    # the word j is obtained by changing digit p of i to d.
    
    # Let's build the cover matrix directly.
    cover = np.zeros((N, N), dtype=bool)
    
    # Self cover
    np.fill_diagonal(cover, True)
    
    # Distance 1 covers
    for pos in range(7):
        for d in range(3):
            # Words that have digit d at position pos
            # We want to connect i to j if j differs from i only at pos and j[pos] == d
            # Actually, easier:
            # For each word i, for each pos, for each d != digits[i][pos]:
            #   j = i with digit pos set to d
            #   cover[i][j] = True
            
            # Vectorized approach:
            # Create a mask for words having digit d at pos
            mask_d = (digits[:, pos] == d)
            # For each i, if digits[i][pos] != d, then the word j formed by setting pos to d is covered.
            # j is uniquely determined by i and d.
            # j = i + (d - digits[i][pos]) * 3**pos
            
            power = 3 ** pos
            # Only for i where digits[i][pos] != d
            mask_i = (digits[:, pos] != d)
            i_indices = np.where(mask_i)[0]
            if len(i_indices) > 0:
                # Calculate j indices
                j_indices = i_indices + (d - digits[i_indices, pos]) * power
                cover[i_indices, j_indices] = True

    # cover is now a boolean matrix N x N.
    # cover[i] is the set of words covered by word i.
    
    full_mask = np.ones(N, dtype=bool)
    
    def greedy_cover(seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        uncovered = full_mask.copy()
        selected = []
        
        # We want to pick i that maximizes sum(cover[i] & uncovered)
        # This is (cover @ uncovered)
        
        while np.any(uncovered):
            # Calculate coverage counts for all words
            # cover is bool, uncovered is bool. 
            # We need integer counts.
            counts = cover @ uncovered
            
            # Find max
            max_idx = np.argmax(counts)
            max_count = counts[max_idx]
            
            if max_count == 0:
                break # Should not happen if uncovered is not empty
            
            selected.append(max_idx)
            uncovered &= ~cover[max_idx]
            
            # Safety check
            if time.time() > deadline - 5.0:
                break
        
        return selected, uncovered

    def local_search(code, deadline):
        """
        Local search with Removal and Swap.
        """
        if not code:
            return code
        
        code_set = set(code)
        # counts[j] = number of codewords covering word j
        counts = np.zeros(N, dtype=np.int32)
        
        for c in code_set:
            counts[cover[c]] += 1
            
        best_code = list(code_set)
        best_size = len(best_code)
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 2.0:
            improved = False
            iterations += 1
            
            # Phase 1: Removals
            # A codeword c can be removed if min(counts[cover[c]]) > 1
            # i.e., all words covered by c are covered by at least one other codeword.
            current_list = list(code_set)
            random.shuffle(current_list)
            
            for c in current_list:
                if c not in code_set:
                    continue
                # Check if removal is safe
                # min count in cover[c]
                min_cnt = np.min(counts[cover[c]])
                if min_cnt > 1:
                    counts[cover[c]] -= 1
                    code_set.remove(c)
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
            
            # Phase 2: Swaps
            if time.time() < deadline - 5.0:
                code_list = list(code_set)
                if code_list:
                    # Identify critical words: those with count == 1
                    critical_mask = (counts == 1)
                    
                    # We want to swap c (in code) with c' (not in code)
                    # Condition: (cover[c] & ~cover[c']) & critical_mask is empty
                    # i.e., cover[c] & critical_mask is a subset of cover[c']
                    
                    # To speed up, we can pre-filter candidates.
                    # But let's just try random swaps.
                    swap_attempts = 100
                    for _ in range(swap_attempts):
                        if time.time() > deadline - 5.0:
                            break
                        c_old = random.choice(code_list)
                        if c_old not in code_set:
                            continue
                        
                        # Pick a random candidate c_new not in code_set
                        # To avoid O(N) check for "not in code_set", we can just pick random and check.
                        # Since code size is small (~186) and N=2187, collision prob is low.
                        c_new = random.randint(0, N - 1)
                        if c_new in code_set:
                            continue
                        
                        # Check validity
                        # Words exclusively covered by c_old (i.e., count 1) must be covered by c_new
                        # exclusive_mask = cover[c_old] & critical_mask
                        # valid = not np.any(exclusive_mask & ~cover[c_new])
                        
                        exclusive = cover[c_old] & critical_mask
                        if np.any(exclusive):
                            if np.any(exclusive & ~cover[c_new]):
                                continue # Invalid swap
                        # If no exclusive words, swap is always valid (counts just change)
                        
                        # Perform swap
                        counts[cover[c_old]] -= 1
                        counts[cover[c_new]] += 1
                        code_set.remove(c_old)
                        code_set.add(c_new)
                        improved = True
                        code_list = list(code_set)
                        
                        # Update best
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)
            
            # Phase 3: 2-for-1 (Remove 2, Add 1)
            if time.time() < deadline - 5.0:
                code_list = list(code_set)
                if len(code_list) >= 2:
                    two_for_one_attempts = 50
                    for _ in range(two_for_one_attempts):
                        if time.time() > deadline - 5.0:
                            break
                        c1, c2 = random.sample(code_list, 2)
                        if c1 not in code_set or c2 not in code_set:
                            continue
                        c_new = random.randint(0, N - 1)
                        if c_new in code_set:
                            continue
                        
                        # Check validity
                        # After removing c1 and c2, counts decrease by 1 for cover[c1] and cover[c2].
                        # Words with count 1 or 2 might become 0.
                        # Specifically:
                        # If j is in cover[c1] and not cover[c2], count decreases by 1.
                        # If j is in cover[c2] and not cover[c1], count decreases by 1.
                        # If j is in both, count decreases by 2.
                        # If j is in neither, count unchanged.
                        
                        # We need: new_counts[j] >= 1 for all j.
                        # And new_counts[j] is restored by cover[c_new].
                        
                        # Let's simulate the change on a small subset or use vectorization.
                        # temp_counts = counts.copy()
                        # temp_counts[cover[c1]] -= 1
                        # temp_counts[cover[c2]] -= 1
                        # temp_counts[cover[c_new]] += 1
                        # valid = np.all(temp_counts >= 1)
                        
                        # This is O(N) per attempt. 50 attempts * 2187 ops is fast.
                        temp_counts = counts.copy()
                        temp_counts[cover[c1]] -= 1
                        temp_counts[cover[c2]] -= 1
                        temp_counts[cover[c_new]] += 1
                        
                        if np.all(temp_counts >= 1):
                            counts = temp_counts
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
    code, unc = greedy_cover()
    if not np.any(unc):
        code = local_search(code, deadline)
        if code:
            best_code = code
            best_size = len(code)
    
    # 2. Randomized Greedy + Local Search
    attempts = 0
    while time.time() < deadline - 8.0:
        attempts += 1
        code, unc = greedy_cover(seed=attempts * 7919)
        if not np.any(unc):
            code = local_search(code, deadline)
            if code and len(code) < best_size:
                best_size = len(code)
                best_code = code
    
    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_cover()
        if np.any(unc):
            # Greedy didn't finish? Just add all uncovered
            for j in np.where(unc)[0]:
                code.append(j)
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
    
    # Simple greedy
    uncovered = set(range(total))
    selected = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            cnt = 0
            for j in uncovered:
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)
            
    # Ensure validity
    if uncovered:
        for j in uncovered:
            selected.append(j)
            
    return [list(all_words[i]) for i in selected]
