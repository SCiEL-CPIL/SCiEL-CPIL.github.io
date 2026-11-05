import itertools
import time
import sys
import numpy as np

def construct(q, n, radius):
    # For this specific task: q=3, n=7, radius=1
    # But the function signature is general. We will specialize for the known task 
    # but keep the structure. If parameters differ, we might fall back to a simple 
    # construction or raise an error, but the prompt implies this is for the specific 
    # ternary covering code problem.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple valid construction if parameters change, 
        # though the task is specific.
        # Direct sum of tetracode is valid for q=3.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: all words (valid but large)
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Specific optimization for 3, 7, 1
    total_words = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Precompute masks as boolean arrays: mask[i] is True for words within radius 1 of all_words[i]
    # We'll use a 2D boolean array of shape (total_words, total_words)
    # mask[i, j] = True if word j is within radius of word i
    
    # For radius 1 in q=3, n=7: each word covers 1 + 7*2 = 15 words
    # We can compute this efficiently by generating all neighbors
    
    # Build adjacency: for each word, list of words within distance 1
    # Use a more efficient approach: precompute the "neighbor list" for each word
    
    # Actually, let's use a different representation for speed:
    # For each word, we know its 15 neighbors (itself + 14 at distance 1)
    # We can store these as indices
    
    # Precompute all neighbor indices
    # For word i with digits d_0...d_6, neighbors are:
    # itself, and for each position p, change d_p to (d_p+1)%3 and (d_p+2)%3
    
    # Let's precompute a list of 15 indices for each word
    neighbor_lists = np.zeros((total_words, 15), dtype=np.int32)
    
    for i in range(total_words):
        word = all_words[i]
        neighbor_lists[i, 0] = i
        count = 1
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Compute index of word with position pos changed to new_val
                # Use mixed-radix conversion
                idx = 0
                for k in range(n):
                    if k == pos:
                        idx = idx * q + new_val
                    else:
                        idx = idx * q + word[k]
                neighbor_lists[i, count] = idx
                count += 1
    
    # Now for local search, we need to check coverage.
    # Represent the set of chosen codewords as a boolean array of length total_words
    # covered[j] = True if word j is within radius 1 of some chosen codeword
    
    # Greedy initialization using vectorized operations
    covered = np.zeros(total_words, dtype=bool)
    C = []  # list of chosen codeword indices
    
    # Precompute the neighbor matrix as a list of arrays for fast access
    # Actually, let's use a different approach for greedy:
    # At each step, compute gain for all unchosen words: gain[i] = number of uncovered words in neighbor_lists[i]
    
    # We can vectorize this: for each candidate i, gain[i] = sum(covered[neighbor_lists[i]] == False)
    # But doing this for all i at once requires a 2D operation
    
    # Let's use a sparse-like approach: 
    # uncovered_mask = ~covered
    # For each i, gain[i] = np.sum(uncovered_mask[neighbor_lists[i]])
    
    # This can be done as: gains = np.sum(uncovered_mask[neighbor_lists], axis=1)
    # But neighbor_lists is (total_words, 15), so neighbor_lists[all] is (total_words, total_words, 15) which is too big
    
    # Alternative: iterate but use numpy for the inner sum
    # Or: precompute a 2D boolean matrix where M[i,j] = True if j is neighbor of i
    # Then gains = M @ uncovered_mask (as vector)
    
    # Let's build the full adjacency matrix M (total_words x total_words) boolean
    # This is 2187 x 2187 = ~4.8M booleans, which is ~5MB, acceptable
    
    M = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        M[i, neighbor_lists[i]] = True
    
    # Greedy
    while not covered.all():
        uncovered = ~covered
        gains = M @ uncovered.astype(np.int32)
        # Among unchosen words, pick max gain
        # We need to exclude already chosen words
        # Create a mask of unchosen words
        chosen_mask = np.zeros(total_words, dtype=bool)
        for c in C:
            chosen_mask[c] = True
        gains[chosen_mask] = -1
        best_idx = np.argmax(gains)
        if gains[best_idx] <= 0:
            break
        C.append(best_idx)
        covered |= M[best_idx]
    
    C_set = set(C)
    
    # Local search with vectorized operations
    start_time = time.time()
    time_limit = 135.0  # Leave 15s margin
    
    # Helper: recompute covered from C
    def compute_covered(C_list):
        cov = np.zeros(total_words, dtype=bool)
        for c in C_list:
            cov |= M[c]
        return cov
    
    # Local search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Redundancy removal (vectorized)
        # For each c in C, check if M[c] is subset of OR(M[c'] for c' != c)
        # i.e., M[c] & ~rest_cover == 0
        # rest_cover = covered & ~M[c] (approximately, but not exactly if multiple codewords cover same words)
        # Actually: rest_cover[j] = True if j is covered by some c' != c
        # This is harder to compute directly. Let's use a different approach:
        # For each c, compute rest_cover by OR-ing all other masks.
        # This is O(|C|) per c, so O(|C|²) total. With |C|~200, that's 40k operations, fine.
        
        to_remove = []
        for c in C:
            rest_cover = np.zeros(total_words, dtype=bool)
            for c2 in C:
                if c2 != c:
                    rest_cover |= M[c2]
            if not (M[c] & ~rest_cover).any():
                to_remove.append(c)
        
        if to_remove:
            for c in to_remove:
                C.remove(c)
                C_set.remove(c)
            improved = True
            continue
        
        # 2. 2-for-1 replacement (vectorized)
        # For each pair (c1, c2) in C, compute exclusive coverage:
        # exclusive = (M[c1] | M[c2]) & ~(OR of all other M[c'])
        # Then check if there exists c_new not in C such that M[c_new] covers exclusive
        # i.e., exclusive & ~M[c_new] == 0, or equivalently, M[c_new] & exclusive == exclusive
        
        # To speed up, we can limit the search or use heuristics.
        # With |C|~200, there are ~20k pairs. For each pair, we need to:
        #   a. Compute rest_cover (OR of |C|-2 masks) - O(|C|)
        #   b. Compute exclusive - O(1) with precomputed OR
        #   c. Check all c_new not in C - O(total_words)
        # Total: O(|C|² * (|C| + total_words)) = O(200² * 2400) = ~96M ops, too slow in Python.
        
        # Better approach: precompute the OR of all masks in C
        # full_cover = OR of all M[c] for c in C
        # For pair (c1, c2): rest_cover = full_cover & ~M[c1] & ~M[c2] is NOT correct
        # because full_cover may have words covered only by c1 or only by c2.
        
        # Alternative: use the fact that if exclusive is small, we can check it quickly.
        # Let's try a randomized approach: sample pairs and check.
        
        # Actually, let's try a different strategy: 
        # For each c_new not in C, check if it can replace some pair.
        # This is still O(total_words * |C|²) in worst case.
        
        # Given time constraints, let's do a limited search:
        # Try up to N random pairs per iteration.
        
        import random
        random.seed(42)
        
        C_list = list(C)
        found = False
        num_pairs_to_try = min(len(C_list) * (len(C_list) - 1) // 2, 5000)
        
        # Sample pairs
        pairs = []
        for i in range(len(C_list)):
            for j in range(i + 1, len(C_list)):
                pairs.append((i, j))
        
        if len(pairs) > num_pairs_to_try:
            random.shuffle(pairs)
            pairs = pairs[:num_pairs_to_try]
        
        for (i, j) in pairs:
            c1 = C_list[i]
            c2 = C_list[j]
            
            # Compute rest_cover
            rest_cover = np.zeros(total_words, dtype=bool)
            for c3 in C_list:
                if c3 != c1 and c3 != c2:
                    rest_cover |= M[c3]
            
            exclusive = (M[c1] | M[c2]) & ~rest_cover
            if not exclusive.any():
                # Both are redundant relative to others
                C.remove(c1)
                C.remove(c2)
                C_set.remove(c1)
                C_set.remove(c2)
                improved = True
                found = True
                break
            
            # Check if any c_new covers exclusive
            # exclusive is a boolean vector. We need M[c_new] to have True wherever exclusive has True.
            # i.e., (M & exclusive[:, None]).all(axis=0) for some c_new
            # Or: for each c_new, check if exclusive & ~M[c_new] is empty
            
            # Vectorized: for all c_new, check if M[c_new] covers exclusive
            # This is: (M @ exclusive.astype(np.int32)) == exclusive.sum()
            # But M is boolean, so M @ exclusive gives the number of exclusive positions covered by each c_new
            
            cover_counts = M @ exclusive.astype(np.int32)
            exclusive_size = exclusive.sum()
            
            # Find c_new not in C with cover_counts[c_new] == exclusive_size
            candidate_mask = cover_counts == exclusive_size
            candidate_mask[list(C_set)] = False
            candidates = np.where(candidate_mask)[0]
            
            if len(candidates) > 0:
                c_new = candidates[0]
                C.remove(c1)
                C.remove(c2)
                C_set.remove(c1)
                C_set.remove(c2)
                C.append(c_new)
                C_set.add(c_new)
                improved = True
                found = True
                break
        
        if found:
            continue
        
        # 3. Kick: remove a random codeword, re-greedy the uncovered words
        if (time.time() - start_time) < time_limit - 10:
            if len(C) > 0:
                c_kick = random.choice(C)
                C.remove(c_kick)
                C_set.remove(c_kick)
                
                # Recompute covered
                covered = compute_covered(C)
                
                # Re-greedy the uncovered words
                while not covered.all():
                    uncovered = ~covered
                    gains = M @ uncovered.astype(np.int32)
                    gains[list(C_set)] = -1
                    best_idx = np.argmax(gains)
                    if gains[best_idx] <= 0:
                        break
                    C.append(best_idx)
                    C_set.add(best_idx)
                    covered |= M[best_idx]
                
                improved = True
                continue
        
        if not improved:
            break
    
    # Convert indices back to words
    return [all_words[i].tolist() for i in C]
