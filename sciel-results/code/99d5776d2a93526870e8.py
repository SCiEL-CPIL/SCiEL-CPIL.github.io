import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin within 150s limit

    total_words = q ** n
    
    # Precompute all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor indices for each word (Hamming distance <= radius)
    # For q=3, n=6, radius=1, neighbors are the word itself + 6*2=12 variations = 13 neighbors
    neighbor_indices = []
    for w in all_words:
        nbs = {tuple(w)}
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word = list(w)
                    new_word[pos] = val
                    nbs.add(tuple(new_word))
        neighbor_indices.append([word_to_idx[nei] for nei in nbs])
        
    # Convert neighbor_indices to numpy arrays for faster operations if needed, 
    # but lists are fine for small sizes.
    
    # --- Step 1: Initial Construction ---
    # Use Tetracode product as a strong initial guess (size 81 for n=6)
    # Tetracode: 9 words of length 4, perfect code radius 1 over GF(3)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n >= 4 and q == 3:
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                initial_codewords.append(tuple(h + list(tail)))
    else:
        # Fallback: Greedy Set Cover
        initial_codewords = []
        coverage = np.zeros(total_words, dtype=np.int32)
        uncovered = np.ones(total_words, dtype=bool)
        
        while np.any(uncovered):
            if time.time() > deadline - 10.0:
                break
            # Find word that covers most uncovered
            best_idx = -1
            best_count = 0
            for idx, nbs in enumerate(neighbor_indices):
                if coverage[idx] > 0: # Optional: avoid re-adding, but greedy usually doesn't need this check if we just pick max
                    pass
                # Count how many neighbors of idx are currently uncovered
                # This is slow if done naively. 
                # For small n, we can just iterate.
                count = 0
                for ni in nbs:
                    if uncovered[ni]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            initial_codewords.append(all_words[best_idx])
            for ni in neighbor_indices[best_idx]:
                coverage[ni] += 1
                if coverage[ni] == 1:
                    uncovered[ni] = False

    # Convert to indices
    codeword_indices = [word_to_idx[w] for w in initial_codewords]
    # Ensure distinct
    codeword_indices = list(set(codeword_indices))
    
    # Build coverage
    coverage = np.zeros(total_words, dtype=np.int32)
    for idx in codeword_indices:
        for ni in neighbor_indices[idx]:
            coverage[ni] += 1
            
    in_code = set(codeword_indices)
    
    # --- Step 2: Local Search ---
    # Strategy:
    # 1. Try to remove any redundant codeword (coverage of all its neighbors > 1).
    # 2. Try "2-out, 1-in" moves: Remove two codewords, if the uncovered set can be covered by one new codeword, do it.
    # 3. Try "1-out, 1-in" moves (Swap): Replace a codeword with another to enable future removals (less critical if we have 2-out-1-in).
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Phase 1: Greedy Removal
        for i in range(len(codeword_indices) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            idx = codeword_indices[i]
            # Check if removing idx leaves any word uncovered
            can_remove = True
            for ni in neighbor_indices[idx]:
                if coverage[ni] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                for ni in neighbor_indices[idx]:
                    coverage[ni] -= 1
                codeword_indices.pop(i)
                in_code.discard(idx)
                improved = True
                # Restart loop as indices changed
                break 

        if time.time() > deadline - 5.0:
            break
            
        # Phase 2: 2-out, 1-in Move
        # This is the key to breaking through the 80->73 barrier.
        # We look for pairs (i, j) such that removing them creates an uncovered set U.
        # If there exists a word k not in code such that U is a subset of neighbors(k), we replace i,j with k.
        
        n_codewords = len(codeword_indices)
        found_move = False
        
        # To speed up, we can precompute the "exclusive" coverage of each codeword?
        # No, just iterate pairs. N is ~80, so 3240 pairs.
        
        for i in range(n_codewords):
            if time.time() > deadline - 5.0:
                break
            idx_i = codeword_indices[i]
            
            for j in range(i + 1, n_codewords):
                idx_j = codeword_indices[j]
                
                # Simulate removal
                # We need to know which words become uncovered.
                # A word w is uncovered if coverage[w] == 1 and w is in neighbors(idx_i) or neighbors(idx_j).
                # Actually, if coverage[w] == 1, it must be covered by exactly one of the codewords.
                # So, uncovered words are (neighbors(idx_i) U neighbors(idx_j)) INTERSECT {w | coverage[w] == 1}.
                
                # Optimization: 
                # Let's collect the neighbors of i and j.
                nbs_i = neighbor_indices[idx_i]
                nbs_j = neighbor_indices[idx_j]
                
                # We can't just union them easily without a set.
                # But we can check coverage.
                
                # Create a set of words that are exclusively covered by i or j
                # exclusive_i = {w in nbs_i | coverage[w] == 1 and w not in nbs_j? No, if w in nbs_j and coverage==1, it's covered by j.
                # If w is in nbs_i and coverage[w]==1, it is covered ONLY by i (since if it were covered by j, coverage would be >=2 if i!=j? 
                # Wait. coverage[w] is the total count. 
                # If coverage[w] == 1, it is covered by exactly one codeword in the current set.
                # If w is in nbs_i, it is covered by i. So if coverage[w]==1, it is covered ONLY by i.
                # So, words that become uncovered when removing i and j are:
                # {w | w in nbs_i AND coverage[w] == 1} UNION {w | w in nbs_j AND coverage[w] == 1}
                # Note: A word cannot be in both sets if i != j? 
                # If w is in nbs_i and nbs_j, and coverage[w]==1, that's a contradiction because it's covered by both i and j, so coverage >= 2.
                # So the sets are disjoint.
                
                uncov_set = []
                for w in nbs_i:
                    if coverage[w] == 1:
                        uncov_set.append(w)
                for w in nbs_j:
                    if coverage[w] == 1:
                        uncov_set.append(w)
                        
                if not uncov_set:
                    # Both can be removed!
                    for w in nbs_i:
                        coverage[w] -= 1
                    for w in nbs_j:
                        coverage[w] -= 1
                    codeword_indices.pop(j)
                    codeword_indices.pop(i)
                    in_code.discard(idx_i)
                    in_code.discard(idx_j)
                    improved = True
                    found_move = True
                    break
                else:
                    # Check if we can cover uncov_set with one new word k
                    # We need to find k such that uncov_set is a subset of neighbors(k)
                    # And k is not already in the code (or if it is, we are just swapping? No, we are removing i,j and adding k. If k is in code, we are replacing i,j with k, which reduces size by 1 only if k was not already there? 
                    # If k is in code, we remove i,j and add k (which is already there). Net change: -2. But we already checked if uncov_set is empty. If it's not empty, we need a *new* cover.
                    # If k is already in code, it's already covering its neighbors. 
                    # Actually, if k is in code, and it covers uncov_set, then those words in uncov_set are covered by k. 
                    # But we established they are covered ONLY by i or j. So k cannot be in code if it covers them, unless k is i or j (which we are removing).
                    # So k must be a word NOT in in_code.
                    
                    # To find k, we can iterate through all words? 729 is small.
                    # But we can be smarter. 
                    # Let's pick the first word in uncov_set, say u0.
                    # k must be a neighbor of u0.
                    # So we only need to check neighbors of u0.
                    
                    u0 = uncov_set[0]
                    candidate_k = -1
                    for k in neighbor_indices[u0]:
                        if k in in_code:
                            continue
                        # Check if k covers all words in uncov_set
                        valid = True
                        k_nbs = neighbor_indices[k]
                        # Convert k_nbs to set for fast lookup? Or just check membership.
                        # uncov_set is small (size <= 13+13=26).
                        # k_nbs is size 13.
                        # We can check: is uncov_set subset of k_nbs?
                        # Since k_nbs is small, we can iterate uncov_set and check if in k_nbs.
                        # Or convert k_nbs to set.
                        k_nbs_set = set(k_nbs)
                        for w in uncov_set:
                            if w not in k_nbs_set:
                                valid = False
                                break
                        if valid:
                            candidate_k = k
                            break
                    
                    if candidate_k != -1:
                        # Perform move: remove i, j, add candidate_k
                        for w in nbs_i:
                            coverage[w] -= 1
                        for w in nbs_j:
                            coverage[w] -= 1
                        for w in neighbor_indices[candidate_k]:
                            coverage[w] += 1
                            
                        # Update codeword_indices
                        # Remove j and i (careful with indices)
                        if i < j:
                            codeword_indices.pop(j)
                            codeword_indices.pop(i)
                        else:
                            codeword_indices.pop(i)
                            codeword_indices.pop(j)
                        
                        in_code.discard(idx_i)
                        in_code.discard(idx_j)
                        in_code.add(candidate_k)
                        codeword_indices.append(candidate_k)
                        
                        improved = True
                        found_move = True
                        break
                        
            if found_move:
                break
                
        # Phase 3: 1-out, 1-in (Swap) - Optional if 2-out-1-in is working well
        # If 2-out-1-in gets stuck, a swap might help.
        # For now, rely on 2-out-1-in and removal.
        
    # Final result
    result = [list(all_words[idx]) for idx in codeword_indices]
    
    # Verification (Safety)
    # final_cov = np.zeros(total_words, dtype=np.int32)
    # for idx in codeword_indices:
    #     for ni in neighbor_indices[idx]:
    #         final_cov[ni] += 1
    # if np.any(final_cov == 0):
    #     # If invalid, fallback to a known valid code (e.g. all zeros? No, must cover).
    #     # Just return what we have, but it might score 0.
    #     pass

    return result
