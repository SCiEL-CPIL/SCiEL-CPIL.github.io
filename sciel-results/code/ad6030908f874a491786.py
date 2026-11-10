import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    total_words = 3 ** 7  # 2187
    
    # Precompute all words as tuples and indices
    # We need a mapping from index to word for output, and a way to compute distances/balls.
    # Vectorized approach:
    # 1. Generate all 2187 words as a matrix W of shape (2187, 7).
    # 2. For each word i, compute its ball (words within distance 1).
    #    Instead of iterating, we can compute the Hamming distance between all pairs?
    #    2187^2 * 7 operations is ~33 million. This is feasible with NumPy.
    
    # Generate words
    indices = np.arange(total_words)
    words_matrix = np.zeros((total_words, n), dtype=np.uint8)
    base = 1
    for i in range(n - 1, -1, -1):
        words_matrix[:, i] = (indices // base) % 3
        base *= 3

    # Compute Hamming distances between all pairs
    # diff[i, j, k] = |words_matrix[i, k] - words_matrix[j, k]|
    # dist[i, j] = sum(diff[i, j, :])
    # This requires 2187 * 2187 * 7 memory if stored as uint8, which is ~33MB. Acceptable.
    # Or we can do it in chunks to save memory, but 33MB is fine.
    
    # Calculate distances
    # W1 = W[:, np.newaxis, :] (2187, 1, 7)
    # W2 = W[np.newaxis, :, :] (1, 2187, 7)
    # Dist = np.sum(W1 != W2, axis=2)
    
    # To avoid huge temporary arrays, we can compute row by row or use a smarter trick.
    # But 33MB is small. Let's try it.
    W1 = words_matrix[:, np.newaxis, :]
    W2 = words_matrix[np.newaxis, :, :]
    
    # Compute inequality matrix
    # This creates a boolean array of shape (2187, 2187, 7)
    # We only need to know if dist <= 1.
    # dist = 0 if all equal.
    # dist = 1 if exactly one position differs.
    # dist > 1 otherwise.
    
    # Let's compute the sum of differences.
    # diff = W1 != W2
    # dist = np.sum(diff, axis=2)
    
    # Optimization: We only care about dist <= 1.
    # We can compute:
    # zero_diffs = np.all(W1 == W2, axis=2) # dist 0
    # one_diff = np.sum(W1 != W2, axis=2) == 1 # dist 1
    
    # Let's just compute the full distance matrix as uint8 to be safe and simple.
    # It's fast enough.
    dist_matrix = np.sum(W1 != W2, axis=2, dtype=np.uint8)
    
    # Ball matrix: B[i, j] is True if dist(i, j) <= radius (which is 1)
    B = dist_matrix <= 1
    
    # Now we have B (2187, 2187) boolean.
    # B[i] is the ball of word i.
    
    # Greedy Construction
    def greedy_construction(randomize=False):
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        
        while uncovered.any():
            # Compute coverage of each candidate
            # coverage[i] = number of uncovered words in ball i
            # coverage = np.sum(uncovered & B, axis=1)
            coverage = np.sum(uncovered[:, np.newaxis] & B, axis=1)
            
            max_cov = coverage.max()
            if max_cov == 0:
                break # Should not happen if B covers everything
            
            # Find candidates with max coverage
            candidates = np.where(coverage == max_cov)[0]
            
            if randomize:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
            
            selected.append(best_idx)
            # Update uncovered
            uncovered &= ~B[best_idx]
            
        return selected

    def reduce_set(selected):
        # Remove redundant codewords
        # selected is a list of indices
        # We want to remove i if B[i] is subset of union of B[j] for j != i.
        # This is equivalent to: for each i, check if there is any word k in B[i]
        # such that k is NOT in union of B[j] for j != i.
        # If such k exists, i is not redundant.
        
        selected = list(set(selected))
        if not selected:
            return selected
            
        # Create a boolean mask for selected indices
        # Instead of building a full matrix, we can iterate.
        # But we can use NumPy for the union.
        
        # Let's convert selected to a numpy array
        sel_arr = np.array(selected)
        
        # We want to find if any selected codeword is redundant.
        # Redundant if: (B[i] & ~UnionOthers) is empty.
        # UnionOthers = OR of B[j] for j in selected, j != i.
        
        # We can compute the union of all selected balls.
        # UnionAll = np.any(B[sel_arr], axis=0)
        
        # For each i in sel_arr:
        # UnionOthers = UnionAll & ~B[i]
        # If (B[i] & ~UnionOthers).any() is False, then i is redundant.
        
        # Note: If multiple codewords cover the same unique word, removing one might make it redundant?
        # No, the condition "B[i] subset of UnionOthers" is necessary and sufficient for i to be removable *immediately*.
        # We should iterate until no more can be removed.
        
        changed = True
        while changed:
            changed = False
            if len(sel_arr) == 0:
                break
                
            UnionAll = np.any(B[sel_arr], axis=0)
            
            to_remove = []
            for i in sel_arr:
                # Union of others
                # We can compute UnionOthers as UnionAll & ~B[i] ONLY IF B[i] is the only one covering some parts?
                # No, UnionAll includes B[i].
                # UnionOthers = (UnionAll | B[i]) & ~B[i] ? No.
                # UnionOthers = OR of B[j] for j != i.
                
                # Efficient way:
                # If we have UnionAll, and we remove i, the new union is UnionAll & ~B[i] ONLY IF B[i] is disjoint from others? No.
                # We need the actual union of the others.
                
                # Let's just compute it directly for correctness, it's small (<= 200 items).
                # But we can optimize:
                # If B[i] is a subset of UnionAll, it doesn't mean it's redundant.
                
                # Let's compute UnionOthers explicitly for each i?
                # That is O(N^2) boolean ops. N=200, N^2=40k. 200*40k = 8M ops. Fast.
                
                # However, we can be smarter.
                # Let's stick to the simple loop for reduction since it's called after greedy (which is slow).
                # Greedy is the bottleneck.
                
                # Actually, let's just use the set logic from before but with NumPy for the bit check?
                # No, let's use the NumPy matrix.
                
                # Union of others:
                # mask_others = np.ones(len(sel_arr), dtype=bool)
                # mask_others[i] = False
                # UnionOthers = np.any(B[sel_arr][mask_others], axis=0)
                
                # This is getting complicated to index.
                # Let's just use the previous Python set logic but optimized?
                # The previous Python set logic was:
                # union_others = 0
                # for c2 in selected: if c2 != c: union_others |= cover_masks[c2]
                # This was slow because of Python loops.
                
                # Let's try to vectorize the reduction check.
                # For each i in sel_arr:
                #   is_redundant = True
                #   # Check if any word in B[i] is not covered by others.
                #   # Covered by others = Union of B[j] for j != i.
                
                # Let's compute UnionAll once.
                # If B[i] is not a subset of UnionAll, it's definitely not redundant.
                # If it IS a subset, we need to check if it's covered by others.
                
                # Let's just do the O(N^2) check with NumPy slices.
                for idx, i in enumerate(sel_arr):
                    # Union of all others
                    others_indices = np.delete(sel_arr, idx)
                    if len(others_indices) == 0:
                        continue
                    UnionOthers = np.any(B[others_indices], axis=0)
                    
                    # Check if B[i] is subset of UnionOthers
                    # i.e., is (B[i] & ~UnionOthers).any() == False?
                    if not (B[i] & ~UnionOthers).any():
                        to_remove.append(i)
                        changed = True
                        break # Restart loop with new set for correctness (greedy reduction)
            
            if to_remove:
                # Remove all found in this pass? Or just one?
                # Removing one at a time is safer for "greedy reduction" logic.
                # But if we found multiple independent redundancies, we can remove them all.
                # Let's remove all found.
                for i in to_remove:
                    sel_arr = sel_arr[sel_arr != i]
                changed = True
        
        return list(sel_arr)

    best_size = float('inf')
    best_set = None
    
    # Run multiple randomized greedy constructions
    num_restarts = 0
    while time.time() < deadline - 20: # Leave 20s for local search
        selected = greedy_construction(randomize=True)
        reduced = reduce_set(selected)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
            num_restarts += 1
            if best_size <= 186:
                if num_restarts > 10:
                    break

    if best_set is None:
        selected = greedy_construction(randomize=False)
        best_set = reduce_set(selected)
        best_size = len(best_set)

    # Local Search / Swap Improvement
    current_set = list(set(best_set))
    current_size = len(current_set)
    
    # We will try to swap codewords to allow further reduction.
    # Strategy:
    # 1. Pick a codeword c_out.
    # 2. Remove it.
    # 3. Find uncovered words.
    # 4. Find a codeword c_in that covers all uncovered words.
    # 5. If found, swap c_out with c_in.
    # 6. Run reduction.
    # 7. If size decreases, accept.
    
    # To speed up, we can precompute the "uncovered" set quickly?
    # No, it changes with the set.
    
    # We can use the same reduction logic.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle to escape local minima
        random.shuffle(current_set)
        
        for c_out in current_set:
            if time.time() > deadline - 5:
                break
                
            # Remove c_out
            temp_set = [c for c in current_set if c != c_out]
            
            # Compute union of temp_set
            if not temp_set:
                union_temp = np.zeros(total_words, dtype=bool)
            else:
                union_temp = np.any(B[temp_set], axis=0)
            
            uncovered = ~union_temp
            
            if not uncovered.any():
                # c_out was redundant
                current_set = temp_set
                current_size -= 1
                improved = True
                break
            
            # Find best c_in
            # c_in must cover all uncovered words.
            # i.e., (uncovered & ~B[c_in]).any() == False
            # Which is equivalent to: B[c_in] covers uncovered.
            
            # We can compute for all candidates i:
            # covers_all = (uncovered[:, np.newaxis] & ~B).any(axis=1) == False
            # Or: (uncovered & ~B[i]).sum() == 0
            
            # Let's compute the number of uncovered words covered by each i.
            # cov_count = np.sum(uncovered[:, np.newaxis] & B, axis=1)
            # We need cov_count[i] == uncovered.sum()
            
            uncovered_count = uncovered.sum()
            cov_counts = np.sum(uncovered[:, np.newaxis] & B, axis=1)
            
            # Candidates that cover all uncovered
            candidates_in = np.where(cov_counts == uncovered_count)[0]
            
            # Filter out those already in temp_set
            temp_set_set = set(temp_set)
            candidates_in = [c for c in candidates_in if c not in temp_set_set]
            
            if candidates_in:
                # Pick one (randomly or first)
                c_in = random.choice(candidates_in)
                
                new_set = temp_set + [c_in]
                new_reduced = reduce_set(new_set)
                
                if len(new_reduced) < current_size:
                    current_set = new_reduced
                    current_size = len(current_set)
                    improved = True
                    break
    
    # Convert indices back to words
    result = [list(words_matrix[i]) for i in current_set]
    
    return result
