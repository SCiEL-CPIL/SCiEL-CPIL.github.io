import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s margin for 120s limit
    
    total_words = q ** n
    
    # Precompute neighbors as a 2D numpy array for vectorized operations
    # neighbors[i] is an array of indices of words within distance <= radius of word i
    # For q=3, n=7, radius=1, each word has 1 + 7*(3-1) = 15 neighbors (including itself)
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Build neighbor lists
    # We store them in a 2D array: neighbors[i, j] is the index of the j-th neighbor of word i
    # We also need to handle variable lengths if radius > 1 or q varies, but for this specific task (q=3, n=7, r=1),
    # the degree is constant. To be safe for general q,n,r, we might need a list of arrays or a padded array.
    # Given the constraints, we assume the specific case or handle padding.
    # Let's compute the max degree first.
    
    # For general q, n, radius:
    # Degree = sum_{k=0}^{radius} C(n, k) * (q-1)^k
    from math import comb
    max_degree = sum(comb(n, k) * ((q-1) ** k) for k in range(radius + 1))
    
    neighbors_arr = np.zeros((total_words, max_degree), dtype=np.int32)
    neighbor_counts = np.zeros(total_words, dtype=np.int32)
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        w_list = list(w)
        neigh_idx = [i]
        
        # Generate all words within Hamming distance <= radius
        # Since n=7 is small, we can just iterate through all possible changes up to radius
        # Or use a recursive generation. For r=1, it's simple.
        # For general r, we can use itertools.product for positions to change and values.
        
        # Efficient generation for small n and r:
        # We'll use a simple recursive approach or just iterate through all 2187 words if total_words is small?
        # 2187^2 is ~4.7M, which is fine for precomputation.
        # But let's stick to the neighbor generation for efficiency.
        
        # Helper to generate neighbors
        def gen_neighbors(word_tuple, r):
            res = [word_tuple]
            if r == 0:
                return res
            # Change 1 position
            for pos in range(n):
                orig = word_tuple[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_w = list(word_tuple)
                    new_w[pos] = new_val
                    new_t = tuple(new_w)
                    res.append(new_t)
                    # Recurse for remaining radius? 
                    # For r=1, we stop here.
                    # For r>1, we need to be careful not to duplicate or go out of bounds.
                    # Given the specific task is r=1, we can optimize for that.
                    # However, to be robust:
            return res
        
        # For r=1, the above is sufficient. For r>1, we need a proper combinatorial generation.
        # Let's implement a general neighbor generator using BFS or simple enumeration if total_words is small.
        # Since total_words = 3^7 = 2187, we can precompute the distance matrix? 
        # 2187x2187 is ~4.7M entries. int8 is ~4.7MB. This is very feasible.
        
        pass # We will use a simpler method below for robustness and speed in precomp
    
    # Re-doing precomputation using distance matrix for simplicity and correctness for any r
    # dist[i, j] = Hamming distance between word i and word j
    # Then neighbors[i] = [j for j in range(total_words) if dist[i, j] <= radius]
    
    # This is O(N^2 * N) to compute distances naively, which is 2187^3 ~ 10^10. Too slow.
    # We need the neighbor generation method.
    
    # Let's use the explicit neighbor generation for r=1 which is the target case.
    # If r > 1, we fall back to a slightly slower but correct method.
    
    neighbors_list = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        neigh = set([i])
        
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    neigh.add(tuple_to_idx(tuple(new_w)))
        
        # If radius > 1, we'd need to add distance 2, etc.
        # For this specific task (r=1), the above is complete.
        # To make it general for the interface, let's check if we need r>1 logic.
        # The prompt says "ternary covering code K_3(7,1)". So r=1 is fixed for the best score.
        # But the function takes radius. Let's assume r=1 for optimization, else use a generic (slower) precomp.
        
        if radius > 1:
             # Generic fallback: compute all distances
             # This is only executed if radius > 1, which is not the main target case.
             pass # Placeholder for generic logic if needed, but we focus on r=1 speed.
        
        neighbors_list.append(list(neigh))
        
    # Pad to max degree
    for i in range(total_words):
        deg = len(neighbors_list[i])
        neighbors_arr[i, :deg] = neighbors_list[i]
        neighbor_counts[i] = deg
        
    # Precompute reverse cover: for each word j, which words i cover j?
    # This is useful for swap search.
    # reverse_cover[j] is a list of i such that j is in neighbors[i]
    # Since the graph is symmetric (Hamming distance is symmetric), reverse_cover[j] == neighbors_list[j].
    # So we can just use neighbors_arr[j] as the list of potential replacements for a hole at j.
    
    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain: for each candidate i, count how many of its neighbors are uncovered
            # neighbors_arr is (N, D). We want to index uncovered with neighbors_arr.
            # uncovered[neighbors_arr[i, :neighbor_counts[i]]]
            
            # To vectorize this fully, we can use a flat array or advanced indexing.
            # gains[i] = sum(uncovered[neighbors_arr[i, :neighbor_counts[i]]])
            
            # Since neighbor_counts are all the same for r=1 (15), we can do:
            # gains = uncovered[neighbors_arr].sum(axis=1)
            # But neighbors_arr is padded. We need to mask out the padding.
            # Or, since padding is 0, and uncovered[0] is a valid word, we might double count or count wrong.
            # Better: use only the valid part.
            
            # Optimization: Since all degrees are equal for r=1, we can assume max_degree is the actual degree.
            # Let's verify if all degrees are equal.
            if np.all(neighbor_counts == max_degree):
                gains = uncovered[neighbors_arr].sum(axis=1)
            else:
                # Slower path for variable degrees
                gains = np.zeros(total_words, dtype=np.int32)
                for i in range(total_words):
                    gains[i] = uncovered[neighbors_arr[i, :neighbor_counts[i]]].sum()
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            # Random tie-breaking
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            
            # Update uncovered
            # If degrees are equal:
            if np.all(neighbor_counts == max_degree):
                uncovered[neighbors_arr[chosen, :]] = False
            else:
                uncovered[neighbors_arr[chosen, :neighbor_counts[chosen]]] = False
                
        return code

    def reduce_code_fast(code):
        if not code:
            return code
            
        # counts[j] = number of codewords covering word j
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            if np.all(neighbor_counts == max_degree):
                counts[neighbors_arr[c, :]] += 1
            else:
                counts[neighbors_arr[c, :neighbor_counts[c]]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                if np.all(neighbor_counts == max_degree):
                    counts[neighbors_arr[c, :]] -= 1
                    if np.any(counts == 0):
                        counts[neighbors_arr[c, :]] += 1
                        new_code.append(c)
                    else:
                        changed = True
                else:
                    counts[neighbors_arr[c, :neighbor_counts[c]]] -= 1
                    if np.any(counts == 0):
                        counts[neighbors_arr[c, :neighbor_counts[c]]] += 1
                        new_code.append(c)
                    else:
                        changed = True
            code = new_code
        return code

    def local_search_improve(code, deadline):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            if np.all(neighbor_counts == max_degree):
                counts[neighbors_arr[c, :]] += 1
            else:
                counts[neighbors_arr[c, :neighbor_counts[c]]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Temporarily remove c_remove
                if np.all(neighbor_counts == max_degree):
                    counts[neighbors_arr[c_remove, :]] -= 1
                else:
                    counts[neighbors_arr[c_remove, :neighbor_counts[c_remove]]] -= 1
                
                # Check if redundant
                if not np.any(counts == 0):
                    # Redundant, remove it
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Not redundant, find holes
                    # Restore counts temporarily? No, we can work with current counts (which are missing c_remove)
                    # Holes are words where counts == 0
                    holes_mask = (counts == 0)
                    
                    # Find a replacement d not in code that covers all holes
                    # d must be in the intersection of neighbors_arr[h] for all h in holes
                    # And d must not be in code.
                    
                    # Get list of holes
                    hole_indices = np.where(holes_mask)[0]
                    
                    if len(hole_indices) == 0:
                         # This shouldn't happen if not redundant, but safe check
                         if np.all(neighbor_counts == max_degree):
                             counts[neighbors_arr[c_remove, :]] += 1
                         else:
                             counts[neighbors_arr[c_remove, :neighbor_counts[c_remove]]] += 1
                         continue

                    # Candidates: start with neighbors of the first hole
                    first_hole = hole_indices[0]
                    if np.all(neighbor_counts == max_degree):
                        candidates = set(neighbors_arr[first_hole, :].tolist())
                    else:
                        candidates = set(neighbors_arr[first_hole, :neighbor_counts[first_hole]].tolist())
                    
                    # Intersect with neighbors of other holes
                    for h in hole_indices[1:]:
                        if not candidates:
                            break
                        if np.all(neighbor_counts == max_degree):
                            h_neighbors = set(neighbors_arr[h, :].tolist())
                        else:
                            h_neighbors = set(neighbors_arr[h, :neighbor_counts[h]].tolist())
                        candidates &= h_neighbors
                    
                    if not candidates:
                        # No swap found, restore counts
                        if np.all(neighbor_counts == max_degree):
                            counts[neighbors_arr[c_remove, :]] += 1
                        else:
                            counts[neighbors_arr[c_remove, :neighbor_counts[c_remove]]] += 1
                        continue
                    
                    # Remove c_remove from candidates (it's already removed from counts, but it's in the code list)
                    candidates.discard(c_remove)
                    
                    # Filter out candidates that are already in code
                    # This is O(N_code * N_candidates). N_code ~ 186, N_candidates is small.
                    code_set = set(code)
                    valid_candidates = [c for c in candidates if c not in code_set]
                    
                    if valid_candidates:
                        # Pick a random valid candidate
                        best_cand = random.choice(valid_candidates)
                        
                        # Perform swap
                        code[i] = best_cand
                        # Update counts: add best_cand
                        if np.all(neighbor_counts == max_degree):
                            counts[neighbors_arr[best_cand, :]] += 1
                        else:
                            counts[neighbors_arr[best_cand, :neighbor_counts[best_cand]]] += 1
                        # Note: counts for c_remove were already decremented.
                        # So the net effect is: -c_remove + best_cand.
                        improved = True
                        break
                    else:
                        # No valid candidate, restore counts
                        if np.all(neighbor_counts == max_degree):
                            counts[neighbors_arr[c_remove, :]] += 1
                        else:
                            counts[neighbors_arr[c_remove, :neighbor_counts[c_remove]]] += 1
                        continue
        
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        # Remove k random words
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
            
        # Repair greedily
        covered_mask = np.zeros(total_words, dtype=np.bool_)
        for c in code:
            if np.all(neighbor_counts == max_degree):
                covered_mask[neighbors_arr[c, :]] = True
            else:
                covered_mask[neighbors_arr[c, :neighbor_counts[c]]] = True
        
        uncovered = ~covered_mask
        
        while uncovered.any() and time.time() < deadline:
            if np.all(neighbor_counts == max_degree):
                gains = uncovered[neighbors_arr].sum(axis=1)
            else:
                gains = np.zeros(total_words, dtype=np.int32)
                for i in range(total_words):
                    gains[i] = uncovered[neighbors_arr[i, :neighbor_counts[i]]].sum()
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            
            if np.all(neighbor_counts == max_degree):
                uncovered[neighbors_arr[chosen, :]] = False
            else:
                uncovered[neighbors_arr[chosen, :neighbor_counts[chosen]]] = False
                
        return code

    best_code = None
    best_size = float('inf')
    
    seed = 0
    # Determine if we can use the fast path (constant degree)
    constant_degree = bool(np.all(neighbor_counts == max_degree))
    
    while time.time() < deadline:
        # 1. Greedy
        code = greedy_construct_vectorized(seed)
        
        # 2. Reduce
        code = reduce_code_fast(code)
        
        current_size = len(code)
        
        # 3. Local Search
        # Only do local search if we have time
        if time.time() < deadline - 5:
            code = local_search_improve(code, deadline - 2)
            code = reduce_code_fast(code)
            current_size = len(code)
            
        if current_size < best_size:
            best_size = current_size
            best_code = code
            
        # 4. Perturb and Repair
        if time.time() < deadline - 10 and best_code:
            # Perturb best code
            k = random.randint(2, 5)
            perturbed = perturb_repair(best_code, k, deadline - 5)
            
            # Reduce and Local Search perturbed
            perturbed = reduce_code_fast(perturbed)
            if time.time() < deadline - 5:
                perturbed = local_search_improve(perturbed, deadline - 2)
                perturbed = reduce_code_fast(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
                
        seed += 1
        
    if best_code is None:
        best_code = greedy_construct_vectorized(0)
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
