import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Fallback algebraic construction
    def get_fallback():
        if n == 4:
            return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n > 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    if q != 3 or n != 8 or radius != 1:
        return get_fallback()

    total_words = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 1 + 8*2 = 17

    # Helper to convert int to base-q digits
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    # Precompute neighbors array using vectorized approach
    # For each word w, its neighbors are w itself and all words at Hamming distance 1
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    # Algebraic direct-sum construction: tetracode ⊕ tetracode
    # The optimal K_3(4,1) = 9 is the tetracode
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Convert tetracode words to integers
    def digits_to_int(digits, n, q):
        val = 0
        for i in range(n):
            val += digits[i] * (q ** i)
        return val

    tetracode_ints = [digits_to_int(t, 4, q) for t in tetracode]
    
    # Direct sum: for each pair of tetracode words, create an 8-word
    # Word = (t1[0], t1[1], t1[2], t1[3], t2[0], t2[1], t2[2], t2[3])
    # In integer form: t1 + t2 * q^4
    q4 = q ** 4
    direct_sum_code = [t1 + t2 * q4 for t1 in tetracode_ints for t2 in tetracode_ints]
    
    selected_indices = list(direct_sum_code)
    
    # Verify coverage
    covered = np.zeros(total_words, dtype=np.uint8)
    for idx in selected_indices:
        covered[neighbors[idx]] += 1
    
    if not covered.all():
        # Fallback to greedy if direct sum doesn't cover (shouldn't happen, but safety)
        return get_fallback()
    
    # Ball-splitting local search: try to remove codewords that are redundant
    # or replace pairs with a single better codeword
    best_size = len(selected_indices)
    improved = True
    
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try to remove redundant codewords
        for i in range(len(selected_indices)):
            if time.time() > deadline - 5.0:
                break
            idx = selected_indices[i]
            if np.all(covered[neighbors[idx]] > 1):
                # This codeword is redundant - all its neighbors are covered by others
                covered[neighbors[idx]] -= 1
                selected_indices.pop(i)
                best_size = len(selected_indices)
                improved = True
                break  # Restart loop with new state
        
        if not improved:
            # Try ball-splitting: find pairs (i, j) where removing both and adding
            # a single codeword c that covers the union of their unique contributions
            # is valid. This is expensive, so we do it sparsely.
            if random.random() < 0.3 and len(selected_indices) > 1:
                # Try a random pair
                i, j = random.sample(range(len(selected_indices)), 2)
                idx_i = selected_indices[i]
                idx_j = selected_indices[j]
                
                # Find points only covered by i or only by j
                # Points covered by i: neighbors[idx_i]
                # Points covered by j: neighbors[idx_j]
                # Points covered by both: intersection
                # Unique to i: covered by i but not by any other codeword except possibly j
                # Actually, we need: after removing i and j, are all points still covered?
                
                # Simulate removing i and j
                covered_sim = covered.copy()
                covered_sim[neighbors[idx_i]] -= 1
                covered_sim[neighbors[idx_j]] -= 1
                
                # Find uncovered points
                uncovered_mask = covered_sim == 0
                if not uncovered_mask.any():
                    # Both can be removed without replacement!
                    covered = covered_sim
                    selected_indices.pop(j)
                    selected_indices.pop(i)
                    best_size = len(selected_indices)
                    improved = True
                else:
                    # Try to find a single codeword c that covers all uncovered points
                    # The uncovered points must all be in the ball of c
                    # c's ball must contain all uncovered points
                    # So c must be within distance 1 of all uncovered points
                    # This means all uncovered points must be within distance 2 of each other
                    # (triangle inequality: if d(c, u) <= 1 and d(c, v) <= 1, then d(u, v) <= 2)
                    
                    # Check if all uncovered points are within distance 2 of each other
                    # If yes, find the center c
                    uncov_idx = np.where(uncovered_mask)[0]
                    if len(uncov_idx) <= 2:
                        # Check if there's a codeword c whose ball covers all uncov_idx
                        # c's ball = neighbors[c]
                        # We need uncov_idx ⊆ neighbors[c] for some c
                        # This is equivalent to: for each u in uncov_idx, c is in neighbors[u]
                        # So c must be in the intersection of neighbors[u] for all u in uncov_idx
                        
                        if len(uncov_idx) == 1:
                            # Any codeword in neighbors[uncov_idx[0]] works
                            candidates = neighbors[uncov_idx[0]]
                        else:
                            # Intersection of neighbors of both points
                            set1 = set(neighbors[uncov_idx[0]].tolist())
                            set2 = set(neighbors[uncov_idx[1]].tolist())
                            candidates = np.array(list(set1 & set2), dtype=np.int32)
                        
                        if len(candidates) > 0:
                            # Pick a candidate not already in the code
                            selected_set = set(selected_indices)
                            valid_candidates = [c for c in candidates if c not in selected_set]
                            if not valid_candidates:
                                valid_candidates = candidates.tolist()
                            
                            if valid_candidates:
                                c = valid_candidates[0]
                                # Verify: adding c covers all uncovered points
                                if covered_sim[neighbors[c]].any() or np.all(covered_sim[neighbors[c]] == 0):
                                    # Check that c covers all uncovered points
                                    if np.all(covered_sim[neighbors[c]] >= 0):  # Always true
                                        # Actually check: all uncov_idx are in neighbors[c]
                                        if all(u in set(neighbors[c].tolist()) for u in uncov_idx):
                                            # Add c, remove i and j
                                            covered = covered_sim
                                            covered[neighbors[c]] += 1
                                            selected_indices.pop(j)
                                            selected_indices.pop(i)
                                            selected_indices.append(c)
                                            best_size = len(selected_indices)
                                            improved = True
    
    # Final redundancy removal pass
    if time.time() < deadline - 2.0:
        for i in range(len(selected_indices) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break
            idx = selected_indices[i]
            if np.all(covered[neighbors[idx]] > 1):
                covered[neighbors[idx]] -= 1
                selected_indices.pop(i)
    
    codewords = [int_to_digits(idx, n, q) for idx in selected_indices]
    
    # Final verification
    if time.time() < deadline - 1.0:
        final_covered = np.zeros(total_words, dtype=np.uint8)
        for idx in selected_indices:
            final_covered[neighbors[idx]] += 1
        if not final_covered.all():
            # Coverage broken, return fallback
            return get_fallback()
    
    return codewords
