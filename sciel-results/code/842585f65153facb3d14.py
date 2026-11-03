import itertools
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute coverage mask: (total_words, total_words)
        # mask[i, j] = 1 if dist(i, j) <= radius
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        coverage_mask = (dist <= radius).astype(np.uint8)
        
        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 115
        
        # Lagrangian Relaxation Parameters
        lambdas = np.zeros(total_words)
        best_lagrangian_code = None
        
        # Run Lagrangian for a fixed number of iterations or time
        # 1000 iterations is fast with NumPy
        for iteration in range(2000):
            if time.time() - start_time > time_limit - 10: # Leave 10s for LS
                break
                
            # 1. Determine primal solution
            # sums[c] = sum(lambdas[i] for i in N(c))
            sums = coverage_mask.T @ lambdas
            
            # Select if sums > 1.0
            # Using 1.0 + epsilon to avoid floating point issues?
            # Or just > 1.0
            selected = (sums > 1.0).astype(np.uint8)
            
            # Check validity
            # covered = coverage_mask @ selected
            covered = coverage_mask @ selected
            is_valid = np.all(covered > 0)
            
            num_selected = int(np.sum(selected))
            
            if is_valid and num_selected < best_size:
                best_size = num_selected
                best_code = np.where(selected)[0].tolist()
                
            # 2. Update lambdas
            # grad[i] = 1 - covered[i]
            grad = 1.0 - covered
            
            # Step size
            # Use a simple decaying step size
            # alpha = 0.5
            # step = alpha * (num_selected - 71) / (np.sum(grad**2) + 1e-6)
            
            # To improve performance, we can just use a fixed small step if we are far from LB
            # and smaller if close.
            if num_selected > 71:
                norm_sq = np.sum(grad**2)
                if norm_sq > 0:
                    step = 0.5 * (num_selected - 71) / norm_sq
                else:
                    step = 0.0
            else:
                step = 0.0
                
            lambdas = np.maximum(0, lambdas + step * grad)
            
            # Add noise to escape local minima
            if iteration % 50 == 0:
                lambdas += np.random.uniform(0, 0.05, size=total_words)
                lambdas = np.maximum(0, lambdas)

        # If we found a code, try to improve it with Local Search
        if best_code is not None:
            # Convert to set for fast operations
            code_set = set(best_code)
            
            # Precompute coverage lists for the current code set?
            # Or just use the mask.
            
            # Local Search: Pruning and Swapping
            # We need to check if removing a word breaks coverage.
            # counts[i] = number of codewords covering i
            
            counts = coverage_mask @ np.array([1 if c in code_set else 0 for c in range(total_words)], dtype=np.uint8)
            
            improved = True
            ls_iterations = 0
            while improved and ls_iterations < 100 and time.time() - start_time < time_limit:
                improved = False
                ls_iterations += 1
                
                # 1. Pruning
                # Find a codeword c such that for all i in N(c), counts[i] > 1
                # This means c is redundant.
                # For each c in code_set:
                #   if min(counts[N(c)]) > 1: remove c
                
                # Vectorized check for pruning:
                # For each c, we want to know if min(counts[coverage_mask[:, c]]) > 1
                # This is hard to vectorize directly without loops.
                # Loop over code_set (size ~73) is fast.
                
                removable = -1
                for c in code_set:
                    # Get counts of points covered by c
                    # mask_col = coverage_mask[:, c]
                    # covered_counts = counts[mask_col]
                    # if np.min(covered_counts) > 1:
                    #     removable = c
                    #     break
                    # 
                    # Optimization: Only check if c is "potentially" removable?
                    # Just check all. 73 * 729 ops is small.
                    covered_counts = counts[coverage_mask[:, c] == 1]
                    if len(covered_counts) > 0 and np.min(covered_counts) > 1:
                        removable = c
                        break
                        
                if removable != -1:
                    code_set.remove(removable)
                    # Update counts
                    counts[coverage_mask[:, removable] == 1] -= 1
                    improved = True
                    continue
                    
                # 2. Swapping
                # Try to swap c with w.
                # Condition: w covers all private points of c.
                # Private points of c: P_c = { i | counts[i] == 1 and i in N(c) }
                # We need w such that N(w) covers P_c.
                
                # Iterate over c in code_set
                swapped = False
                # Randomize order to avoid bias
                code_list = list(code_set)
                np.random.shuffle(code_list)
                
                for c in code_list:
                    # Find private points
                    mask_c = coverage_mask[:, c] == 1
                    private_mask = (counts == 1) & mask_c
                    private_points = np.where(private_mask)[0]
                    
                    if len(private_points) == 0:
                        # No private points, can remove? 
                        # Wait, if no private points, it's redundant (handled by pruning).
                        # But pruning checks min > 1. If min == 1, it's not redundant.
                        # If private_points is empty, it means all covered points have count > 1?
                        # No, private_mask is count==1 AND covered by c.
                        # If empty, c has no unique coverage. It is redundant.
                        # Why didn't pruning catch it?
                        # Pruning checks min(counts[N(c)]) > 1.
                        # If min is 1, there is a private point.
                        # So if private_points is empty, min must be > 1.
                        # So this case should be caught by pruning.
                        continue
                        
                    # Find w not in code_set such that w covers all private_points
                    # For each w not in code_set:
                    #   if all(private_points in N(w)):
                    
                    # To speed up, we can check if there exists such w.
                    # Intersection of neighborhoods of private points?
                    # If we intersect N(p) for all p in private_points, we get candidates.
                    # Since private_points is small (usually 1-3), this is fast.
                    
                    candidates = np.ones(total_words, dtype=bool)
                    for p in private_points:
                        candidates &= (coverage_mask[p, :] == 1)
                        
                    # Exclude current code_set
                    for c2 in code_set:
                        candidates[c2] = False
                        
                    # Pick first candidate
                    w_candidates = np.where(candidates)[0]
                    if len(w_candidates) > 0:
                        w = w_candidates[0]
                        # Perform swap
                        code_set.remove(c)
                        code_set.add(w)
                        
                        # Update counts
                        # Remove c's contribution
                        counts[coverage_mask[:, c] == 1] -= 1
                        # Add w's contribution
                        counts[coverage_mask[:, w] == 1] += 1
                        
                        improved = True
                        swapped = True
                        break
                        
                if not swapped:
                    break
                    
            # Update best if improved
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)

        # If still no code (shouldn't happen), fallback
        if best_code is None:
             # Simple greedy
             uncovered = set(range(total_words))
             code_indices = []
             while uncovered:
                 best_idx = -1
                 best_count = -1
                 for i in range(total_words):
                     count = np.sum(coverage_mask[i, :] & [1 if j in uncovered else 0 for j in range(total_words)])
                     if count > best_count:
                         best_count = count
                         best_idx = i
                 if best_idx == -1: break
                 code_indices.append(best_idx)
                 # Update uncovered
                 covered_by_new = np.where(coverage_mask[best_idx, :] == 1)[0]
                 for c in covered_by_new:
                     if c in uncovered:
                         uncovered.remove(c)
             best_code = code_indices

        return [list(all_words[i]) for i in best_code]
    
    else:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        coverage = [set() for _ in range(total_words)]
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
