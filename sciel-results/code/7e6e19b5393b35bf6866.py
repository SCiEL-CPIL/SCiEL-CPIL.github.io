import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0

    def seed_solution():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if n != 8 or q != 3 or radius != 1:
        return seed_solution()

    try:
        # Total points
        total_points = 3 ** n
        
        # Generate all codewords
        all_codewords = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int64)
        
        # Precompute balls for all codewords
        # ball[i] is the list of point indices covered by codeword i
        # Point index is base-3 integer representation
        # We need to map a vector of digits to an index
        # index = sum(digit[i] * 3^(n-1-i))
        
        # Precompute powers of 3
        powers = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        
        # Compute index for each codeword
        # all_codewords is (6561, 8)
        # indices = all_codewords @ powers
        cw_indices = (all_codewords @ powers).astype(np.int64)
        
        # For each codeword, generate the 17 points it covers
        # Center: cw_indices[i]
        # Neighbors: change one position to the other 2 values
        balls = np.zeros((total_points, 17), dtype=np.int64)
        balls[:, 0] = cw_indices
        
        # For each position pos in 0..n-1
        for pos in range(n):
            # The change in index is (new_val - old_val) * 3^(n-1-pos)
            # We can precompute the deltas
            # For a given codeword i, old_val = all_codewords[i, pos]
            # new_val can be (old_val + 1) % 3 and (old_val + 2) % 3
            # delta1 = ((old_val + 1) % 3 - old_val) * powers[pos]
            # delta2 = ((old_val + 2) % 3 - old_val) * powers[pos]
            
            old_vals = all_codewords[:, pos]
            new_vals_1 = (old_vals + 1) % 3
            new_vals_2 = (old_vals + 2) % 3
            
            deltas_1 = (new_vals_1 - old_vals) * powers[pos]
            deltas_2 = (new_vals_2 - old_vals) * powers[pos]
            
            balls[:, 2*pos + 1] = cw_indices + deltas_1
            balls[:, 2*pos + 2] = cw_indices + deltas_2
            
        # Note: The indices in balls might need to be modulo total_points?
        # No, the index calculation is unique for each vector in {0,1,2}^n.
        # Since we are within the space, the resulting index will be in [0, 3^n - 1].
        
        def greedy_solve(noise_scale=0.0):
            covered = np.zeros(total_points, dtype=bool)
            selected = []
            
            # Pre-allocate gains array? No, it changes.
            # Loop until all covered
            # Max iterations is total_points (if we pick 1 point each time, but we pick 17)
            # So max ~ 400 iterations.
            
            for _ in range(total_points):
                if np.sum(covered) == total_points:
                    break
                
                # Calculate gains for all candidates
                # ~covered is boolean array of size 6561
                # balls is (6561, 17)
                # covered[balls] is (6561, 17) boolean
                # ~covered[balls] is (6561, 17) boolean
                # sum along axis 1
                gains = np.sum(~covered[balls], axis=1)
                
                if noise_scale > 0:
                    gains = gains + np.random.random(total_points) * noise_scale
                
                best_idx = np.argmax(gains)
                best_gain = gains[best_idx]
                
                if best_gain == 0:
                    break # Should not happen if not fully covered
                
                selected.append(best_idx)
                covered[balls[best_idx]] = True
                
                # Time check
                if time.time() > deadline - 10:
                    break
            
            # Local search: try to remove codewords
            # Build cover_count
            cover_count = np.zeros(total_points, dtype=np.int32)
            for idx in selected:
                cover_count[balls[idx]] += 1
            
            # Try to remove
            # We iterate multiple times because removing one might enable removing another
            # Or we can just do one pass in reverse order?
            # Let's do a loop until no more removals are possible
            max_passes = 5
            for _ in range(max_passes):
                removed_any = False
                # Iterate in reverse to maintain some stability or random order?
                # Reverse is fine.
                for i in range(len(selected) - 1, -1, -1):
                    idx = selected[i]
                    # Check if any point in balls[idx] has count 1
                    # If min count is 1, it's essential
                    # If min count > 1, it's removable
                    if np.min(cover_count[balls[idx]]) > 1:
                        # Remove it
                        cover_count[balls[idx]] -= 1
                        selected.pop(i)
                        removed_any = True
                
                if not removed_any:
                    break
            
            return selected

        best_solution = None
        best_size = float('inf')
        
        # Run multiple times with different noise
        # The first run can be deterministic (noise=0)
        runs = 0
        while time.time() < deadline - 20:
            noise = 0.0 if runs == 0 else 0.1
            sol = greedy_solve(noise)
            if sol:
                size = len(sol)
                if size < best_size:
                    best_size = size
                    best_solution = sol
            runs += 1
            # If we found a solution close to lower bound, we might stop early?
            # Lower bound is 386.
            if best_size <= 400:
                break
                
        if best_solution:
            # Convert indices back to codewords
            # all_codewords is the array of all codewords
            # best_solution is a list of indices into all_codewords
            return [list(all_codewords[i]) for i in best_solution]
        else:
            return seed_solution()

    except Exception:
        return seed_solution()
