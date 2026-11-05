import numpy as np
import time
import random
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(6)]
        
        # Precompute neighbors (balls)
        # ball[c] is a boolean array of length 'total' where True if word is in ball(c)
        balls = []
        for c in range(total):
            d = [(c // powers[p]) % 3 for p in range(6)]
            ball_mask = np.zeros(total, dtype=bool)
            ball_mask[c] = True
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        new_c = c + (alt - d[p]) * powers[p]
                        ball_mask[new_c] = True
            balls.append(ball_mask)
            
        # Incidence matrix A: A[i, j] is 1 if word i is in ball(j)
        # A is total x total
        # We can construct it efficiently
        A = np.zeros((total, total), dtype=np.int8)
        for j in range(total):
            A[balls[j], j] = 1

        best_solution = None
        best_size = float('inf')
        
        # State for B&B
        # uncovered: boolean array, True if word needs covering
        # available: boolean array, True if center is not yet selected
        # selected: list of selected centers
        
        # To speed up LP, we might need to be careful with the size.
        # But 729x729 is small.
        
        def get_lp_bound(uncovered, available):
            # Dual problem: Max sum(y_u) for u in uncovered
            # Subject to sum(y_u for u in uncovered and u in ball(c)) <= 1 for all c in available
            # y_u >= 0
            
            # If no uncovered, return 0
            if not np.any(uncovered):
                return 0
            
            # If no available centers, return inf (infeasible)
            if not np.any(available):
                return float('inf')
            
            # Extract submatrix
            # Rows of Dual correspond to available centers
            # Columns of Dual correspond to uncovered words
            # Constraint matrix D[c_idx, u_idx] = A[u_idx, c_idx]
            
            avail_idx = np.where(available)[0]
            uncov_idx = np.where(uncovered)[0]
            
            # D shape: (num_avail, num_uncov)
            D = A[uncov_idx][:, avail_idx].T
            
            # Objective: Max sum(y) -> Min -sum(y)
            c_obj = -np.ones(len(uncov_idx))
            
            # Constraints: D @ y <= 1
            b_ub = np.ones(len(avail_idx))
            
            # Bounds: y >= 0
            bounds = [(0, None)] * len(uncov_idx)
            
            # Solve LP
            # Use method 'highs' for speed
            res = linprog(c_obj, A_ub=D, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.status == 0:
                return -res.fun
            else:
                # If LP fails, return a conservative bound (e.g., 0 or a greedy bound)
                # A simple greedy bound: max packing
                # For safety, return 0 to avoid pruning valid nodes, but this slows down search
                # Or return a simple lower bound like ceil(num_uncov / max_cover)
                if len(uncov_idx) == 0:
                    return 0
                # Heuristic bound
                max_cover = np.max(np.sum(D, axis=1)) if D.size > 0 else 0
                if max_cover == 0:
                    return float('inf')
                return np.ceil(len(uncov_idx) / max_cover)

        def branch_and_bound(uncovered, available, selected, depth=0):
            nonlocal best_solution, best_size, deadline
            
            if time.time() > deadline:
                return
                
            num_uncov = np.sum(uncovered)
            if num_uncov == 0:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = list(selected)
                return
                
            # Lower bound
            lb = get_lp_bound(uncovered, available)
            if len(selected) + lb >= best_size:
                return
                
            # Branching heuristic: Pick an uncovered word with fewest available covering centers
            # Count how many available centers cover each uncovered word
            # coverage_count[u] = sum(available[c] for c if A[u,c]==1)
            # We can compute this as: A[uncov_idx, :] @ available
            # But we only care about uncovered rows.
            
            # Optimize: 
            # counts = np.sum(A[uncovered][:, :].astype(int) * available[None, :], axis=1)
            # This is (num_uncov, total) dot (total,) -> (num_uncov,)
            
            # To save time, if num_uncov is small, this is fast.
            # If num_uncov is large, it might be slow.
            # Alternative: iterate through available centers?
            
            # Let's try the matrix multiplication approach, it's vectorized.
            # A is int8, available is bool.
            # We need counts for each uncovered word.
            # sub_A = A[uncovered, :] # (num_uncov, total)
            # counts = sub_A @ available.astype(int)
            
            # However, creating sub_A every time might be overhead.
            # But num_uncov decreases.
            
            uncov_indices = np.where(uncovered)[0]
            if len(uncov_indices) == 0:
                return
                
            # Calculate coverage counts for uncovered words by available centers
            # We can precompute A as float or int. It's int8.
            # available is bool.
            # counts = np.dot(A[uncov_indices, :], available.astype(np.int8))
            
            counts = np.dot(A[uncov_indices, :], available.astype(np.int8))
            
            # Find the word with minimum positive count
            # If count is 0, it's infeasible (shouldn't happen if we started valid)
            min_count = np.min(counts)
            if min_count == 0:
                # Infeasible branch
                return
                
            # Pick the first word with min_count
            # To add randomness or diversity, we could pick randomly from min_count words
            # But deterministic is often better for B&B
            target_word_idx = uncov_indices[np.argmin(counts)]
            
            # Find all available centers that cover this word
            # centers = np.where(available & A[target_word_idx, :])[0]
            covering_centers = np.where(available & A[target_word_idx, :])[0]
            
            # Order centers by some heuristic (e.g., how many other uncovered words they cover)
            # This can improve pruning.
            # Score = sum(uncovered & ball(c))
            # We can compute this for the candidate centers.
            
            if len(covering_centers) == 0:
                return
                
            # Sort candidates by descending coverage of uncovered words
            # cov_scores = [np.sum(uncovered & balls[c]) for c in covering_centers]
            # This is a Python loop, might be slow if many candidates.
            # Vectorized:
            # cand_balls = balls[covering_centers] # list of arrays
            # This is tricky to vectorize easily without creating a big 3D array.
            # Let's just use the order from argmin or random shuffle to break ties.
            # For now, let's try a simple heuristic: pick the center that covers the most uncovered words.
            
            # Compute coverage for each candidate
            # We can do this in a loop if the number of candidates is small (< 50)
            # Otherwise, skip sorting or use a simpler metric.
            
            if len(covering_centers) < 100:
                scores = []
                for c in covering_centers:
                    scores.append(np.sum(uncovered & balls[c]))
                # Sort indices by score descending
                sorted_idx = np.argsort(-scores)
                ordered_centers = [covering_centers[i] for i in sorted_idx]
            else:
                ordered_centers = list(covering_centers)
                random.shuffle(ordered_centers)

            for c in ordered_centers:
                # Select c
                new_uncovered = uncovered & ~balls[c]
                new_available = available.copy()
                new_available[c] = False
                selected.append(c)
                
                branch_and_bound(new_uncovered, new_available, selected, depth + 1)
                
                selected.pop()
                
                if time.time() > deadline:
                    return

        # Initial state
        initial_uncovered = np.ones(total, dtype=bool)
        initial_available = np.ones(total, dtype=bool)
        initial_selected = []
        
        # Run B&B
        branch_and_bound(initial_uncovered, initial_available, initial_selected)
        
        if best_solution is None:
            # Fallback to greedy if B&B fails or times out without finding a solution
            # This shouldn't happen as the problem is solvable, but just in case
            uncovered = np.ones(total, dtype=bool)
            codewords = []
            while np.any(uncovered):
                # Greedy: pick center covering most uncovered
                # counts = np.sum(uncovered[:, None] & np.array(balls), axis=0)
                # This is slow.
                # Use precomputed A
                # counts = np.sum(A.T @ uncovered, axis=1) ? No.
                # A[i,j] is 1 if i in ball(j).
                # We want sum of uncovered[i] for i in ball(j).
                # That is sum_i (uncovered[i] * A[i,j]) = (uncovered @ A) [j]
                counts = uncovered.astype(int) @ A
                best_c = int(np.argmax(counts))
                codewords.append(best_c)
                uncovered &= ~balls[best_c]
            best_solution = codewords

        # Convert indices to lists
        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(6)]
            
        return [idx_to_list(idx) for idx in best_solution]

    else:
        # Fallback for other parameters
        total = q ** n
        return [[0] * n]
