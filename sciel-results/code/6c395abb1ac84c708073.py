import numpy as np
import time
import random
from scipy.spatial.distance import cdist

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    # Objective function: Maximize sum of radii
    # We minimize: -sum(r) + Penalty
    # Penalty = sum(max(0, overlap_ij)^2) + sum(max(0, boundary_violation)^2)
    
    def evaluate(x):
        """
        x: flattened array of size 1 + 2*N + N = 3N + 1
        x[0]: aspect ratio a (w/h)
        x[1:1+2N]: centers (cx, cy) for N circles
        x[1+2N:]: radii r for N circles
        """
        # Unpack
        a = x[0]
        # Ensure aspect ratio is within reasonable bounds to avoid degenerate rectangles
        # Perimeter 4 => w+h=2. w = 2a/(1+a), h = 2/(1+a)
        # If a is very small, w is small, h is large.
        # Let's clamp a to [0.2, 5.0] effectively by penalizing out of bounds or just clamping in eval
        # Actually, better to keep a continuous and let the penalty handle it, 
        # but extreme a leads to tiny w or h, making packing impossible.
        
        # To handle the non-linearity of w,h from a, we compute them.
        # If a is negative or zero, it's invalid.
        if a <= 0:
            return 1e10
            
        w = 2.0 * a / (1.0 + a)
        h = 2.0 / (1.0 + a)
        
        centers = x[1:1+2*N].reshape(N, 2)
        radii = x[1+2*N:]
        
        # Ensure radii are non-negative for penalty calculation, though bounds will handle it
        # For penalty, we treat negative radii as 0 for overlap but they contribute -r to objective (bad)
        
        # Boundary Penalties
        # Circle i must satisfy:
        # r_i <= cx_i
        # r_i <= w - cx_i
        # r_i <= cy_i
        # r_i <= h - cy_i
        # Violation = max(0, r_i - dist_to_wall)
        
        # Left: r - cx
        # Right: r - (w - cx)
        # Bottom: r - cy
        # Top: r - (h - cy)
        
        v_left = np.maximum(0, radii - centers[:, 0])
        v_right = np.maximum(0, radii - (w - centers[:, 0]))
        v_bottom = np.maximum(0, radii - centers[:, 1])
        v_top = np.maximum(0, radii - (h - centers[:, 1]))
        
        boundary_penalty = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Pairwise Overlap Penalties
        # dist_ij >= r_i + r_j
        # Violation = max(0, r_i + r_j - dist_ij)
        
        dists = cdist(centers, centers)
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Zero out diagonal
        np.fill_diagonal(r_sums, 0)
        np.fill_diagonal(dists, 1e9) # Avoid division issues if we were doing ratios, but here just distance
        
        overlaps = np.maximum(0, r_sums - dists)
        overlap_penalty = np.sum(overlaps**2)
        
        # Objective: Minimize -sum(radii) + Large_Penalty * (boundary_penalty + overlap_penalty)
        # We want sum(radii) to be maximized.
        # If there is any violation, penalty dominates.
        
        total_penalty = boundary_penalty + overlap_penalty
        penalty_weight = 10.0 # Needs to be strong enough to prevent overlap
        
        score = -np.sum(radii) + penalty_weight * total_penalty
        
        return score

    def sa_step(x, T, step_size):
        """
        Perform one step of Simulated Annealing.
        """
        x_new = x.copy()
        
        # Perturb Aspect Ratio
        x_new[0] += np.random.normal(0, step_size * 0.1)
        if x_new[0] < 0.1: x_new[0] = 0.1
        if x_new[0] > 10.0: x_new[0] = 10.0
        
        # Perturb Centers
        # Scale step size by current dimensions? 
        # For simplicity, use fixed relative step or absolute.
        # Let's use a step relative to the box size (approx 1.0)
        center_noise = np.random.normal(0, step_size * 0.05, (N, 2))
        x_new[1:1+2*N] += center_noise.flatten()
        
        # Perturb Radii
        radius_noise = np.random.normal(0, step_size * 0.02, N)
        x_new[1+2*N:] += radius_noise
        
        # Clamp radii to be non-negative and not larger than min(w,h)/2 (roughly)
        # We don't know w,h here easily without recalculating, but we can clamp to [0, 1.0]
        # The penalty will push them down if they are too big.
        x_new[1+2*N:] = np.maximum(0, x_new[1+2:N])
        
        return x_new

    def run_sa(x_init, time_limit):
        """
        Run Simulated Annealing until time_limit is reached.
        """
        current_x = x_init.copy()
        current_score = evaluate(current_x)
        
        best_x = current_x.copy()
        best_score = current_score
        
        T = 1.0
        T_min = 1e-4
        cooling_rate = 0.9995
        
        # Initial step size
        step_size = 0.5
        
        while time.time() < time_limit:
            # Batch of steps to reduce function call overhead? 
            # evaluate is vectorized mostly, but cdist is O(N^2). 
            # N=21, so 441 distances. Very fast.
            # We can do many steps.
            
            for _ in range(100):
                x_new = sa_step(current_x, T, step_size)
                score_new = evaluate(x_new)
                
                # Accept if better or with probability exp(-delta/T)
                delta = score_new - current_score
                if delta < 0:
                    accept = True
                else:
                    # If T is very small, exp(-large) is 0
                    if T > T_min:
                        accept = np.random.random() < np.exp(-delta / T)
                    else:
                        accept = False
                
                if accept:
                    current_x = x_new
                    current_score = score_new
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_x = current_x.copy()
                
                # Cool down
                T *= cooling_rate
                if T < T_min:
                    T = T_min
            
            # Adjust step size occasionally? 
            # If we are stuck, maybe increase step size? 
            # Or just let it cool.
            
            # Check time more frequently
            if time.time() > time_limit:
                break

        return best_x, best_score

    # Main Logic
    random.seed(42)
    np.random.seed(42)
    
    best_global_x = None
    best_global_score = 1e10
    
    # Define a few initial guesses for aspect ratios
    # Square (1.0), Wide (2.0), Tall (0.5)
    aspects_to_try = [1.0, 1.5, 0.7, 2.0, 0.5]
    
    time_per_run = (deadline - start_time) / len(aspects_to_try)
    
    for a_init in aspects_to_try:
        if time.time() > deadline:
            break
            
        # Initialize x
        # x = [a, cx1, cy1, ..., cxN, cyN, r1, ..., rN]
        
        # Start with a grid-like layout for the specific aspect
        w = 2.0 * a_init / (1.0 + a_init)
        h = 2.0 / (1.0 + a_init)
        
        # Grid initialization
        # 5x4 grid = 20, plus 1? Or 4x5?
        # Let's just scatter randomly but within bounds to avoid immediate huge penalties
        # Or use a grid.
        # Grid 5 cols, 4 rows (20) + 1 center?
        # Let's just use random uniform for centers within [0.1, w-0.1] and [0.1, h-0.1]
        # And radii uniform [0.05, 0.1]
        
        centers_init = np.random.uniform(0.1, max(0.11, w-0.1), (N, 2))
        centers_init[:, 1] = np.random.uniform(0.1, max(0.11, h-0.1), N)
        radii_init = np.random.uniform(0.05, 0.1, N)
        
        x_init = np.concatenate([[a_init], centers_init.flatten(), radii_init])
        
        # Run SA
        x_res, score_res = run_sa(x_init, time.time() + time_per_run)
        
        if score_res < best_global_score:
            best_global_score = score_res
            best_global_x = x_res

    # If SA didn't find a valid solution (score is high due to penalty), 
    # we might need a fallback or just return the best found.
    # The evaluator checks validity. If invalid, score is 0.
    # We should try to "repair" the best solution if it has small violations.
    
    if best_global_x is None:
        # Fallback: Tiny circles in square
        w, h = 1.0, 1.0
        res = np.zeros((N, 3))
        r = 0.05
        for i in range(N):
            row = i // 5
            col = i % 5
            res[i, 0] = (col + 0.5) * 0.2
            res[i, 1] = (row + 0.5) * 0.2
            res[i, 2] = r
        return res

    # Extract result
    a = best_global_x[0]
    w = 2.0 * a / (1.0 + a)
    h = 2.0 / (1.0 + a)
    
    centers = best_global_x[1:1+2*N].reshape(N, 2)
    radii = best_global_x[1+2*N:]
    
    # Final Check and Repair
    # If there are small overlaps, we can shrink radii slightly to make it valid.
    # The task requires disjoint interiors.
    # Let's verify and adjust.
    
    def get_valid_result(centers, radii, w, h):
        # Check boundaries
        # Shrink radii to fit boundaries
        for i in range(N):
            # Distance to walls
            d_left = centers[i, 0]
            d_right = w - centers[i, 0]
            d_bottom = centers[i, 1]
            d_top = h - centers[i, 1]
            max_r = min(d_left, d_right, d_bottom, d_top)
            if radii[i] > max_r:
                radii[i] = max_r * 0.999 # Slight shrink for safety
        
        # Check pairwise overlaps
        # If overlap, shrink the smaller radius? Or both?
        # Simple iterative relaxation
        for _ in range(10):
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    sum_r = radii[i] + radii[j]
                    if sum_r > dist:
                        # Overlap exists
                        # Shrink both proportionally to their size
                        # Or just shrink the one with more "room"?
                        # Simple: shrink both by half the overlap
                        overlap = sum_r - dist
                        # Ensure we don't go negative
                        shrink_i = overlap * 0.5
                        shrink_j = overlap * 0.5
                        
                        # Check if we can shrink i
                        if radii[i] > shrink_i:
                            radii[i] -= shrink_i
                        else:
                            # i is too small, shrink j more
                            radii[j] -= (shrink_i - radii[i]) + shrink_j
                            radii[i] = 0 # or max(0, radii[i] - shrink_i)
                        
                        if radii[j] < 0: radii[j] = 0
                        if radii[i] < 0: radii[i] = 0

        # Final validity check
        # If still invalid, return tiny circles
        # But this heuristic should work for "small" overlaps.
        
        # Construct output array
        res = np.column_stack([centers, radii])
        
        # Final strict check
        # Boundary
        if np.any(centers[:, 0] < radii - 1e-5) or np.any(centers[:, 0] > w - radii + 1e-5):
             # Shrink to fit
             for i in range(N):
                 max_r = min(centers[i, 0], w - centers[i, 0], centers[i, 1], h - centers[i, 1])
                 if radii[i] > max_r:
                     radii[i] = max_r
                     res[i, 2] = radii[i]
        
        if np.any(centers[:, 1] < radii - 1e-5) or np.any(centers[:, 1] > h - radii + 1e-5):
             for i in range(N):
                 max_r = min(centers[i, 0], w - centers[i, 0], centers[i, 1], h - centers[i, 1])
                 if radii[i] > max_r:
                     radii[i] = max_r
                     res[i, 2] = radii[i]
                     
        # Pairwise
        dists = cdist(centers, centers)
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        np.fill_diagonal(r_sums, 0)
        overlaps = r_sums - dists
        if np.any(overlaps > 1e-5):
             # Still overlapping. Force shrink.
             # This is a last resort.
             for i in range(N):
                 for j in range(i+1, N):
                     if overlaps[i, j] > 1e-5:
                         # Shrink j
                         radii[j] -= overlaps[i, j]
                         res[j, 2] = radii[j]
                         if radii[j] < 0: radii[j] = 0
                         # Update overlap matrix? No, just accept potential residual if complex.
                         # But for N=21, this loop is fast.
                         # Let's do a few more passes
             # Re-check
             for _ in range(5):
                 dists = cdist(centers, centers)
                 r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
                 np.fill_diagonal(r_sums, 0)
                 overlaps = r_sums - dists
                 if np.all(overlaps <= 1e-5):
                     break
                 for i in range(N):
                     for j in range(i+1, N):
                         if overlaps[i, j] > 1e-5:
                             # Shrink the one with larger radius? Or split?
                             # Splitting is safer for validity
                             shrink = overlaps[i, j] * 0.5 + 1e-6
                             if radii[i] > shrink:
                                radii[i] -= shrink
                                res[i, 2] = radii[i]
                             else:
                                radii[i] = 0
                                res[i, 2] = 0
                                if radii[j] > shrink:
                                    radii[j] -= shrink
                                    res[j, 2] = radii[j]
                             if radii[j] > shrink:
                                radii[j] -= shrink
                                res[j, 2] = radii[j]
                             else:
                                radii[j] = 0
                                res[j, 2] = 0
                                

        return res

    result = get_valid_result(centers, radii, w, h)
    
    # One final check: if the sum of radii is extremely small (e.g., < 0.1), 
    # it might be a failed run. 
    # But we return it anyway as it's valid.
    
    return result

