import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.arange(1, n + 1) / n  # shape (64,)
    
    # Keich seed construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    seed_positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = np.array(list(str(bin(i))[2:]), dtype=np.int8)
        bits = np.pad(bits, (k - len(bits), 0), constant_values=0)
        seed_positions[i] = np.sum((1.0 - j) * bits * (2.0 ** -j) / k)

    # Precompute grids for objective evaluation
    # We use a moderately high resolution grid for the optimization to ensure fidelity
    num_h = 2000
    ys = (np.arange(num_h) + 0.5) / num_h
    ws = (1.0 - ys) / n

    def union_area_exact(x):
        """
        Computes the exact area using the min() function as per the task definition.
        Used for final validation and coordinate descent.
        """
        Ls = x[None, :] + a[None, :] * ys[:, None]
        sorted_Ls = np.sort(Ls, axis=1)
        diffs = np.diff(sorted_Ls, axis=1)
        extensions = np.minimum(ws[:, None], diffs)
        union_lens = ws + extensions.sum(axis=1)
        return union_lens.mean()

    def union_area_smooth(x, temp):
        """
        Computes a smoothed version of the area using a soft-min approximation.
        min(a, b) approximated by -1/beta * logsumexp(-beta*a, -beta*b).
        Here we want min(d, w).
        softmin(d, w) = -1/beta * log(exp(-beta*d) + exp(-beta*w))
        To avoid overflow issues, we use the log-sum-exp trick.
        """
        Ls = x[None, :] + a[None, :] * ys[:, None]  # (num_h, n)
        sorted_Ls = np.sort(Ls, axis=1)             # (num_h, n)
        diffs = np.diff(sorted_Ls, axis=1)          # (num_h, n-1)
        
        # We need to sum min(d, w) over the n-1 gaps.
        # Let M = min(d, w).
        # softmin(d, w) = -1/beta * log(exp(-beta*d) + exp(-beta*w))
        # To stabilize, let max_term = max(-beta*d, -beta*w)
        # softmin = max_term/beta - log(exp(-beta*d - max_term) + exp(-beta*w - max_term))
        
        beta = 1.0 / temp
        
        # Vectorized softmin calculation
        # diffs shape (num_h, n-1), ws[:, None] shape (num_h, 1)
        d = diffs
        w = ws[:, None]
        
        # Calculate -beta * d and -beta * w
        val_d = -beta * d
        val_w = -beta * w
        
        # max_val = max(val_d, val_w)
        max_val = np.maximum(val_d, val_w)
        
        # sum_exp = exp(val_d - max_val) + exp(val_w - max_val)
        # Note: exp(0)=1, so one term will always be 1.
        sum_exp = np.exp(val_d - max_val) + np.exp(val_w - max_val)
        
        soft_min_val = (max_val - np.log(sum_exp)) / beta
        
        union_lens = ws + soft_min_val.sum(axis=1)
        return union_lens.mean()

    start_time = time.time()
    max_time = 625.0  # 5s safety margin

    best_x = seed_positions.copy()
    # Evaluate best_x on the EXACT objective to have a correct baseline
    best_area = union_area_exact(best_x)

    np.random.seed(42)

    # Generate restart points
    restarts = [seed_positions.copy()]
    # Structured perturbations: shift by small amounts, perhaps correlated
    for _ in range(10):
        restarts.append(seed_positions + np.random.normal(0, 0.05, n))
    # Random uniform restarts
    for _ in range(5):
        restarts.append(np.random.uniform(-0.5, 1.5, n))

    # Optimization loop with annealing
    # We will run L-BFGS-B on the smooth objective.
    # We start with a relatively high temperature to smooth out the non-differentiable kinks.
    temps = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    current_x = best_x.copy()
    
    for temp in temps:
        if time.time() - start_time > max_time - 60:
            break
            
        # Define objective for this temperature
        def obj_smooth(x):
            return union_area_smooth(x, temp)
            
        # Try to improve current_x
        try:
            res = minimize(obj_smooth, current_x, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'disp': False})
            if res.fun < union_area_smooth(current_x, temp) or res.fun < union_area_smooth(best_x, temp):
                # Check if this improvement holds on the exact metric (approximate check via smooth)
                # We always accept if smooth score improves, but we track the exact score of the best found so far.
                candidate_x = res.x
                # Evaluate candidate on exact metric to update global best
                candidate_exact_area = union_area_exact(candidate_x)
                if candidate_exact_area < best_area:
                    best_area = candidate_exact_area
                    best_x = candidate_x
                # Update current_x for next stage, even if it didn't beat global best on exact,
                # as it might be a better starting point for the next sharper temp
                # Actually, it's safer to stick to the best exact found so far if we have time,
                # but using the result of the optimizer as the next start is standard.
                current_x = candidate_x
        except Exception:
            pass

    # If we still have time, do a few more random restarts with low temp
    if time.time() - start_time < max_time - 30:
        for i in range(5):
            if time.time() - start_time > max_time - 15:
                break
            x0 = seed_positions + np.random.normal(0, 0.01, n)
            def obj_smooth_low(x):
                return union_area_smooth(x, 0.001)
            try:
                res = minimize(obj_smooth_low, x0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'disp': False})
                candidate_exact_area = union_area_exact(res.x)
                if candidate_exact_area < best_area:
                    best_area = candidate_exact_area
                    best_x = res.x
            except Exception:
                pass

    # Final Polish: Coordinate Descent on the EXACT objective
    # This ensures we are at a local minimum of the actual metric.
    if time.time() - start_time < max_time - 5:
        x = best_x.copy()
        # Use a very small step for the final polish
        step = 1e-4
        # We don't need many iterations if L-BFGS already got us close
        for _ in range(50):
            if time.time() - start_time > max_time - 2:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > max_time - 2:
                    break
                old_val = union_area_exact(x)
                # Try small perturbations
                # We can use a simpler finite difference check or just fixed small steps
                deltas = [step, -step]
                best_delta = 0.0
                best_val = old_val
                for d in deltas:
                    x[i] += d
                    new_val = union_area_exact(x)
                    x[i] -= d
                    if new_val < best_val - 1e-15:
                        best_val = new_val
                        best_delta = d
                if best_delta != 0.0:
                    x[i] += best_delta
                    improved = True
            if not improved:
                break
            
    final_x = best_x
    final_area = union_area_exact(final_x)
    if union_area_exact(x) < final_area:
        final_x = x
    
    return np.array(final_x, dtype=np.float64)
