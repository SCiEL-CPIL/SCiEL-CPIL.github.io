"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # We will explore a range of aspect ratios.
    # Hexagonal packing is densest in infinite plane, but boundary effects matter.
    # Square (1x1) is a strong candidate. Slightly rectangular might help fit rows better.
    # Based on inspiration, broader search (0.75-2.2) might help.
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.75, 1.3, 0.95, 1.05, 0.85, 1.15]
    
    # Pre-compute pairs for overlap constraints
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        # Run optimization with a few different seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initial guess: Grid-like distribution or random
            rng = np.random.RandomState(seed)
            
            # Random initialization is often better for escaping local minima in non-convex problems
            # Ensure initial points are well separated to avoid immediate heavy penalty
            cx = rng.uniform(0.1, 0.9, N) * w
            cy = rng.uniform(0.1, 0.9, N) * h
            r = np.full(N, 0.05 * min(w, h))
            
            x0 = np.concatenate([cx, cy, r])
            
            # Bounds
            r_max = min(w, h) / 2.0
            bounds = [(0.0, w)] * N + [(0.0, h)] * N + [(0.0, r_max)] * N
            
            # Successive Convex Relaxation / Penalty Method
            # We minimize: -sum(r) + lambda * sum( max(0, violation)^2 )
            
            # To make it faster, we use L-BFGS-B which handles bounds natively.
            # We only need to provide the gradient of the objective + penalty.
            
            # Start with a moderate penalty
            penalty = 10.0
            x_curr = x0.copy()
            
            # Initial validation and score
            def validate_and_score(x):
                cx_v = x[:N]
                cy_v = x[N:2*N]
                r_v = x[2*N:]
                
                # Check bounds
                if np.any(cx_v - r_v < -1e-7) or np.any(cx_v + r_v - w > 1e-7):
                    return -1, None
                if np.any(cy_v - r_v < -1e-7) or np.any(cy_v + r_v - h > 1e-7):
                    return -1, None
                
                # Check overlaps
                # Vectorized
                dx = cx_v[:, None] - cx_v[None, :]
                dy = cy_v[:, None] - cy_v[None, :]
                dist_sq = dx*dx + dy*dy
                r_sum = r_v[:, None] + r_v[None, :]
                
                # Only check upper triangle
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                viol_sq = dist_sq[mask] - r_sum[mask]**2
                if np.any(viol_sq < -1e-7):
                    return -1, None
                
                score = np.sum(r_v)
                return score, np.column_stack((cx_v, cy_v, r_v))
            
            score_curr, res_curr = validate_and_score(x_curr)
            if score_curr > best_score:
                best_score = score_curr
                best_result = res_curr
            
            # Optimization loop
            # We do a few iterations of increasing penalty or just one strong optimization
            # Given time constraints, we'll do a single L-BFGS-B run with a high penalty
            # but careful gradient.
            
            def objective_and_grad(x):
                cx_v = x[:N]
                cy_v = x[N:2*N]
                r_v = x[2*N:]
                
                # Base objective: -sum(r)
                obj = -np.sum(r_v)
                grad = np.zeros(3*N)
                grad[2*N:] = -1.0
                
                # Penalty for boundary violations
                # cx - r < 0, cx + r - w > 0, cy - r < 0, cy + r - h > 0
                # We can rely on bounds for hard constraints, but soft penalty helps 
                # if L-BFGS-B struggles near boundaries? L-BFGS-B handles bounds well.
                # So we focus penalty on overlaps.
                
                # Overlap penalty
                # d = dist(i,j), r_sum = r_i + r_j
                # viol = max(0, r_sum - d)
                # penalty = sum(viol^2)
                
                dx = cx_v[:, None] - cx_v[None, :]
                dy = cy_v[:, None] - cy_v[None, :]
                dist_sq = dx*dx + dy*dy
                # Avoid sqrt(0)
                dist_sq[dist_sq < 1e-12] = 1e-12
                dist = np.sqrt(dist_sq)
                
                r_sum = r_v[:, None] + r_v[None, :]
                
                # Only upper triangle
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                d_pairs = dist[mask]
                rs_pairs = r_sum[mask]
                
                viol = rs_pairs - d_pairs
                # Only positive violations
                viol_pos = np.maximum(0, viol)
                
                obj += penalty * np.sum(viol_pos**2)
                
                # Gradient of penalty
                # d/dx_i (viol^2) = 2 * viol * d/dx_i (rs - d)
                # d/dx_i (rs) = 0 (rs depends on r, not x)
                # d/dx_i (-d) = - (x_i - x_j) / d = (x_j - x_i) / d
                # So d/dx_i (rs - d) = (x_j - x_i) / d
                # Wait, let's be precise.
                # f = (rs - d)^2
                # df/dx_i = 2 (rs - d) * (-1) * d/dx_i (d)
                # d/dx_i (d) = (x_i - x_j) / d
                # So df/dx_i = -2 (rs - d) * (x_i - x_j) / d
                # If viol > 0, rs - d = viol.
                # df/dx_i = -2 * viol * (x_i - x_j) / d
                
                # We need to accumulate gradients for all pairs.
                # This is expensive if done naively in python loops.
                # Vectorized gradient accumulation is tricky.
                
                # Alternative: Use a simpler penalty or just rely on L-BFGS-B bounds 
                # and a different strategy?
                
                # Actually, the "Inspiration 1" mentioned SCR. 
                # Maybe a simpler approach: 
                # Just use L-BFGS-B to maximize sum(r) subject to bounds, 
                # and use a penalty that is differentiable.
                
                # Let's compute the gradient of the penalty term vectorized.
                # grad_pen = 0
                # For each pair (i,j) with viol > 0:
                #   factor = -2 * penalty * viol / d
                #   grad_x[i] += factor * (cx_i - cx_j)
                #   grad_x[j] -= factor * (cx_i - cx_j)
                #   grad_y[i] += factor * (cy_i - cy_j)
                #   grad_y[j] -= factor * (cy_i - cy_j)
                #   grad_r[i] += 2 * penalty * viol
                #   grad_r[j] += 2 * penalty * viol
                
                # We can use numpy indexing to do this.
                # Get indices
                ii, jj = np.where(mask)
                # Only keep those with viol > 0
                active = viol_pos > 0
                ii_act = ii[active]
                jj_act = jj[active]
                viol_act = viol_pos[active]
                d_act = d_pairs[active]
                
                if len(ii_act) > 0:
                    factor = -2.0 * penalty * viol_act / d_act
                    dx_act = cx_v[ii_act] - cx_v[jj_act]
                    dy_act = cy_v[ii_act] - cy_v[jj_act]
                    
                    # Accumulate using np.add.at
                    np.add.at(grad, ii_act, factor * dx_act)
                    np.add.at(grad, jj_act, -factor * dx_act)
                    np.add.at(grad, N + ii_act, factor * dy_act)
                    np.add.at(grad, N + jj_act, -factor * dy_act)
                    
                    # Radius gradient
                    # d/dr_i (viol^2) = 2 * viol * 1
                    # d/dr_j (viol^2) = 2 * viol * 1
                    np.add.at(grad, 2*N + ii_act, 2.0 * penalty * viol_act)
                    np.add.at(grad, 2*N + jj_act, 2.0 * penalty * viol_act)
                
                return obj, grad
            
            try:
                res = minimize(
                    objective_and_grad,
                    x_curr,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                x_opt = res.x
                score_opt, res_opt = validate_and_score(x_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_result = res_opt
                    
            except Exception as e:
                pass
            
            # If time is running out, break inner loop
            if time.time() > deadline - 2:
                break

    # Fallback: simple grid if optimization failed
    if best_result is None:
        # 5x5 grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
