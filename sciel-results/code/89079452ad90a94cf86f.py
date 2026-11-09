"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    t0 = time.time()
    
    # We'll try several aspect ratios W/H where W+H=2
    # For each, we optimize positions and radii
    best_score = 0.0
    best_result = None
    
    # Seed fallback (grid in unit square)
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    seed_arr = np.array(cells[:N])
    best_score = np.sum(seed_arr[:, 2])
    best_result = seed_arr.copy()
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5, 2.0, 0.4, 2.5, 0.33]
    
    np.random.seed(42)
    
    for ar in aspect_ratios:
        if time.time() - t0 > DEADLINE - 10:
            break
        # W + H = 2, W/H = ar => W = 2*ar/(1+ar), H = 2/(1+ar)
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        # Max possible radius if all equal: bounded by W/N_cols and H/N_rows
        # Try multiple random initializations
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() - t0 > DEADLINE - 10:
                break
            
            # Initialize positions on a grid with some jitter, radii equal
            # Try different grid arrangements
            # For N=21, try grids like 7x3, 5x5 (partial), 4x6, 3x7, etc.
            grid_configs = [(7, 3), (5, 5), (4, 6), (6, 4), (3, 7), (5, 4), (4, 5), (9, 3), (3, 9)]
            gc = grid_configs[restart % len(grid_configs)]
            nc, nr = gc
            # Ensure we have at least N cells
            if nc * nr < N:
                nc = N  # single row
                nr = 1
            
            # Create grid positions
            positions = []
            for i in range(nc):
                for j in range(nr):
                    if len(positions) >= N:
                        break
                    x = (i + 0.5) * W / nc
                    y = (j + 0.5) * H / nr
                    # Add small random jitter
                    if restart > 0:
                        x += np.random.uniform(-0.05, 0.05) * W / nc
                        y += np.random.uniform(-0.05, 0.05) * H / nr
                        x = np.clip(x, 0.01, W - 0.01)
                        y = np.clip(y, 0.01, H - 0.01)
                    positions.append([x, y])
            
            positions = np.array(positions[:N])
            # Initial radii: min of half-cell-size
            r_init = min(W / nc, H / nr) * 0.45
            radii = np.full(N, r_init)
            
            # Combine into x vector: [x0, y0, r0, x1, y1, r1, ...]
            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = radii[i]
            
            # Objective: maximize sum of radii => minimize negative sum
            def objective(x):
                return -np.sum(x[2::3])
            
            # Constraints:
            # 1. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2 for all i<j
            # 2. Boundary: ri <= xi <= W-ri, ri <= yi <= H-ri
            # 3. ri > 0
            
            # Inequality constraints (scipy wants g(x) >= 0)
            # Non-overlap: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            def non_overlap_constraints(x):
                cons = []
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                        xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                        cons.append({
                            'type': 'ineq',
                            'fun': lambda x, i=i, j=j: (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2 - (x[3*i+2]+x[3*j+2])**2,
                        })
                return cons
            
            # Boundary constraints
            def boundary_constraints(x):
                cons = []
                for i in range(N):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    # xi >= ri
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
                    # W - xi >= ri => W - xi - ri >= 0
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: W - x[3*i] - x[3*i+2]})
                    # yi >= ri
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
                    # H - yi >= ri
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i: H - x[3*i+1] - x[3*i+2]})
                return cons
            
            # Lower bounds: radii >= 0.001
            bounds = []
            for i in range(N):
                bounds.append((0.01, W))      # x
                bounds.append((0.01, H))      # y
                bounds.append((0.001, min(W, H)/2))  # radius
            
            constraints = non_overlap_constraints(x0) + boundary_constraints(x0)
            
            try:
                res = minimize(
                    objective, x0, method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                
                if res.success or (hasattr(res, 'fun') and res.fun < 0):
                    r_sum = -res.fun
                    if r_sum > best_score:
                        # Verify solution
                        sol = res.x
                        valid = True
                        for i in range(N):
                            xi, yi, ri = sol[3*i], sol[3*i+1], sol[3*i+2]
                            if ri < 0.0005 or xi < ri - 1e-6 or W - xi < ri - 1e-6 or \
                               yi < ri - 1e-6 or H - yi < ri - 1e-6:
                                valid = False
                                break
                        if valid:
                            for i in range(N):
                                for j in range(i+1, N):
                                    xi, yi, ri = sol[3*i], sol[3*i+1], sol[3*i+2]
                                    xj, yj, rj = sol[3*j], sol[3*j+1], sol[3*j+2]
                                    dist_sq = (xi-xj)**2 + (yi-yj)**2
                                    if dist_sq < (ri+rj)**2 - 1e-6:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                best_score = r_sum
                                result = np.zeros((N, 3))
                                for i in range(N):
                                    result[i] = [sol[3*i], sol[3*i+1], sol[3*i+2]]
                                best_result = result
            except Exception:
                pass
        
        # After optimizing for this aspect ratio, also try a "stretch" refinement
        # where we fix positions and optimize radii
        if time.time() - t0 > DEADLINE - 10:
            break
    
    # Final: also try optimizing the rectangle dimensions themselves
    # For the best solution found, try adjusting W,H (keeping W+H=2)
    if best_result is not None and time.time() - t0 < DEADLINE - 15:
        # Try a few W values around the best
        best_W = 2.0 * aspect_ratios[0] / (1.0 + aspect_ratios[0])  # default
        # Actually, let's just use the seed as fallback if optimization failed
        pass
    
    return best_result

# Quick sanity check at module level (won't run during evaluation unless imported)
if __name__ == "__main__":
    result = construct()
    print(f"Number of circles: {len(result)}")
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = np.linalg.norm(result[i,:2] - result[j,:2])
            if d < result[i,2] + result[j,2] - 1e-6:
                print(f"OVERLAP: circles {i} and {j}, dist={d:.6f}, sum_r={result[i,2]+result[j,2]:.6f}")
    print("Done")
