"""
Constrained NLP circle packing in the unit square.

Strategy:
1. Represent the packing by N centers (x, y) and N radii (r).
2. Formulate as a Nonlinear Programming (NLP) problem:
   - Maximize sum(r)
   - Subject to:
     - x_i, y_i, r_i >= 0 (and implicitly <= 1 via constraints)
     - x_i + r_i <= 1, y_i + r_i <= 1 (Boundary constraints)
     - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 (Pairwise non-overlap)
3. Use scipy.optimize.minimize with 'trust-constr' method.
   - This solver handles nonlinear inequality constraints directly.
   - It is robust for small-to-medium scale problems like N=26.
4. Use multi-start with diverse initializations (grid, random, hexagonal).
5. Deadline guard to return best feasible solution.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize, NonlinearConstraint

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute constraint indices for efficiency
    # Pairwise constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # Variables: x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}
    # Total variables: 3N = 78
    
    # Constraint functions for trust-constr
    # g(x) >= 0
    
    def objective(v):
        """Minimize negative sum of radii."""
        # v: (3N,) -> x[0:N], y[N:2N], r[2N:3N]
        r = v[2*N:]
        return -np.sum(r)

    def objective_grad(v):
        """Gradient of objective."""
        g = np.zeros(3*N)
        g[2*N:] = -1.0
        return g

    # Boundary constraints:
    # x_i >= r_i  => x_i - r_i >= 0
    # 1 - x_i >= r_i => 1 - x_i - r_i >= 0
    # y_i >= r_i  => y_i - r_i >= 0
    # 1 - y_i >= r_i => 1 - y_i - r_i >= 0
    # r_i >= 0 => r_i >= 0
    
    # We can combine these into a single vector constraint function.
    # Number of boundary constraints: 5 * N
    
    def boundary_constraints(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        c1 = x - r
        c2 = 1.0 - x - r
        c3 = y - r
        c4 = 1.0 - y - r
        c5 = r
        
        return np.concatenate([c1, c2, c3, c4, c5])

    def boundary_jacobian(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        # Jacobian is sparse, but trust-constr accepts dense.
        # Shape: (5N, 3N)
        J = np.zeros((5*N, 3*N))
        
        # c1 = x - r
        # d c1 / dx = I, d c1 / dr = -I
        J[0:N, 0:N] = np.eye(N)
        J[0:N, 2*N:3*N] = -np.eye(N)
        
        # c2 = 1 - x - r
        # d c2 / dx = -I, d c2 / dr = -I
        J[N:2*N, 0:N] = -np.eye(N)
        J[N:2*N, 2*N:3*N] = -np.eye(N)
        
        # c3 = y - r
        # d c3 / dy = I, d c3 / dr = -I
        J[2*N:3*N, N:2*N] = np.eye(N)
        J[2*N:3*N, 2*N:3*N] = -np.eye(N)
        
        # c4 = 1 - y - r
        # d c4 / dy = -I, d c4 / dr = -I
        J[3*N:4*N, N:2*N] = -np.eye(N)
        J[3*N:4*N, 2*N:3*N] = -np.eye(N)
        
        # c5 = r
        # d c5 / dr = I
        J[4*N:5*N, 2*N:3*N] = np.eye(N)
        
        return J

    # Pairwise constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # Let f_ij = (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2
    
    def pairwise_constraints(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        vals = np.zeros(N*(N-1)//2)
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                vals[idx] = dx*dx + dy*dy - dr*dr
                idx += 1
        return vals

    def pairwise_jacobian(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        m = N*(N-1)//2
        J = np.zeros((m, 3*N))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                
                # f = dx^2 + dy^2 - dr^2
                # d f / dx_i = 2 dx
                # d f / dx_j = -2 dx
                # d f / dy_i = 2 dy
                # d f / dy_j = -2 dy
                # d f / dr_i = -2 dr
                # d f / dr_j = -2 dr
                
                J[idx, i] = 2*dx
                J[idx, N+j] = -2*dx
                J[idx, N+i] = 2*dy
                J[idx, 2*N+j] = -2*dy
                J[idx, 2*N+i] = -2*dr
                J[idx, 2*N+j] = -2*dr
                
                idx += 1
        return J

    # Define constraints for scipy
    # Note: trust-constr requires bounds on variables.
    # x, y in [0, 1], r in [0, 0.5] (max radius in unit square is 0.5)
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    # Nonlinear constraints
    # g(x) >= 0
    cons = [
        NonlinearConstraint(boundary_constraints, 0, np.inf, jac=boundary_jacobian),
        NonlinearConstraint(pairwise_constraints, 0, np.inf, jac=pairwise_jacobian)
    ]

    def run_nlp(v0, time_limit):
        """Run NLP solver from initial guess v0."""
        # Options for trust-constr
        options = {
            'maxiter': 200,
            'gtol': 1e-6,
            'xtol': 1e-8,
            'verbose': 0
        }
        
        try:
            res = minimize(
                objective,
                v0,
                method='trust-constr',
                jac=objective_grad,
                bounds=bounds,
                constraints=cons,
                options=options
            )
            return res.x, -res.fun  # Return best x and sum of radii
        except Exception:
            return None, -np.inf

    def validate_and_fix(v):
        """
        Check if solution is valid. If slightly invalid due to numerical errors,
        shrink radii to make it valid.
        Returns (x, y, r) arrays.
        """
        x = v[0:N].copy()
        y = v[N:2*N].copy()
        r = v[2*N:3*N].copy()
        
        # Clip to bounds
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.clip(r, 0, 0.5)
        
        # Enforce boundary constraints strictly
        # r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        r_bound = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        r = np.minimum(r, r_bound)
        
        # Enforce pairwise constraints strictly
        # If overlap, shrink radii.
        # Simple iterative shrink
        for _ in range(10):
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j]:
                        valid = False
                        # Shrink both radii proportionally to fit distance
                        # Let new_r_i + new_r_j = d
                        # Scale factor
                        overlap = r[i] + r[j] - d
                        if overlap > 0:
                            # Reduce r_i and r_j
                            # Simple fix: reduce each by overlap/2? 
                            # Or scale to fit.
                            # Scaling to fit might be better.
                            scale = d / (r[i] + r[j]) if (r[i]+r[j]) > 0 else 0
                            r[i] *= scale
                            r[j] *= scale
                            # Also re-check boundary
                            r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
                            r[j] = min(r[j], x[j], 1-x[j], y[j], 1-y[j])
            if valid:
                break
                
        return x, y, r

    # Generate initial candidates
    candidates = []
    
    # 1. Grid-like starts (good for N=26, approx 5x5 or 6x5)
    # 5x5 = 25, add 1
    xg, yg = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    base_grid = np.column_stack((xg.flatten(), yg.flatten()))
    # Add one random point
    extra = np.array([[0.5, 0.5]]) # Center
    full_grid = np.vstack([base_grid, extra])
    # Shuffle
    np.random.shuffle(full_grid)
    candidates.append(full_grid)
    
    # 6x5 grid = 30, take 26
    xg, yg = np.meshgrid(np.linspace(0.08, 0.92, 6), np.linspace(0.1, 0.9, 5))
    full_grid_6x5 = np.column_stack((xg.flatten(), yg.flatten()))
    np.random.shuffle(full_grid_6x5)
    candidates.append(full_grid_6x5[:26])
    
    # 2. Random starts
    for _ in range(10):
        pts = np.random.rand(N, 2)
        candidates.append(pts)
        
    # 3. Hexagonal-ish
    s = 0.18
    hex_pts = []
    for i in range(6):
        for j in range(5):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * 0.866
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.min(axis=0)) / (hex_pts.max(axis=0) - hex_pts.min(axis=0))
    noisy_hex = hex_pts[:26] + np.random.normal(0, 0.02, (26, 2))
    noisy_hex = np.clip(noisy_hex, 0.0, 1.0)
    candidates.append(noisy_hex)

    best_val = -np.inf
    best_v = None

    # Time budget per candidate
    # We have ~13 candidates. 145s / 13 ~ 11s each.
    # trust-constr can be slow. Let's limit iterations or time per run.
    # Since we can't easily interrupt minimize, we rely on maxiter and fast fails.
    # We will run each candidate with a strict maxiter.
    
    for i, cand_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Construct initial v0
        # x, y from cand_pts
        # r: estimate max possible radius. 
        # Use Voronoi radius estimate for a good start.
        x = cand_pts[:, 0]
        y = cand_pts[:, 1]
        r_bound = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        
        # Estimate neighbor distance
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists, axis=1) / 2.0
        
        r_init = np.minimum(r_bound, min_dist) * 0.9 # Slightly smaller to be feasible
        
        v0 = np.concatenate([x, y, r_init])
        
        # Run NLP
        # To avoid hanging, we don't have a time limit inside minimize.
        # We must trust maxiter to finish quickly.
        # For N=26, 78 vars, ~300 constraints.
        # trust-constr might take a few seconds per iteration.
        # Let's try.
        
        v_opt, val_opt = run_nlp(v0, deadline)
        
        if v_opt is not None:
            # Validate
            x_f, y_f, r_f = validate_and_fix(v_opt)
            val_f = np.sum(r_f)
            
            if val_f > best_val:
                best_val = val_f
                best_v = v_opt
                # Store fixed version for output
                best_x, best_y, best_r = x_f, y_f, r_f

    # If we have time left, do a final polish on the best result
    if best_v is not None and time.time() < deadline - 10.0:
        # Re-run with best_v as start, maybe tighter tolerances?
        # Or just run again to see if it converges better.
        v_opt, val_opt = run_nlp(best_v, deadline)
        if v_opt is not None:
            x_f, y_f, r_f = validate_and_fix(v_opt)
            val_f = np.sum(r_f)
            if val_f > best_val:
                best_val = val_f
                best_v = v_opt
                best_x, best_y, best_r = x_f, y_f, r_f

    if best_v is None:
        # Fallback: simple grid
        n_x, n_y = 5, 5
        r_val = 0.09
        x_coords = np.linspace(r_val, 1-r_val, n_x)
        y_coords = np.linspace(r_val, 1-r_val, n_y)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        x_f = x_f.flatten()
        y_f = y_f.flatten()
        # We need 26 points. 5x5=25. Add one more.
        # Just take first 26 from a denser grid or repeat.
        # Let's use 6x5=30 grid, take 26.
        x_coords = np.linspace(0.1, 0.9, 6)
        y_coords = np.linspace(0.1, 0.9, 5)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        x_f = x_f.flatten()[:26]
        y_f = y_f.flatten()[:26]
        
        # Calculate radii for fallback
        r_bound = np.minimum(np.minimum(x_f, 1.0-x_f), np.minimum(y_f, 1.0-y_f))
        dx = x_f[:, None] - x_f[None, :]
        dy = y_f[:, None] - y_f[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists, axis=1) / 2.0
        r_f = np.minimum(r_bound, min_dist)
        
        best_x, best_y, best_r = x_f, y_f, r_f

    # Construct output
    result = np.column_stack((best_x, best_y, best_r))
    
    return result

# EVOLVE-BLOCK-END
