import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 135.0  # Safety margin from 150s limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time

    def compute_objective_and_grad(params):
        """
        params: [x_0...x_N-1, y_0...y_N-1, r_0...r_N-1]
        Objective: Minimize -sum(r) + penalty_terms
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        
        # Hard constraints via penalty:
        # 1. Boundary: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        # 2. Overlap: dist(i,j) >= r_i + r_j
        
        obj_val = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2*N:] = -1.0 # d(-sum r)/dr = -1
        
        # Boundary penalties (quadratic)
        # x - r >= 0 => x - r + penalty
        # We use a soft penalty that is 0 when feasible and grows quadratically when violated
        # Let d_b = min(x, 1-x, y, 1-y) - r
        # If d_b < 0, penalty = 0.5 * d_b^2 * k
        
        k_boundary = 500.0
        
        # Calculate distances to boundaries
        d_left = x - r
        d_right = 1.0 - x - r
        d_bottom = y - r
        d_top = 1.0 - y - r
        
        # Sum of penalties for all 4 boundaries
        # penalty = 0.5 * k * sum( max(0, -d)^2 )
        
        # Gradient contributions from boundary
        # d/dx (0.5 * k * max(0, x-r-1)^2) ... wait, let's just do it term by term or vectorized.
        
        # Vectorized penalty calculation for boundaries
        # d = [x-r, 1-x-r, y-r, 1-y-r] for each i
        # We want to penalize if any d < 0.
        
        # Let's use a simple loop for clarity and gradient accumulation, N=26 is small.
        # Or vectorize:
        # d_x_neg = np.maximum(0, -(x - r)) # violation of x-r >= 0
        # d_x_pos = np.maximum(0, -(1 - x - r)) # violation of x+r <= 1
        
        d_viol_left = np.maximum(0, r - x)
        d_viol_right = np.maximum(0, x + r - 1.0)
        d_viol_bottom = np.maximum(0, r - y)
        d_viol_top = np.maximum(0, y + r - 1.0)
        
        # Penalty value
        p_bound = 0.5 * k_boundary * (
            np.sum(d_viol_left**2) + np.sum(d_viol_right**2) + 
            np.sum(d_viol_bottom**2) + np.sum(d_viol_top**2)
        )
        obj_val += p_bound
        
        # Gradients
        # d/dx (0.5 k (r-x)^2) = k (r-x) (-1) = -k (r-x) = k (x-r)
        # Only if r-x > 0
        grad[:N] += k_boundary * np.where(d_viol_left > 0, x - r, 0)
        # d/dx (0.5 k (x+r-1)^2) = k (x+r-1) (1)
        grad[:N] += k_boundary * np.where(d_viol_right > 0, x + r - 1.0, 0)
        # d/dr (0.5 k (r-x)^2) = k (r-x) (1)
        grad[2*N:] += k_boundary * np.where(d_viol_left > 0, r - x, 0)
        # d/dr (0.5 k (x+r-1)^2) = k (x+r-1) (1)
        grad[2*N:] += k_boundary * np.where(d_viol_right > 0, x + r - 1.0, 0)
        
        # y boundaries
        grad[N:2*N] += k_boundary * np.where(d_viol_bottom > 0, y - r, 0)
        grad[N:2*N] += k_boundary * np.where(d_viol_top > 0, y + r - 1.0, 0)
        grad[2*N:] += k_boundary * np.where(d_viol_bottom > 0, r - y, 0)
        grad[2*N:] += k_boundary * np.where(d_viol_top > 0, y + r - 1.0, 0)

        # Overlap penalties
        k_overlap = 500.0
        # dist_ij = ||p_i - p_j||
        # violation = max(0, r_i + r_j - dist_ij)
        
        # Vectorized distance matrix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        np.fill_diagonal(dist, 1e10)
        
        r_sum = r[:, None] + r[None, :]
        
        # Violation matrix
        # v_ij = max(0, r_sum - dist)
        v = np.maximum(0, r_sum - dist)
        
        p_overlap = 0.5 * k_overlap * np.sum(v**2)
        obj_val += p_overlap
        
        # Gradients for overlap
        # d/dx_i (0.5 k v_ij^2) = k v_ij * d v_ij / d x_i
        # v_ij = r_i + r_j - dist_ij
        # d v_ij / d x_i = - (x_i - x_j) / dist_ij
        # d v_ij / d x_j = - (x_j - x_i) / dist_ij = (x_i - x_j) / dist_ij
        # d v_ij / d r_i = 1
        # d v_ij / d r_j = 1
        
        # We only care about v > 0
        mask = v > 0
        if np.any(mask):
            # Compute unit vectors
            # dist_safe to avoid div by 0
            dist_safe = np.where(dist > 1e-9, dist, 1.0)
            
            # Gradient wrt x_i: sum_j k * v_ij * (-(x_i - x_j)/dist_ij)
            # Note: v is symmetric, dist is symmetric, dx is antisymmetric.
            
            # Let's accumulate gradients
            # grad_x_i += sum_{j!=i} k * v_ij * (x_j - x_i) / dist_ij
            # grad_x_j += sum_{i!=j} k * v_ij * (x_i - x_j) / dist_ij
            
            # Using matrix operations:
            # G_x = -k * (V * (Dx / D)) . 1_vector ?
            # Let Ux_ij = (x_i - x_j) / dist_ij
            # grad_x_i = sum_j k * v_ij * (-Ux_ij)
            # grad_x_j = sum_i k * v_ij * Ux_ij  (since Ux_ji = -Ux_ij)
            
            Ux = dx / dist_safe
            Uy = dy / dist_safe
            
            # Zero out diagonal
            np.fill_diagonal(Ux, 0)
            np.fill_diagonal(Uy, 0)
            
            # Weighted sum
            # grad_x[i] = -k * sum_j v[i,j] * Ux[i,j]
            # Note: v is symmetric.
            grad_x_overlap = -k_overlap * (v * Ux).sum(axis=1)
            grad_y_overlap = -k_overlap * (v * Uy).sum(axis=1)
            
            # Gradient wrt r
            # grad_r_i = sum_j k * v_ij * 1
            grad_r_overlap = k_overlap * v.sum(axis=1)
            
            grad[:N] += grad_x_overlap
            grad[N:2*N] += grad_y_overlap
            grad[2*N:] += grad_r_overlap

        return obj_val, grad

    def is_valid(x, y, r, tol=1e-8):
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
            
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, 1e10)
        
        r_sum = r[:, None] + r[None, :]
        if np.any(dist < r_sum - tol):
            return False
        return True

    best_score = -np.inf
    best_params = None

    # Multi-start optimization
    # 1. Grid start
    # 2. Random starts
    
    starts = []
    
    # Grid
    x_grid, y_grid = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    r_grid = np.full(N, 0.1)
    starts.append(np.concatenate([x_grid, y_grid, r_grid]))
    
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        x_rand = rng.uniform(0.1, 0.9, N)
        y_rand = rng.uniform(0.1, 0.9, N)
        r_rand = rng.uniform(0.05, 0.15, N)
        starts.append(np.concatenate([x_rand, y_rand, r_rand]))
    
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N # Radii upper bound 0.5 (max possible in square is 0.5 for 1 circle, but for 26, much smaller. 0.5 is safe)
    
    for i, p0 in enumerate(starts):
        if elapsed() > DEADLINE - 10:
            break
            
        try:
            res = minimize(
                compute_objective_and_grad, 
                p0, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            
            # Check validity
            if is_valid(x_opt, y_opt, r_opt):
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: simple grid with minimal radii
        x = np.linspace(0.1, 0.9, N)
        y = np.linspace(0.1, 0.9, N)
        r = np.full(N, 0.01)
        best_params = np.concatenate([x, y, r])
        best_score = np.sum(r)
    else:
        x_opt = best_params[:N]
        y_opt = best_params[N:2*N]
        r_opt = best_params[2*N:]
        
        # Final polish: try to increase radii slightly if possible
        # This is a local search step: perturb radii up and check validity
        # Or just run a few more iterations of L-BFGS with tighter tolerance
        
        if elapsed() < DEADLINE - 5:
            try:
                res2 = minimize(
                    compute_objective_and_grad, 
                    best_params, 
                    method='L-BFGS-B', 
                    jac=True, 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res2.success:
                    x_p = res2.x[:N]
                    y_p = res2.x[N:2*N]
                    r_p = res2.x[2*N:]
                    if is_valid(x_p, y_p, r_p):
                        score_p = np.sum(r_p)
                        if score_p > best_score:
                            best_score = score_p
                            best_params = res2.x.copy()
            except Exception:
                pass

    x_final = best_params[:N]
    y_final = best_params[N:2*N]
    r_final = best_params[2*N:]
    
    # One last validity check and shrink if necessary
    if not is_valid(x_final, y_final, r_final, tol=1e-9):
        # Shrink radii until valid
        scale = 1.0
        while not is_valid(x_final, y_final, r_final * scale, tol=1e-9) and scale > 0.5:
            scale -= 0.01
        r_final = r_final * scale
        
    return np.column_stack((x_final, y_final, r_final))
