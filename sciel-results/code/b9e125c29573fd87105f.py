"""
Pack 200 equal circles in a unit square using soft-minimum gradient optimization.
Strategy: 
1. Generate diverse initial configurations (hex, grid, random).
2. Use L-BFGS-B to maximize radius r with a soft-minimum penalty for overlaps and boundaries.
   This transforms the constrained problem into an unconstrained (or box-constrained) smooth problem,
   allowing efficient global convergence to a local optimum without managing active sets explicitly.
3. Polish with direct gradient ascent on the exact minimum clearance if time permits.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
HALF = 0.5
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _compute_radius(c):
    """Compute the valid radius for a configuration. Returns 0 if invalid."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(np.abs(c) > HALF + 1e-12):
        return 0.0
    
    # Compute min pairwise distance / 2
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    # Compute min wall margin
    margins = HALF - np.max(np.abs(c), axis=1)
    min_margin = np.min(margins)
    
    return min(min_pair, min_margin)

def _initial_hex_grid(s, ox, oy):
    """Generate hex grid points, select N, return array (N, 2)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    # Generate a large enough grid
    cols = int(np.ceil(1.0 / dx)) + 4
    rows = int(np.ceil(1.0 / dy)) + 4
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x, y + yoff))
    arr = np.array(pts)
    
    # Filter to square
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    arr = arr[keep]
    
    if arr.shape[0] > N:
        selected = []
        min_dists = np.full(arr.shape[0], np.inf)
        rng = np.random.default_rng(42)
        start = rng.choice(arr.shape[0])
        selected.append(start)
        
        for _ in range(N - 1):
            last = arr[selected[-1]]
            d = np.linalg.norm(arr - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            for s_idx in selected:
                min_dists[s_idx] = -1
            next_idx = np.argmax(min_dists)
            if min_dists[next_idx] < 0:
                break
            selected.append(next_idx)
        arr = arr[selected]
        
    return arr

def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 5.0
    
    rng = np.random.default_rng(2024)
    
    best_c = None
    best_r = 0.0
    
    # Initial radius estimate
    r_init = 0.035
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Hex grid centered
    s_hex = 2.0 * r_init * 1.01
    c_hex = _initial_hex_grid(s_hex, 0.0, 0.0)
    if c_hex.shape[0] == N:
        configs.append(c_hex)
        
    # 2. Hex grid shifted
    c_hex2 = _initial_hex_grid(s_hex, 0.01, 0.01)
    if c_hex2.shape[0] == N:
        configs.append(c_hex2)
        
    # 3. Rectangular grid
    nx, ny = 15, 14
    xs = np.linspace(-HALF + r_init, HALF - r_init, nx)
    ys = np.linspace(-HALF + r_init, HALF - r_init, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    if grid_pts.shape[0] >= N:
        selected = []
        min_dists = np.full(grid_pts.shape[0], np.inf)
        start = rng.choice(grid_pts.shape[0])
        selected.append(start)
        for _ in range(N - 1):
            last = grid_pts[selected[-1]]
            d = np.linalg.norm(grid_pts - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            for s_idx in selected:
                min_dists[s_idx] = -1
            next_idx = np.argmax(min_dists)
            if min_dists[next_idx] < 0:
                break
            selected.append(next_idx)
        c_rect = grid_pts[selected]
        if c_rect.shape[0] == N:
            configs.append(c_rect)
            
    # 4. Random perturbation of hex
    if len(configs) > 0:
        c_rand = configs[0] + rng.normal(0, 0.005, (N, 2))
        c_rand = np.clip(c_rand, -HALF + r_init*0.5, HALF - r_init*0.5)
        configs.append(c_rand)
        
    if not configs:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_c = np.clip(best_c, -HALF + 0.03, HALF - 0.03)
        return best_c

    # Optimization parameters
    # Soft-min beta: higher is sharper (closer to true min), lower is smoother (easier to optimize)
    # Start with moderate beta, increase in later stages if needed
    beta = 50.0
    
    # Precompute indices for pairs
    iu = np.triu_indices(N, k=1)
    pair_i, pair_j = iu
    n_pairs = len(pair_i)
    
    # Wall indices
    wall_x_pos = np.arange(N)
    wall_x_neg = np.arange(N)
    wall_y_pos = np.arange(N)
    wall_y_neg = np.arange(N)
    
    def objective_and_grad(x):
        """
        x: [c_1x, c_1y, ..., c_Nx, c_Ny, r]
        Objective: -r + penalty
        Penalty: soft_min of (2r - d_ij) for pairs and (2r - margin_i) for walls
        We want to maximize r such that constraints are satisfied.
        Equivalent to minimizing: -r + sum(soft_plus(2r - d_ij)) ... 
        Actually, standard approach: 
        Minimize F(x) = -r + lambda * sum( max(0, 2r - d_ij)^2 ) ... but that's non-smooth.
        
        Better: 
        Minimize F(x) = -r + sum( log(1 + exp(beta * (2r - d_ij))) ) / beta ...
        This approximates -r + sum( max(0, 2r - d_ij) )
        If we can push r up while keeping sum(max(0, ...)) small, we win.
        
        However, L-BFGS-B works best with a single smooth scalar objective.
        Let's define:
        f(x) = -r + (1/beta) * sum( log(1 + exp(beta * (2r - d_ij))) ) 
               + (1/beta) * sum( log(1 + exp(beta * (2r - margin_i))) )
        
        Gradient:
        d/dr: -1 + (1/beta) * sum( beta * exp(beta*(2r-d)) / (1+exp(beta*(2r-d))) ) * 2
             = -1 + 2 * sum( sigmoid(beta*(2r-d)) )
        
        d/dc_i: (1/beta) * sum_{j!=i} ( -beta * exp(beta*(2r-d_ij)) / (1+exp(beta*(2r-d_ij))) ) * (c_i - c_j)
               = - sum_{j!=i} sigmoid(beta*(2r-d_ij)) * (c_i - c_j)
        
        d/c_i for walls: similar.
        """
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        # Pairwise distances
        # To save memory/time, we can compute only for pairs, but N=200 is small enough for full matrix?
        # 200x200x2 is 80k floats, fine.
        diff = c[:, None, :] - c[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        d = np.sqrt(d2)
        np.fill_diagonal(d, np.inf)
        
        # Margins
        margins = HALF - np.max(np.abs(c), axis=1)
        
        # Soft plus function: log(1+exp(beta*z))/beta approx max(0,z)
        # We use z = 2r - d for pairs, z = 2r - margin for walls
        # Note: if 2r < d, exp is small, log(1+small) ~ small, term ~ 0.
        # If 2r > d, exp is large, log(exp) ~ beta*z, term ~ z.
        
        z_pairs = 2.0 * r - d
        # Mask diagonal to avoid issues, though d=inf makes z=-inf, exp=0, log(1)=0
        # We only care about upper triangle for symmetry in gradient? 
        # The objective sums over all pairs. Since d is symmetric, sum over all i,j is 2 * sum over i<j.
        # Let's just sum over the full matrix but divide by 2 to match the "each pair once" intuition, 
        # or just sum everything. The gradient will handle it.
        
        # For numerical stability:
        # log(1+exp(z)) is stable if we clip z.
        z_pairs_clipped = np.clip(z_pairs, -100, 100)
        soft_sum_pairs = np.sum(np.log1p(np.exp(beta * z_pairs_clipped))) / beta
        
        # Gradient for pairs
        # d/dz log(1+exp(beta*z)) = beta * exp(beta*z)/(1+exp(beta*z)) = beta * sigmoid(beta*z)
        # d/dx soft_sum = sum( sigmoid(beta*z) * d(z)/dx )
        # z = 2r - d
        # d(z)/dr = 2
        # d(z)/dc_i = - (c_i - c_j) / d  (for each j)
        
        sig_pairs = np.zeros_like(z_pairs)
        # sigmoid(beta*z)
        # Stable sigmoid: 1/(1+exp(-x))
        x_sig = beta * z_pairs_clipped
        sig_pairs = 1.0 / (1.0 + np.exp(-x_sig))
        # Where z is -inf (diagonal), sigmoid is 0. Correct.
        
        # Gradient w.r.t r from pairs
        grad_r_pairs = 2.0 * np.sum(sig_pairs)
        
        # Gradient w.r.t centers from pairs
        # G_i = - sum_{j!=i} sig_pairs[i,j] * (c_i - c_j) / d_ij
        # Note: sig_pairs is 0 on diagonal.
        # (c_i - c_j) / d_ij is the unit vector from j to i.
        
        # We can compute this as:
        # For each i, sum over j of sig[i,j] * (c_j - c_i) / d_ij
        # Let's use matrix operations.
        # sig_pairs is N x N.
        # diff is N x N x 2.
        # We need to weight diff by sig_pairs / d.
        
        # d has inf on diagonal, so 1/d is 0 on diagonal.
        inv_d = np.zeros_like(d)
        mask = d > 1e-10
        inv_d[mask] = 1.0 / d[mask]
        
        # Weighted diff: sig_pairs[:, :, None] * diff * inv_d[:, :, None]
        # Sum over j (axis 1)
        # Result shape (N, 2)
        # This gives sum_j sig[i,j] * (c_i - c_j) / d_ij
        # We want negative of this for gradient?
        # d/d(c_i) sum_j log(1+exp(beta(2r-d_ij)))
        # = sum_j sigmoid(beta(2r-d_ij)) * d(2r-d_ij)/d(c_i)
        # = sum_j sig[i,j] * ( -(c_i - c_j) / d_ij )
        # = - sum_j sig[i,j] * (c_i - c_j) / d_ij
        
        weighted_diff = sig_pairs[:, :, None] * diff * inv_d[:, :, None]
        grad_c_pairs = -np.sum(weighted_diff, axis=1)
        
        # Walls
        # z_wall_x_pos = 2r - (HALF - x_i) = 2r - HALF + x_i
        # z_wall_x_neg = 2r - (HALF + x_i) = 2r - HALF - x_i
        # z_wall_y_pos = 2r - (HALF - y_i) = 2r - HALF + y_i
        # z_wall_y_neg = 2r - (HALF + y_i) = 2r - HALF - y_i
        
        # Wait, margin is HALF - |x|.
        # If x > 0, margin = HALF - x. Constraint: 2r <= HALF - x => x + 2r <= HALF.
        # Violation: 2r - (HALF - x) = 2r - HALF + x.
        # If x < 0, margin = HALF + x. Constraint: 2r <= HALF + x => -x + 2r <= HALF.
        # Violation: 2r - (HALF + x) = 2r - HALF - x.
        
        # Let's just use the margin values directly.
        # margin_i = HALF - max(|x_i|, |y_i|)
        # This is non-differentiable at axes. 
        # To keep it smooth, we can use soft-max for the max? Or just handle the 4 walls separately with smooth approximations.
        # Simpler: Use the 4 linear constraints directly in the soft-sum.
        
        z_w1 = 2.0 * r - (HALF - c[:, 0])  # Right wall
        z_w2 = 2.0 * r - (HALF + c[:, 0])  # Left wall
        z_w3 = 2.0 * r - (HALF - c[:, 1])  # Top wall
        z_w4 = 2.0 * r - (HALF + c[:, 1])  # Bottom wall
        
        z_w1 = np.clip(z_w1, -100, 100)
        z_w2 = np.clip(z_w2, -100, 100)
        z_w3 = np.clip(z_w3, -100, 100)
        z_w4 = np.clip(z_w4, -100, 100)
        
        soft_sum_walls = (np.sum(np.log1p(np.exp(beta * z_w1))) + 
                          np.sum(np.log1p(np.exp(beta * z_w2))) + 
                          np.sum(np.log1p(np.exp(beta * z_w3))) + 
                          np.sum(np.log1p(np.exp(beta * z_w4)))) / beta
        
        sig_w1 = 1.0 / (1.0 + np.exp(-beta * z_w1))
        sig_w2 = 1.0 / (1.0 + np.exp(-beta * z_w2))
        sig_w3 = 1.0 / (1.0 + np.exp(-beta * z_w3))
        sig_w4 = 1.0 / (1.0 + np.exp(-beta * z_w4))
        
        grad_r_walls = 2.0 * (np.sum(sig_w1) + np.sum(sig_w2) + np.sum(sig_w3) + np.sum(sig_w4))
        
        # Gradient w.r.t centers for walls
        # d/dx_i z_w1 = 1
        # d/dx_i z_w2 = -1
        # d/dy_i z_w3 = 1
        # d/dy_i z_w4 = -1
        
        grad_c_walls = np.zeros((N, 2))
        grad_c_walls[:, 0] += sig_w1  # d/dx of z_w1 is 1, grad is -sig? No.
        # Objective term: log(1+exp(beta*z))
        # d/dx = sigmoid(beta*z) * d(z)/dx
        # So grad_c = sum( sig * d(z)/dc )
        
        grad_c_walls[:, 0] = sig_w1 * 1.0 + sig_w2 * (-1.0)
        grad_c_walls[:, 1] = sig_w3 * 1.0 + sig_w4 * (-1.0)
        
        # Total Gradient
        grad_r = -1.0 + grad_r_pairs + grad_r_walls
        grad_c = grad_c_pairs + grad_c_walls
        
        # Objective value
        val = -r + soft_sum_pairs + soft_sum_walls
        
        # Flatten gradient
        grad_flat = np.concatenate([grad_c.ravel(), [grad_r]])
        
        return val, grad_flat

    # Run optimization for each config
    for c_init in configs:
        if time.time() > DEADLINE:
            break
            
        r_current = _compute_radius(c_init)
        if r_current <= 0:
            c_init = np.clip(c_init, -HALF + 0.03, HALF - 0.03)
            r_current = _compute_radius(c_init)
            if r_current <= 0:
                continue
                
        # Start r slightly below current to be safe
        r_start = r_current * 0.99
        
        x0 = np.concatenate([c_init.ravel(), [r_start]])
        
        # Bounds
        # Centers: [-HALF, HALF]
        # r: [0, 0.5]
        bounds = [(-HALF, HALF)] * (2*N) + [(0, 0.5)]
        
        # L-BFGS-B is efficient for large N
        # We use a moderate maxiter and increase beta in a second pass if needed
        try:
            # Phase 1: Moderate beta
            res = minimize(
                objective_and_grad,
                x0,
                jac=True,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False, 'maxls': 50}
            )
            
            if res.fun < 0 or res.success:
                c_opt = res.x[:2*N].reshape(N, 2)
                r_opt = res.x[-1]
                
                # Verify
                r_valid = _compute_radius(c_opt)
                if r_valid > best_r:
                    best_r = r_valid
                    best_c = c_opt.copy()
                    
                # Phase 2: Sharper beta (if time permits)
                if time.time() < DEADLINE - 10:
                    # Increase beta to sharpen the minimum
                    global beta
                    beta_old = beta
                    beta = 200.0
                    
                    # Restart from current best
                    x0_phase2 = res.x
                    res2 = minimize(
                        objective_and_grad,
                        x0_phase2,
                        jac=True,
                        bounds=bounds,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False, 'maxls': 50}
                    )
                    
                    beta = beta_old # Restore
                    
                    if res2.fun < res.fun or res2.success:
                        c_opt2 = res2.x[:2*N].reshape(N, 2)
                        r_opt2 = res2.x[-1]
                        
                        r_valid2 = _compute_radius(c_opt2)
                        if r_valid2 > best_r:
                            best_r = r_valid2
                            best_c = c_opt2.copy()
                            
        except Exception as e:
            # If optimization fails, skip
            pass

    # If no valid configuration found, fallback
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_c = np.clip(best_c, -HALF + 0.03, HALF - 0.03)
    else:
        # Final clamp to ensure validity
        best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
        
    return best_c

if __name__ == "__main__":
    construct()
