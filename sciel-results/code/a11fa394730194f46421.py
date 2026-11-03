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

def _lj_energy_and_grad(x, sigma, epsilon, k_wall):
    """
    Compute Lennard-Jones potential energy and gradient.
    x: flattened array of shape (2N).
    
    E = sum_{i<j} epsilon * (2*(sigma/d)^12 - (sigma/d)^6) + sum_i k_wall * (dist_to_wall)^2
    where dist_to_wall is the distance to the nearest boundary (penalized if < 0, but we use a soft repulsion)
    
    Actually, a better wall potential is:
    For each point, let d_x = HALF - |x_i|, d_y = HALF - |y_i|.
    min_d = min(d_x, d_y).
    If min_d < 0, penalty is large.
    Let's use a smooth repulsive potential for walls:
    V_wall = sum_i k_wall * exp(-min_d / sigma_w)  ? No, that's not standard.
    
    Let's stick to a simple harmonic potential that pushes points away from boundaries if they cross,
    but since we want them INSIDE, we can use a potential that is 0 inside and rises sharply outside?
    Or simply: V_wall = k_wall * sum_i (max(0, |x_i| - HALF + r_margin)^2 + max(0, |y_i| - HALF + r_margin)^2)
    
    However, for BFGS to work well, we want a smooth function everywhere.
    Let's use:
    V_pair = sum_{i<j} epsilon * [ (sigma/d_ij)^12 - 2*(sigma/d_ij)^6 ]  (standard LJ, min at r=sigma*2^(1/6))
    V_wall = k_wall * sum_i ( (HALF - |x_i|)^2 + (HALF - |y_i|)^2 )  ? No, this pushes to center.
    
    We want to maximize the minimum distance.
    Minimizing a repulsive potential generally pushes points apart until constrained by the box.
    The equilibrium of a system of repelling particles in a box will naturally maximize the minimum distance to some extent.
    
    Let's use:
    V = sum_{i<j} epsilon * (sigma/d_ij)^12 - epsilon * (sigma/d_ij)^6
    V_wall = k_wall * sum_i ( exp(-d_i / l) ) where d_i is distance to nearest wall?
    
    Let's try a simpler approach often used in these tasks:
    Minimize sum of inverse powers of distances?
    Or just use the LJ potential.
    
    The "optimal" packing for LJ particles in a box is known to be quite close to the densest packing.
    
    Gradient of V_pair:
    d/dx_i [ (sigma/d)^12 - 2(sigma/d)^6 ]
    Let u = sigma^12/d^12 - 2 sigma^6/d^6
    du/dd = -12 sigma^12/d^13 + 12 sigma^6/d^7 = 12 sigma^6/d^7 (1 - 2(sigma/d)^6)
    dd/dx_i = (x_i - x_j)/d
    So dV/dx_i = 12 epsilon sigma^6/d^7 (1 - 2(sigma/d)^6) * (x_i - x_j)/d
    = 12 epsilon sigma^6/d^8 (1 - 2(sigma/d)^6) (x_i - x_j)
    
    Let's define r0 = sigma * 2^(1/6). At r0, force is 0.
    
    For the wall, to keep particles inside, we can add a potential that increases as they approach the boundary.
    V_wall = k_wall * sum_i ( (HALF - |x_i|)^-2 + (HALF - |y_i|)^-2 ) ? Singular at boundary.
    Better: V_wall = k_wall * sum_i exp(-d_i / l) where d_i is distance to boundary.
    Or simply, since we initialize inside and LJ repels, they might stay inside if k_wall is strong enough to prevent crossing.
    Actually, if we just minimize LJ, they will try to go to infinity or cluster? No, LJ has a minimum.
    But they need to be confined.
    
    Let's use a quadratic wall potential that is active only if they cross? No, smooth.
    V_wall = k_wall * sum_i ( (HALF - x_i)^2 if x_i > 0 else (HALF + x_i)^2 ) ... this is just (HALF - |x_i|)^2.
    This pushes points towards the center? No, it pushes them away from the boundary.
    Wait, (HALF - |x|)^2 is maximized at x=0. We want to MINIMIZE energy.
    So minimizing (HALF - |x|)^2 pushes x towards +/- HALF. That's bad.
    
    We want to penalize being close to the boundary? No, we want to ALLOW being close, but not outside.
    Actually, the packing is constrained by the box. The "pressure" from the repulsive forces pushes them to the walls.
    So we need a wall potential that PREVENTS them from going outside.
    V_wall = k_wall * sum_i [ max(0, |x_i| - (HALF - r_min))^2 + max(0, |y_i| - (HALF - r_min))^2 ]
    This is non-smooth at the boundary.
    
    Let's use a smooth exponential repulsion from the walls:
    V_wall = k_wall * sum_i ( exp(-(HALF - |x_i|)/l) + exp(-(HALF - |y_i|)/l) )
    This is high when close to the wall. Minimizing this pushes points away from the wall?
    No, if we minimize the sum of LJ + V_wall, and V_wall is high at the wall, the points will be pushed INWARDS by the wall potential?
    Wait. If V_wall is high at the wall, the gradient points inwards (towards lower V_wall).
    But the LJ repulsion pushes points OUTWARDS (away from each other).
    So the balance will be somewhere in between.
    This effectively creates a "soft box".
    
    Let's try:
    V = sum_{i<j} LJ(d_ij) + k_wall * sum_i ( exp(-d_xi/l) + exp(-d_yi/l) )
    where d_xi = HALF - |x_i|.
    
    Gradient of V_wall:
    d/dx_i [ exp(-(HALF - |x_i|)/l) ]
    If x_i > 0: d/dx_i [ exp(-(HALF - x_i)/l) ] = exp(-(HALF - x_i)/l) * (1/l)
    If x_i < 0: d/dx_i [ exp(-(HALF + x_i)/l) ] = exp(-(HALF + x_i)/l) * (-1/l)
    So dV_wall/dx_i = sign(x_i) * exp(-d_xi/l) / l
    
    This seems correct.
    """
    c = x.reshape(N, 2)
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # LJ Potential
    # Avoid division by zero
    d_safe = np.maximum(d, 1e-6)
    
    inv_d6 = (sigma / d_safe) ** 6
    lj_val = epsilon * (2 * inv_d6**2 - inv_d6)
    # Sum over upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    E_lj = np.sum(lj_val[iu])
    
    # Gradient of LJ
    # dV/dx_i = 12 * epsilon * sigma^6 / d^8 * (1 - 2*(sigma/d)^6) * (x_i - x_j)
    # Let's compute the factor for each pair
    factor = 12 * epsilon * (sigma**6) / (d_safe**8) * (1 - 2 * inv_d6)
    # factor has shape (N, N)
    # grad[i] += sum_j factor[i,j] * (c[i] - c[j])
    # grad[j] -= sum_i factor[i,j] * (c[i] - c[j])
    
    # We only need upper triangle for the sum, but the matrix is symmetric.
    # Let's use the full matrix for simplicity, then sum.
    # Actually, dV/dx_i = sum_{j != i} factor[i,j] * (c[i] - c[j])
    # This can be computed as:
    # grad_lj = np.sum(factor[:, :, None] * diff, axis=1)
    # But diff[i,j] = c[i] - c[j].
    # factor[i,j] is symmetric?
    # d/dx_i (sigma/d)^12 ... depends on d.
    # The force on i due to j is F_ij = factor[i,j] * (c[i] - c[j]).
    # The force on j due to i is F_ji = factor[j,i] * (c[j] - c[i]) = - factor[i,j] * (c[i] - c[j]) (since factor is symmetric and diff is anti-symmetric).
    # So yes, Newton's 3rd law holds.
    
    grad_lj = np.sum(factor[:, :, None] * diff, axis=1)
    
    # Wall Potential
    abs_c = np.abs(c)
    d_x = HALF - abs_c[:, 0]
    d_y = HALF - abs_c[:, 1]
    
    l = 0.05  # Length scale for wall repulsion
    wall_x = np.exp(-d_x / l)
    wall_y = np.exp(-d_y / l)
    
    E_wall = k_wall * np.sum(wall_x + wall_y)
    
    # Gradient of Wall
    # dV/dx_i = k_wall * sign(x_i) * wall_x / l
    # dV/dy_i = k_wall * sign(y_i) * wall_y / l
    
    grad_wall = np.zeros_like(c)
    grad_wall[:, 0] = k_wall * np.sign(c[:, 0]) * wall_x / l
    grad_wall[:, 1] = k_wall * np.sign(c[:, 1]) * wall_y / l
    
    E_total = E_lj + E_wall
    grad_total = grad_lj + grad_wall
    
    return E_total, grad_total.ravel()

def _initial_hex_grid(s, ox, oy):
    """Generate hex grid points, select N, return array (N, 2)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
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
    
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    arr = arr[keep]
    
    if arr.shape[0] > N:
        # Select N points furthest from each other (greedy farthest point sampling)
        selected = []
        min_dists = np.full(arr.shape[0], np.inf)
        idxs = np.arange(arr.shape[0])
        rng = np.random.default_rng(42)
        start = rng.choice(idxs)
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
    
    # Parameters for LJ potential
    # We want the equilibrium distance to be around the expected packing radius * 2
    # Expected r ~ 0.0366, so d ~ 0.073
    # sigma * 2^(1/6) = 0.073 => sigma = 0.073 / 2^(1/6) ~ 0.062
    sigma = 0.06
    epsilon = 1.0
    k_wall = 100.0  # Strong wall repulsion
    
    # Initial configurations
    configs = []
    
    # 1. Hex grid
    s_hex = 2.0 * 0.0366 * 1.05
    c_hex = _initial_hex_grid(s_hex, 0.0, 0.0)
    if c_hex.shape[0] == N:
        configs.append(c_hex)
        
    # 2. Hex grid shifted
    c_hex2 = _initial_hex_grid(s_hex, 0.01, 0.01)
    if c_hex2.shape[0] == N:
        configs.append(c_hex2)
        
    # 3. Random
    c_rand = rng.uniform(-HALF + 0.04, HALF - 0.04, (N, 2))
    configs.append(c_rand)
    
    # 4. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if grid.shape[0] >= N:
        c_grid = grid[:N]
        configs.append(c_grid)

    for c_init in configs:
        if time.time() > DEADLINE:
            break
            
        # Verify initial config
        r_current = _compute_radius(c_init)
        if r_current <= 0:
            c_init = np.clip(c_init, -HALF + 0.03, HALF - 0.03)
            r_current = _compute_radius(c_init)
            if r_current <= 0:
                continue
                
        x0 = c_init.ravel()
        
        try:
            res = minimize(
                _lj_energy_and_grad,
                x0,
                args=(sigma, epsilon, k_wall),
                jac=True,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-8, 'disp': False}
            )
            
            c_opt = res.x.reshape(N, 2)
            
            # Clip to ensure validity (BFGS doesn't have bounds)
            c_opt = np.clip(c_opt, -HALF + 1e-9, HALF - 1e-9)
            
            r_opt = _compute_radius(c_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
                
        except Exception as e:
            pass

    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_c = np.clip(best_c, -HALF + 0.03, HALF - 0.03)
    else:
        # Final polish with a tighter wall potential or just return
        # Try a few more iterations with higher k_wall to push against walls
        # But for now, just return the best found
        
        # One more polish step: minimize a soft-min objective using the best_c as start
        # This combines the LJ result with the direct objective
        # But time might be tight. Let's just return best_c.
        
        pass

    return best_c

if __name__ == "__main__":
    construct()
