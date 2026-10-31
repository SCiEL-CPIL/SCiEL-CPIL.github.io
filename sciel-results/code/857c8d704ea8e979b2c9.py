"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint, LinearConstraint
import itertools

N = 26

# EVOLVE-BLOCK-START

def _vectorized_penalty(pos, radii, weight=1000.0):
    """Vectorized penalty function for fast L-BFGS-B optimization."""
    # Boundary penalties
    # pos shape (N, 2), radii shape (N,)
    # Left: x < r -> max(0, r - x)^2
    # Right: x > 1-r -> max(0, x - (1-r))^2
    
    r = radii
    x = pos[:, 0]
    y = pos[:, 1]
    
    pen_x_left = np.maximum(0, r - x)**2
    pen_x_right = np.maximum(0, x - (1.0 - r))**2
    pen_y_left = np.maximum(0, r - y)**2
    pen_y_right = np.maximum(0, y - (1.0 - r))**2
    
    boundary_pen = np.sum(pen_x_left + pen_x_right + pen_y_left + pen_y_right)
    
    # Pairwise overlap penalties
    # dist_sq[i,j] = (x_i-x_j)^2 + (y_i-y_j)^2
    # min_dist[i,j] = r_i + r_j
    # violation = max(0, min_dist^2 - dist_sq)^2
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    min_dist = r[:, None] + r[None, :]
    min_dist_sq = min_dist**2
    
    # Only consider i < j to avoid double counting and self-interaction
    # Create a mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Calculate penalty only for pairs where dist_sq < min_dist_sq
    # violation = (min_dist_sq - dist_sq)
    # penalty = violation^2
    # We want sum over i<j
    
    diff = min_dist_sq - dist_sq
    # Set diagonal and lower triangle to 0
    diff = diff * mask
    # Only positive violations contribute
    violation = np.maximum(0, diff)
    # Squared penalty for smoothness
    pair_pen = np.sum(violation**2)
    
    return -np.sum(radii) + weight * (boundary_pen + pair_pen)

def _objective_penalty(x, weight=1000.0):
    """Wrapper for vectorized penalty."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    radii = np.maximum(radii, 1e-6)
    return _vectorized_penalty(pos, radii, weight)

def _get_constraints():
    """
    Returns a list of NonlinearConstraint objects for trust-constr.
    Variables x: [x_0...x_{N-1}, y_0...y_{N-1}, r_0...r_{N-1}]
    Total dim: 3N
    """
    constraints = []
    
    # 1. Boundary constraints:
    # x_i - r_i >= 0
    # x_i - (1 - r_i) <= 0  => -(x_i - (1 - r_i)) >= 0 => r_i - x_i + 1 >= 0
    # y_i - r_i >= 0
    # y_i - (1 - r_i) <= 0  => r_i - y_i + 1 >= 0
    
    # We can formulate these as linear constraints for efficiency if we use LinearConstraint,
    # but NonlinearConstraint is fine too. Let's use LinearConstraint for boundaries as they are linear in x.
    
    # x_i - r_i >= 0
    # x_i + 0*y_i - r_i >= 0
    A_lin = np.zeros((4*N, 3*N))
    b_lin = np.zeros(4*N)
    
    for i in range(N):
        idx_x = i
        idx_y = N + i
        idx_r = 2*N + i
        
        # x_i - r_i >= 0
        A_lin[i, idx_x] = 1.0
        A_lin[i, idx_r] = -1.0
        b_lin[i] = 0.0
        
        # 1 - x_i - r_i >= 0  => -x_i - r_i >= -1
        A_lin[N+i, idx_x] = -1.0
        A_lin[N+i, idx_r] = -1.0
        b_lin[N+i] = -1.0
        
        # y_i - r_i >= 0
        A_lin[2*N+i, idx_y] = 1.0
        A_lin[2*N+i, idx_r] = -1.0
        b_lin[2*N+i] = 0.0
        
        # 1 - y_i - r_i >= 0 => -y_i - r_i >= -1
        A_lin[3*N+i, idx_y] = -1.0
        A_lin[3*N+i, idx_r] = -1.0
        b_lin[3*N+i] = -1.0
        
    # r_i >= 0 is handled by bounds, but let's ensure r_i > 0 via bounds.
    
    constraints.append(LinearConstraint(A_lin, b_lin, np.inf))
    
    # 2. Pairwise distance constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    pairs = list(itertools.combinations(range(N), 2))
    n_pairs = len(pairs)
    
    def fun_pairwise(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        vals = np.zeros(n_pairs)
        for k, (i, j) in enumerate(pairs):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            vals[k] = dist_sq - min_dist*min_dist
        return vals
        
    def jac_pairwise(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        J = np.zeros((n_pairs, 3*N))
        for k, (i, j) in enumerate(pairs):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            min_dist = radii[i] + radii[j]
            
            # d(dist_sq - min_dist^2) / d(x_i) = 2*dx
            # d(dist_sq - min_dist^2) / d(x_j) = -2*dx
            # d(dist_sq - min_dist^2) / d(y_i) = 2*dy
            # d(dist_sq - min_dist^2) / d(y_j) = -2*dy
            # d(dist_sq - min_dist^2) / d(r_i) = -2*min_dist
            # d(dist_sq - min_dist^2) / d(r_j) = -2*min_dist
            
            idx_xi = i
            idx_xj = j
            idx_yi = N + i
            idx_yj = N + j
            idx_ri = 2*N + i
            idx_rj = 2*N + j
            
            J[k, idx_xi] = 2.0 * dx
            J[k, idx_xj] = -2.0 * dx
            J[k, idx_yi] = 2.0 * dy
            J[k, idx_yj] = -2.0 * dy
            J[k, idx_ri] = -2.0 * min_dist
            J[k, idx_rj] = -2.0 * min_dist
            
        return J
        
    constraints.append(NonlinearConstraint(fun_pairwise, 0, np.inf, jac=jac_pairwise))
    
    return constraints

def _objective_tc(x):
    """Objective for trust-constr: minimize negative sum of radii."""
    radii = x[2*N:]
    return -np.sum(radii)

def _jac_objective_tc(x):
    """Jacobian of objective for trust-constr."""
    J = np.zeros(3*N)
    J[2*N:] = -1.0
    return J

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(100):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii

def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Try to fit 5 rows of 5 circles? No, 26 circles.
    # 5x5 grid = 25. 1 extra.
    # Let's use a hex grid with slightly larger spacing to allow growth.
    r_init = 0.09
    y_spacing = r_init * np.sqrt(3)
    
    # Rows: 5, 4, 5, 4, 5, 3? Total 26.
    # Let's just use a 5x5 grid and put the 26th in a gap, then let optimizer fix it.
    # This is safer.
    pos[:25, 0] = np.tile(np.linspace(0.1, 0.9, 5), 5)
    pos[:25, 1] = np.repeat(np.linspace(0.1, 0.9, 5), 5)
    radii[:25] = r_init
    
    # 26th circle in center gap?
    pos[25, 0] = 0.5
    pos[25, 1] = 0.5
    radii[25] = r_init * 0.5
    
    return pos, radii

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.2, 0.8, size=(N, 2))
    radii = rng.uniform(0.05, 0.1, size=N)
    return pos, radii

def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 1. Fast L-BFGS-B with penalty
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective_penalty, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < _objective_penalty(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                        x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass

    # 2. Trust-Constr for high precision (if time permits)
    if time.time() < deadline - 5:
        try:
            constraints = _get_constraints()
            result = minimize(_objective_tc, x0, method='trust-constr',
                              bounds=bounds, constraints=constraints,
                              jac=_jac_objective_tc,
                              options={'maxiter': 100, 'gtol': 1e-8, 'xtol': 1e-10})
            
            if result.success and result.fun < -best_score + 1e-6:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass

    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # 1. Global Search with Differential Evolution
    # We want to use DE to find good basins.
    # DE is slow, so we limit iterations.
    bounds = []
    for i in range(N):
        bounds.append((0.05, 0.95))
        bounds.append((0.05, 0.95))
    for i in range(N):
        bounds.append((0.05, 0.2))
        
    de_deadline = start_time + 30.0 # Give DE 30s
    
    try:
        # Use a smaller popsize to fit in time
        result_de = differential_evolution(
            _objective_penalty, bounds,
            maxiter=50, popsize=8,
            tol=1e-5, mutation=(0.5, 1.5), recombination=0.7,
            seed=42, polish=False,
            updating='deferred',
            callback=lambda xk, convergence: (time.time() > de_deadline)
        )
        
        if result_de.fun < 0:
            x_opt = result_de.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    except Exception:
        pass

    # 2. Local Optimization from Multiple Starts
    # If DE failed or time is left, try local starts.
    starts = []
    
    # Hexagonal
    pos, radii = _initial_hexagonal()
    starts.append((pos, radii))
    
    # Random starts
    for seed in [1, 2, 3]:
        pos, radii = _initial_random(seed)
        starts.append((pos, radii))
        
    # If we have a best from DE, add a perturbed version of it
    if best_pos is not None:
        rng = np.random.RandomState(100)
        pos_pert = best_pos + rng.normal(0, 0.01, size=best_pos.shape)
        pos_pert = np.clip(pos_pert, 0.05, 0.95)
        starts.append((pos_pert, best_radii.copy()))
        
    for pos, radii in starts:
        if time.time() > deadline - 15:
            break
            
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
            
    # 3. Final Polish
    if best_pos is not None and time.time() < deadline - 2:
        # One more pass of Trust-Constr on the absolute best
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            constraints = _get_constraints()
            result = minimize(_objective_tc, x0, method='trust-constr',
                              bounds=[(0,1)]*2*N + [(1e-5, 0.5)]*N,
                              constraints=constraints,
                              jac=_jac_objective_tc,
                              options={'maxiter': 50, 'gtol': 1e-9, 'xtol': 1e-12})
            
            if result.success:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx2 = i * 5 + j
                if idx2 < N:
                    pos[idx2, 0] = (i + 0.5) / 5.0
                    pos[idx2, 1] = (j + 0.5) / 5.0
                    radii[idx2] = 0.09
        pos[25, 0] = 0.5
        pos[25, 1] = 0.5
        radii[25] = 0.05
        best_pos = pos
        best_radii = radii
        
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
