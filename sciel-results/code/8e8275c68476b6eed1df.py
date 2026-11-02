"""Circle packing in perimeter-4 rectangle: maximize sum of 21 circle radii."""
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

N = 21
DEADLINE = 145.0  # safety margin before 150s kill

def _softplus(x, beta=0.01):
    """Numerically stable softplus."""
    return (1.0 / beta) * np.log1p(np.exp(-beta * np.abs(x))) * np.sign(x) + np.maximum(x, 0.0)

def _softplus_grad(x, beta=0.01):
    """Gradient of softplus."""
    return 1.0 / (1.0 + np.exp(beta * x))

def _delaunay_edges(centers, W, H):
    """Get Delaunay edges for contact detection."""
    try:
        # Add boundary points to help with edge detection
        pts = np.vstack([
            centers,
            np.array([[0, 0], [W, 0], [W, H], [0, H]])
        ])
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a < N and b < N:
                        edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _solve_lp_radii(centers, W, H, max_iter=100):
    """Solve LP to maximize sum of radii given fixed centers."""
    # Variables: r_0, ..., r_{N-1}
    # Maximize: sum(r_i)
    # Constraints:
    #   r_i + r_j <= d_ij for all pairs (i,j)
    #   r_i <= min(x_i, W-x_i, y_i, H-y_i) for all i
    #   r_i >= 0
    
    c = -np.ones(N)  # minimize -sum(r)
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= d_ij
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    
    # Boundary constraints: r_i <= bound_i
    bounds_list = []
    for i in range(N):
        bound = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
        bounds_list.append(bound)
    
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, b) for b in bounds_list], method='highs')
    
    if res.success:
        return res.x
    return None

def _max_equal_radius(centers, W, H):
    """Find maximum equal radius for given centers."""
    min_bound = np.min([
        np.min(centers[:, 0]), np.min(W - centers[:, 0]),
        np.min(centers[:, 1]), np.min(H - centers[:, 1])
    ])
    
    min_pair = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            min_pair = min(min_pair, d / 2.0)
    
    return min(min_bound, min_pair)

def _make_hex(n, W, H):
    """Create hexagonal packing layout."""
    positions = []
    # Try various row/column arrangements
    best = None
    best_score = -1
    
    for ncol in range(3, 8):
        nrow = int(np.ceil(n / ncol))
        pos = []
        for i in range(nrow):
            for j in range(ncol):
                if len(pos) >= n:
                    break
                x = (j + 0.5) / ncol
                y = (i + 0.5) / nrow
                if i % 2 == 1:
                    x += 0.5 / ncol
                # Clamp to bounds
                x = np.clip(x, 0.1, W - 0.1)
                y = np.clip(y, 0.1, H - 0.1)
                pos.append([x, y])
        
        if len(pos) == n:
            pos = np.array(pos)
            # Score by min distance
            min_d = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_d = min(min_d, d)
            score = min_d + min(pos.min(), (W - pos).min().min(), (H - pos).min().min())
            if score > best_score:
                best_score = score
                best = pos
    
    if best is None:
        side = int(np.ceil(np.sqrt(n)))
        best = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                         for i in range(side) for j in range(side)][:n])
    return best

def _make_grid(n, W, H):
    """Create grid layout."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                    for i in range(side) for j in range(side)][:n])
    return pos

def _make_random(n, W, H, seed=42):
    """Create random layout with minimum spacing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (n, 2))
    return pos

def _make_ring(n, W, H):
    """Create concentric ring layout."""
    pos = np.zeros((n, 2))
    cx, cy = W/2, H/2
    pos[0] = [cx, cy]
    # Ring 1: 6 circles
    for i in range(6):
        angle = i * 2 * np.pi / 6
        pos[i+1] = [cx + 0.2*W*np.cos(angle), cy + 0.2*H*np.sin(angle)]
    # Ring 2: 14 circles
    for i in range(14):
        angle = i * 2 * np.pi / 14
        pos[i+7] = [cx + 0.4*W*np.cos(angle), cy + 0.4*H*np.sin(angle)]
    return pos

def _objective_and_grad(x, W, H, beta=0.01, penalty_weight=500.0):
    """
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    Minimize: -sum(r_i) + penalty for overlaps and boundary violations
    Using softplus for smooth penalties.
    """
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Boundary violations: r_i > min(x_i, W-x_i, y_i, H-y_i)
    bounds = np.minimum(
        np.minimum(centers[:, 0], W - centers[:, 0]),
        np.minimum(centers[:, 1], H - centers[:, 1])
    )
    boundary_viol = radii - bounds  # positive when violated
    
    # Pairwise overlap violations: r_i + r_j > d_ij
    overlap_viol = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            v = radii[i] + radii[j] - d
            overlap_viol[i, j] = v
            overlap_viol[j, i] = v
    
    # Softplus penalty
    boundary_pen = np.sum(_softplus(boundary_viol, beta))
    overlap_pen = np.sum(_softplus(overlap_viol, beta))
    
    obj = -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)
    
    # Gradient
    grad = np.zeros_like(x)
    grad[2*N:] = -1.0  # d(-sum(r))/dr
    
    # Boundary penalty gradient
    for i in range(N):
        bv = boundary_viol[i]
        sp_grad = _softplus_grad(bv, beta)
        if sp_grad > 1e-10:
            # d(boundary_pen)/dr_i = sp_grad
            grad[2*N + i] += penalty_weight * sp_grad
            # d(boundary_viol)/d(centers[i]) depends on which bound is active
            m_x = min(centers[i, 0], W - centers[i, 0])
            m_y = min(centers[i, 1], H - centers[i, 1])
            if m_x <= m_y:
                if centers[i, 0] <= W - centers[i, 0]:
                    grad[2*i] -= penalty_weight * sp_grad
                else:
                    grad[2*i] += penalty_weight * sp_grad
            else:
                if centers[i, 1] <= H - centers[i, 1]:
                    grad[2*i+1] -= penalty_weight * sp_grad
                else:
                    grad[2*i+1] += penalty_weight * sp_grad
    
    # Overlap penalty gradient
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            v = radii[i] + radii[j] - d
            sp_grad = _softplus_grad(v, beta)
            if sp_grad > 1e-10:
                diff = centers[i] - centers[j]
                if d > 1e-10:
                    # d(v)/d(centers[i]) = (centers[i]-centers[j])/d
                    grad[2*i] += penalty_weight * sp_grad * diff[0] / d
                    grad[2*i+1] += penalty_weight * sp_grad * diff[1] / d
                    # d(v)/d(centers[j]) = -(centers[i]-centers[j])/d
                    grad[2*j] -= penalty_weight * sp_grad * diff[0] / d
                    grad[2*j+1] -= penalty_weight * sp_grad * diff[1] / d
                # d(v)/dr_i = 1, d(v)/dr_j = 1
                grad[2*N + i] += penalty_weight * sp_grad
                grad[2*N + j] += penalty_weight * sp_grad
    
    return obj, grad

def _refine_slsqp(x0, W, H, time_budget=5.0):
    """Refine using SLSQP with explicit constraints."""
    t_start = time.time()
    
    def obj(x):
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        return -np.sum(radii)
    
    def obj_grad(x):
        return np.concatenate([np.zeros(2*N), -np.ones(N)])
    
    # Constraints: r_i + r_j <= d_ij
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            def make_constraint(i, j):
                def constraint(x):
                    centers = x[:2*N].reshape(N, 2)
                    radii = x[2*N:]
                    d = np.linalg.norm(centers[i] - centers[j])
                    return d - radii[i] - radii[j]
                def constraint_grad(x):
                    centers = x[:2*N].reshape(N, 2)
                    grad = np.zeros_like(x)
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d > 1e-10:
                        diff = centers[i] - centers[j]
                        grad[2*i] = diff[0] / d
                        grad[2*i+1] = diff[1] / d
                        grad[2*j] = -diff[0] / d
                        grad[2*j+1] = -diff[1] / d
                    grad[2*N + i] = -1.0
                    grad[2*N + j] = -1.0
                    return grad
                return {'type': 'ineq', 'fun': constraint, 'jac': constraint_grad}
            cons.append(make_constraint(i, j))
    
    # Boundary constraints: r_i <= min(x_i, W-x_i, y_i, H-y_i)
    for i in range(N):
        def make_boundary(i, side):
            def constraint(x):
                centers = x[:2*N].reshape(N, 2)
                radii = x[2*N:]
                if side == 0:
                    return centers[i, 0] - radii[i]
                elif side == 1:
                    return W - centers[i, 0] - radii[i]
                elif side == 2:
                    return centers[i, 1] - radii[i]
                else:
                    return H - centers[i, 1] - radii[i]
            def constraint_grad(x):
                grad = np.zeros_like(x)
                if side == 0:
                    grad[2*i] = 1.0
                elif side == 1:
                    grad[2*i] = -1.0
                elif side == 2:
                    grad[2*i+1] = 1.0
                else:
                    grad[2*i+1] = -1.0
                grad[2*N + i] = -1.0
                return grad
            return {'type': 'ineq', 'fun': constraint, 'jac': constraint_grad}
        for side in range(4):
            cons.append(make_boundary(i, side))
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, W))
        bounds.append((0.0, H))
    for i in range(N):
        bounds.append((0.0, 0.5))
    
    try:
        result = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                         bounds=bounds, constraints=cons,
                         options={'maxiter': 100, 'ftol': 1e-12})
        if result.success:
            return result.x
    except Exception:
        pass
    
    return x0

def _polish(x0, W, H, time_budget=10.0):
    """Polish using L-BFGS-B with softplus objective."""
    t_start = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, W))
        bounds.append((0.0, H))
    for i in range(N):
        bounds.append((0.0, 0.5))
    
    try:
        result = minimize(_objective_and_grad, x0, args=(W, H, 0.005, 500.0),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
        if result.success or result.fun < 1e6:
            return result.x
    except Exception:
        pass
    
    return x0

def _make_result_from_lp(centers, W, H):
    """Create result from LP-solved radii."""
    radii = _solve_lp_radii(centers, W, H)
    if radii is None:
        # Fallback: equal radius
        r = _max_equal_radius(centers, W, H)
        radii = np.full(N, max(0, r - 1e-6))
    return np.column_stack([centers, radii])

def _make_result_equal(centers, W, H):
    """Create result with equal radii."""
    r = _max_equal_radius(centers, W, H)
    radii = np.full(N, max(0, r - 1e-6))
    return np.column_stack([centers, radii])

def _validate_and_fix(result, W, H):
    """Validate result and fix any violations."""
    centers = result[:, :2].copy()
    radii = result[:, 2].copy()
    
    # Clamp radii to bounds
    for i in range(N):
        radii[i] = min(radii[i], centers[i, 0], W - centers[i, 0],
                       centers[i, 1], H - centers[i, 1])
        radii[i] = max(0, radii[i])
    
    # Fix overlaps iteratively
    for _ in range(20):
        fixed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d + 1e-9:
                    excess = radii[i] + radii[j] - d
                    # Reduce proportionally
                    total = radii[i] + radii[j]
                    if total > 1e-10:
                        scale = max(0, (d - 1e-7) / total)
                        radii[i] *= scale
                        radii[j] *= scale
                    fixed = True
        if not fixed:
            break
    
    result[:, :2] = centers
    result[:, 2] = radii
    return result

def _is_valid(result, W, H):
    """Check if result is valid."""
    centers = result[:, :2]
    radii = result[:, 2]
    
    # Check bounds
    for i in range(N):
        if radii[i] < -1e-9:
            return False
        if centers[i, 0] - radii[i] < -1e-7 or W - centers[i, 0] - radii[i] < -1e-7:
            return False
        if centers[i, 1] - radii[i] < -1e-7 or H - centers[i, 1] - radii[i] < -1e-7:
            return False
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d + 1e-7:
                return False
    
    return True

def construct():
    t0 = time.time()
    
    # Rectangle with perimeter 4: w + h = 2. Use square w=h=1 for best packing.
    W, H = 1.0, 1.0
    
    best_result = None
    best_value = -1
    
    # Generate diverse initial configurations
    configs = []
    configs.append(_make_hex(N, W, H))
    configs.append(_make_grid(N, W, H))
    configs.append(_make_ring(N, W, H))
    
    # Multiple random configs with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        configs.append(_make_random(N, W, H, seed=seed))
    
    # Add perturbed hex
    hex_pos = _make_hex(N, W, H).copy()
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = hex_pos + rng.normal(0, 0.05, (N, 2))
        perturbed = np.clip(perturbed, 0.05, 0.95)
        configs.append(perturbed)
    
    for idx, config in enumerate(configs):
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 15:
            break
        
        # Ensure config is within bounds
        config = np.clip(config, 0.01, 0.99)
        
        # Strategy 1: LP-based radius solving with fixed centers
        result_lp = _make_result_from_lp(config, W, H)
        result_lp = _validate_and_fix(result_lp, W, H)
        if _is_valid(result_lp, W, H):
            value = np.sum(result_lp[:, 2])
            if value > best_value:
                best_value = value
                best_result = result_lp.copy()
        
        # Strategy 2: Equal radius
        result_eq = _make_result_equal(config, W, H)
        result_eq = _validate_and_fix(result_eq, W, H)
        if _is_valid(result_eq, W, H):
            value = np.sum(result_eq[:, 2])
            if value > best_value:
                best_value = value
                best_result = result_eq.copy()
        
        # Strategy 3: Full optimization with L-BFGS-B polish
        if elapsed < DEADLINE - 20:
            # Initial radii from LP
            radii_init = _solve_lp_radii(config, W, H)
            if radii_init is None:
                radii_init = np.full(N, 0.05)
            radii_init = np.clip(radii_init, 0.01, 0.4)
            
            x0 = np.concatenate([config.ravel(), radii_init])
            
            # Polish
            x_polished = _polish(x0, W, H, time_budget=10.0)
            
            centers = x_polished[:2*N].reshape(N, 2)
            radii = x_polished[2*N:]
            
            # Final LP solve for optimal radii at these centers
            radii_lp = _solve_lp_radii(centers, W, H)
            if radii_lp is not None:
                radii = radii_lp
            
            result = np.column_stack([centers, radii])
            result = _validate_and_fix(result, W, H)
            if _is_valid(result, W, H):
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result.copy()
        
        # Strategy 4: SLSQP refinement (if time allows)
        if elapsed < DEADLINE - 10 and idx < 5:
            radii_init = _solve_lp_radii(config, W, H)
            if radii_init is None:
                radii_init = np.full(N, 0.05)
            radii_init = np.clip(radii_init, 0.01, 0.4)
            
            x0 = np.concatenate([config.ravel(), radii_init])
            x_slsqp = _refine_slsqp(x0, W, H, time_budget=5.0)
            
            centers = x_slsqp[:2*N].reshape(N, 2)
            radii = x_slsqp[2*N:]
            
            # Final LP solve
            radii_lp = _solve_lp_radii(centers, W, H)
            if radii_lp is not None:
                radii = radii_lp
            
            result = np.column_stack([centers, radii])
            result = _validate_and_fix(result, W, H)
            if _is_valid(result, W, H):
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result.copy()
    
    # If no valid result, fallback to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
