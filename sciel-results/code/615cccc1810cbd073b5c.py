"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


def _compute_radius(pts):
    """Compute the minimum radius for a given set of centers using vectorized ops."""
    n = len(pts)
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary_dist)
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    min_pair = np.min(dist_matrix[iu]) / 2.0
    return min(min_boundary, min_pair)


def _soft_min_radius(pts, beta=100.0):
    """Soft-min approximation of the radius using log-sum-exp."""
    n = len(pts)
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu] / 2.0
    all_dists = np.concatenate([boundary_dist, pair_dists])
    min_val = np.min(all_dists)
    if min_val < -100:
        return min_val
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return min_val
    return min_val - np.log(sum_exp) / beta


def _soft_min_radius_grad(pts, beta=100.0):
    """Gradient of the soft-min radius."""
    n = len(pts)
    grad = np.zeros_like(pts)
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu] / 2.0
    all_dists = np.concatenate([boundary_dist, pair_dists])
    min_val = np.min(all_dists)
    if min_val < -100:
        return grad
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return grad
    weights = exp_vals / sum_exp
    for i in range(n):
        w = weights[i]
        if norms[i] > 1e-10:
            grad[i] += w * (-pts[i] / norms[i])
    for k in range(len(iu[0])):
        w = weights[n + k]
        i, j = iu[0][k], iu[1][k]
        diff = pts[i] - pts[j]
        norm_diff = np.linalg.norm(diff)
        if norm_diff > 1e-10:
            grad[i] += w * (diff / norm_diff)
            grad[j] += w * (-diff / norm_diff)
    return grad


def _simulated_annealing(pts, deadline, seed=0):
    """
    Simulated annealing on the packing graph.
    Moves: push circles apart along their contact directions, or nudge toward boundary.
    Energy = -radius (we minimize energy = maximize radius).
    Accept non-improving moves with probability exp(-delta_E / T).
    """
    rng = np.random.RandomState(seed)
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.0005
    n_steps = 20000
    cooling = (T_end / T_start) ** (1.0 / n_steps)
    
    current = pts.copy()
    current_r = best_r
    T = T_start
    
    # Precompute indices
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    
    for step in range(n_steps):
        if time.time() > deadline - 3:
            break
        
        # Pick a random circle to move
        i = rng.randint(n)
        
        # Compute forces on circle i from its neighbors
        force = np.zeros(2)
        # Boundary force
        norm_i = np.linalg.norm(current[i])
        if norm_i > 1e-10:
            # Push inward if too close to boundary
            boundary_margin = 1.0 - norm_i
            if boundary_margin < 0.15:
                force -= (current[i] / norm_i) * (0.15 - boundary_margin)
        
        # Pair forces
        for k in range(n_pairs):
            a, b = iu[0][k], iu[1][k]
            if a == i:
                j = b
            elif b == i:
                j = a
            else:
                continue
            diff = current[i] - current[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                # Random push
                force += rng.randn(2) * 0.01
                continue
            # If circles are too close, push apart
            # Target separation ~ 2*current_r
            target_sep = 2.0 * max(current_r, 0.05)
            if d < target_sep:
                force += (diff / d) * (target_sep - d) * 0.5
        
        # Propose a move
        move_size = 0.02 + 0.03 * (T / T_start)
        proposed = current.copy()
        proposed[i] = current[i] + force * move_size + rng.randn(2) * move_size * 0.3
        
        # Keep inside container (hard constraint on position, not radius)
        norm_prop = np.linalg.norm(proposed[i])
        if norm_prop > 0.98:
            proposed[i] = proposed[i] * (0.98 / norm_prop)
        
        proposed_r = _compute_radius(proposed)
        delta_E = -proposed_r + current_r  # positive means worse
        
        if delta_E < 0 or rng.rand() < np.exp(-delta_E / T):
            current = proposed
            current_r = proposed_r
            if current_r > best_r:
                best_r = current_r
                best_pts = current.copy()
        
        T *= cooling
    
    return best_pts, best_r


def _graph_refine(pts, deadline, seed=0):
    """
    Graph-based refinement: build contact graph, then do local optimization
    where we only consider moves that affect connected components.
    """
    rng = np.random.RandomState(seed)
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Build contact graph (edges where distance < 2.2 * r)
    r = best_r
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    contact_thresh = 2.2 * r
    
    for iteration in range(3000):
        if time.time() > deadline - 5:
            break
        
        # Rebuild contact graph
        r = _compute_radius(best_pts)
        dist_matrix = cdist(best_pts, pts)  # placeholder, recompute
        dist_matrix = cdist(best_pts, best_pts)
        contact_mask = dist_matrix[iu] < 2.2 * r
        
        # Find the most constrained circle (smallest margin)
        norms = np.linalg.norm(best_pts, axis=1)
        margins = np.minimum(1.0 - norms, dist_matrix[iu].min(axis=1) / 2.0)
        # Actually compute per-circle min margin
        circle_margins = 1.0 - norms
        for k in range(n_pairs):
            a, b = iu[0][k], iu[1][k]
            d = dist_matrix[a, b] / 2.0
            circle_margins[a] = min(circle_margins[a], d)
            circle_margins[b] = min(circle_margins[b], d)
        
        worst = np.argmin(circle_margins)
        
        # Try to move the worst circle
        best_move_r = best_r
        best_move = best_pts[worst].copy()
        
        for attempt in range(20):
            angle = rng.rand() * 2 * np.pi
            step = 0.01 * (1 + rng.rand())
            new_pos = best_pts[worst] + np.array([np.cos(angle), np.sin(angle)]) * step
            norm_new = np.linalg.norm(new_pos)
            if norm_new > 0.97:
                new_pos = new_pos * (0.97 / norm_new)
            
            trial = best_pts.copy()
            trial[worst] = new_pos
            trial_r = _compute_radius(trial)
            if trial_r > best_move_r:
                best_move_r = trial_r
                best_move = new_pos
        
        if best_move_r > best_r:
            best_pts[worst] = best_move
            best_r = best_move_r
        else:
            # Occasionally accept a small random perturbation to escape
            if rng.rand() < 0.1:
                trial = best_pts.copy()
                trial[worst] = best_pts[worst] + rng.randn(2) * 0.005
                norm_new = np.linalg.norm(trial[worst])
                if norm_new > 0.97:
                    trial[worst] = trial[worst] * (0.97 / norm_new)
                trial_r = _compute_radius(trial)
                if trial_r > best_r * 0.995:  # allow small degradation
                    best_pts = trial
                    best_r = trial_r
    
    return best_pts, best_r


def _lbfgs_optimize(pts, beta, maxiter, deadline):
    """Run L-BFGS-B with given beta."""
    if time.time() > deadline - 3:
        return pts, _compute_radius(pts)
    
    def objective(x):
        return -_soft_min_radius(x.reshape(-1, 2), beta)
    
    def gradient(x):
        return -_soft_min_radius_grad(x.reshape(-1, 2), beta)
    
    try:
        res = minimize(objective, pts.flatten(), jac=gradient, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        new_pts = res.x.reshape(-1, 2)
        return new_pts, _compute_radius(new_pts)
    except Exception:
        return pts, _compute_radius(pts)


def _nelder_mead_polish(pts, deadline):
    """Polish with Nelder-Mead."""
    if time.time() > deadline - 3:
        return pts
    
    def objective(x):
        return -_compute_radius(x.reshape(-1, 2))
    
    try:
        res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14})
        new_pts = res.x.reshape(-1, 2)
        if _compute_radius(new_pts) > _compute_radius(pts):
            return new_pts
    except Exception:
        pass
    return pts


def _hex_init(spacing):
    """Generate hexagonal lattice points."""
    k = int(1.0 / spacing) + 2
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append([x, y])
    points = np.array(rows)
    radius = np.linalg.norm(points, axis=1)
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    return points


def _structured_init():
    """Structured initialization based on known packing patterns."""
    points = [[0.0, 0.0]]
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 0.55
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    r3 = 0.82
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    r4 = 0.95
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    return np.array(points)


def _random_init(seed):
    """Random initialization."""
    np.random.seed(seed)
    points = []
    for _ in range(N):
        r = np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7 second safety margin
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal lattices
    for spacing in [0.19, 0.20, 0.21, 0.22, 0.24]:
        pts = _hex_init(spacing)
        if len(pts) >= N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii, kind='stable')[:N]
            inits.append(pts[idx])
    
    # Structured
    struct_pts = _structured_init()
    if len(struct_pts) >= N:
        inits.append(struct_pts[:N])
    
    # Random
    for seed in [42, 123, 456, 789, 1024, 2048, 3141, 5678]:
        if time.time() > deadline - 40:
            break
        inits.append(_random_init(seed))
    
    # Filter valid
    valid_inits = []
    for pts in inits:
        if len(pts) == N and np.all(np.isfinite(pts)):
            r = _compute_radius(pts)
            if r > 0:
                valid_inits.append((pts, r))
    
    if not valid_inits:
        return _random_init(42)
    
    valid_inits.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = valid_inits[0][0]
    best_r = valid_inits[0][1]
    
    # Phase 1: L-BFGS optimization from top inits
    for i, (init_pts, init_r) in enumerate(valid_inits[:5]):
        if time.time() > deadline - 50:
            break
        
        # Multi-stage beta annealing with L-BFGS
        current = init_pts.copy()
        for beta in [30, 80, 150, 300, 600]:
            if time.time() > deadline - 45:
                break
            current, r = _lbfgs_optimize(current, beta, 60, deadline)
            if r > best_r:
                best_pts = current
                best_r = r
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - 30:
        sa_pts, sa_r = _simulated_annealing(best_pts, deadline, seed=1)
        if sa_r > best_r:
            best_pts = sa_pts
            best_r = sa_r
    
    # Phase 3: Graph refinement
    if time.time() < deadline - 15:
        gr_pts, gr_r = _graph_refine(best_pts, deadline, seed=2)
        if gr_r > best_r:
            best_pts = gr_pts
            best_r = gr_r
    
    # Phase 4: Final L-BFGS polish with high beta
    if time.time() < deadline - 8:
        polished, pr = _lbfgs_optimize(best_pts, 1000, 100, deadline)
        if pr > best_r:
            best_pts = polished
            best_r = pr
    
    # Phase 5: Nelder-Mead final polish
    if time.time() < deadline - 3:
        polished = _nelder_mead_polish(best_pts, deadline)
        r = _compute_radius(polished)
        if r > best_r:
            best_pts = polished
            best_r = r
    
    # Final validity check
    if not np.all(np.isfinite(best_pts)) or len(best_pts) != N:
        return _random_init(42)
    
    return best_pts
# EVOLVE-BLOCK-END
