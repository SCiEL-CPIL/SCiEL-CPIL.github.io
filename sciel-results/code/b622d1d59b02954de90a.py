import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))

def _pairwise_distances(points):
    """Compute all pairwise distances."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)

def _score(points):
    """Compute the packing radius."""
    points = points.reshape(-1, 2)
    n = len(points)
    boundary = 1.0 - np.linalg.norm(points, axis=1)
    pd = _pairwise_distances(points)
    iu = np.triu_indices(n, 1)
    pair_dists = pd[iu] / 2.0
    return min(np.min(boundary), np.min(pair_dists))

def _softmin_logsumexp(vals, beta):
    """Numerically stable soft-min via log-sum-exp."""
    m = np.min(vals)
    return m + np.log(np.sum(np.exp(-beta * (vals - m)))) / beta

def _smooth_objective(x, beta):
    """
    Smooth surrogate: -softmin of all constraints.
    We maximize r, so we minimize -softmin.
    """
    c = x.reshape(N, 2)
    pair_idx = _pairwise_indices(N)
    ci = c[pair_idx[:, 0]]
    cj = c[pair_idx[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2.0
    b = 1.0 - np.hypot(c[:, 0], c[:, 1])
    all_vals = np.concatenate([d, b])
    return -_softmin_logsumexp(all_vals, beta)

def _smooth_grad(x, beta):
    """Gradient of the smooth objective."""
    c = x.reshape(N, 2)
    pair_idx = _pairwise_indices(N)
    ci = c[pair_idx[:, 0]]
    cj = c[pair_idx[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2.0
    b = 1.0 - np.hypot(c[:, 0], c[:, 1])
    all_vals = np.concatenate([d, b])
    
    m = np.min(all_vals)
    weights = np.exp(-beta * (all_vals - m))
    weights /= np.sum(weights)
    
    grad_c = np.zeros((N, 2))
    n_pairs = len(d)
    
    for idx in range(n_pairs):
        w = -weights[idx]  # negative because we're minimizing -softmin
        if abs(w) < 1e-15:
            continue
        i, j = pair_idx[idx, 0], pair_idx[idx, 1]
        d_vec = c[i] - c[j]
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 1e-12:
            continue
        unit = d_vec / d_norm
        grad_c[i] += w * unit * 0.5
        grad_c[j] -= w * unit * 0.5
    
    for idx in range(N):
        w = -weights[n_pairs + idx]
        if abs(w) < 1e-15:
            continue
        norm = np.linalg.norm(c[idx])
        if norm < 1e-12:
            continue
        grad_c[idx] -= w * c[idx] / norm
    
    return grad_c.flatten()

def _ring_initialization(n):
    """
    Ring-structured initialization: place circles on concentric rings.
    Returns (points, ring_assignment) where ring_assignment[i] = ring index.
    """
    # Try different ring configurations
    best_points = None
    best_r = 0.0
    
    for num_rings in range(4, 8):
        # Distribute n circles across rings
        # Ring 0: 1 circle at center
        # Ring k: approximately k*6 circles (hexagonal-like)
        ring_counts = [1]
        remaining = n - 1
        for k in range(1, num_rings):
            count = min(remaining, max(4, k * 6))
            if k == num_rings - 1:
                count = remaining
            ring_counts.append(count)
            remaining -= count
            if remaining <= 0:
                break
        
        if remaining > 0:
            continue
        
        points = []
        ring_assign = []
        
        # Center
        points.append([0.0, 0.0])
        ring_assign.append(0)
        
        # Rings
        for k in range(1, len(ring_counts)):
            count = ring_counts[k]
            if count <= 0:
                continue
            ring_radius = k * 0.22
            for m in range(count):
                angle = 2 * np.pi * m / count + np.pi / (2 * count) * (k % 2)
                points.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
                ring_assign.append(k)
        
        points = np.array(points[:n])
        if len(points) < n:
            continue
        
        r = _score(points)
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.3
    
    return best_points

def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.08, 0.22, 30):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.4
    return best_points

def _anneal_points(points, beta_start=10.0, beta_end=500.0, n_iterations=100, time_limit=10.0):
    """
    Progressive beta annealing with L-BFGS-B at each stage.
    This is the core optimization: start with low beta (smooth) and increase
    to high beta (sharp, approaching the true min).
    """
    t_start = time.time()
    x = points.flatten().copy()
    
    # Beta schedule: geometric progression
    betas = np.geomspace(beta_start, beta_end, n_iterations)
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
        
        try:
            res = minimize(
                _smooth_objective,
                x,
                args=(beta,),
                jac=lambda xx, bb: _smooth_grad(xx, bb),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res.fun < _smooth_objective(x, beta):
                x = res.x
        except Exception:
            pass
    
    return x.reshape(N, 2)

def _binary_search_polish(points, time_limit=5.0):
    """
    Binary search on radius: check if packing with radius r is feasible,
    using L-BFGS-B to find a feasible configuration.
    """
    t_start = time.time()
    best_pts = points.copy()
    best_r = _score(points)
    
    lo = best_r
    hi = 0.15
    
    while time.time() - t_start < time_limit and hi - lo > 1e-7:
        mid = (lo + hi) / 2
        # Try to find a packing with radius >= mid
        # Use soft-min with high beta as feasibility check
        try:
            res = minimize(
                _smooth_objective,
                best_pts.flatten(),
                args=(1000.0,),
                jac=lambda xx, bb: _smooth_grad(xx, bb),
                method='L-BFGS-B',
                options={'maxiter': 30, 'ftol': 1e-12},
            )
            new_pts = res.x.reshape(N, 2)
            new_r = _score(new_pts)
            if new_r >= mid:
                lo = mid
                best_pts = new_pts
                best_r = new_r
            else:
                hi = mid
        except Exception:
            hi = mid
    
    return best_pts, best_r

def _de_ring_optimize(points, time_limit=15.0):
    """
    De-ring optimization: start from ring structure, progressively
    allow points to move off their assigned rings.
    """
    t_start = time.time()
    x = points.flatten().copy()
    
    # Progressive beta annealing
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
        try:
            res = minimize(
                _smooth_objective,
                x,
                args=(beta,),
                jac=lambda xx, bb: _smooth_grad(xx, bb),
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-13, 'gtol': 1e-11},
            )
            if res.fun < _smooth_objective(x, beta):
                x = res.x
        except Exception:
            pass
    
    return x.reshape(N, 2)

def construct():
    t_start = time.time()
    
    # Multiple initializations
    init1 = _ring_initialization(N)
    init2 = _hexagonal_init(N)
    init3 = np.random.randn(N, 2) * 0.35
    
    best_points = None
    best_score = 0.0
    
    for init in [init1, init2, init3]:
        r = _score(init)
        if r > best_score:
            best_score = r
            best_points = init.copy()
    
    # Phase 1: Progressive beta annealing from best init
    if time.time() < DEADLINE - 40:
        remaining = DEADLINE - (time.time() - t_start) - 40
        c_opt = _anneal_points(best_points, beta_start=5.0, beta_end=300.0, 
                               n_iterations=60, time_limit=min(25.0, remaining))
        r_opt = _score(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Phase 2: De-ring optimization (more aggressive)
    if time.time() < DEADLINE - 25:
        remaining = DEADLINE - (time.time() - t_start) - 25
        c_opt = _de_ring_optimize(best_points, time_limit=min(15.0, remaining))
        r_opt = _score(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Phase 3: Binary search polish
    if time.time() < DEADLINE - 15:
        remaining = DEADLINE - (time.time() - t_start) - 15
        polished, r_polish = _binary_search_polish(best_points, time_limit=min(10.0, remaining))
        if r_polish > best_score:
            best_score = r_polish
            best_points = polished.copy()
    
    # Phase 4: Multi-restart with perturbations
    n_restarts = 0
    while time.time() < DEADLINE - 10 and n_restarts < 8:
        remaining = DEADLINE - (time.time() - t_start) - 10
        if remaining < 8:
            break
        
        pert_scale = 0.02 / (1.0 + 0.2 * n_restarts)
        x0 = best_points + np.random.randn(N, 2) * pert_scale
        
        c_opt = _anneal_points(x0, beta_start=10.0, beta_end=400.0,
                               n_iterations=40, time_limit=min(8.0, remaining))
        r_opt = _score(c_opt)
        
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
        
        n_restarts += 1
    
    # Phase 5: Final high-precision polish
    if time.time() < DEADLINE - 3:
        remaining = DEADLINE - (time.time() - t_start) - 3
        try:
            res = minimize(
                _smooth_objective,
                best_points.flatten(),
                args=(1000.0,),
                jac=lambda xx, bb: _smooth_grad(xx, bb),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            final_pts = res.x.reshape(N, 2)
            final_r = _score(final_pts)
            if final_r > best_score:
                best_points = final_pts
                best_score = final_r
        except Exception:
            pass
    
    # Ensure validity
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1.0:
            scale = (1.0 - 1e-10) / r2[i]
            c_final[i] *= scale
    
    r_check = _score(c_final)
    if r_check < best_score - 1e-12:
        c_final *= (1.0 - 1e-10)
    
    return c_final
