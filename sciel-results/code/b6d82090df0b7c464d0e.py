"""Pack 50 equal circles in the unit disk: soft-min L-BFGS-B + simulated annealing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _raw_score(centers):
    """
    Calculate the maximum radius r such that all circles fit.
    r = min( min_pair_dist/2, min_edge_dist )
    """
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    centers = centers.reshape(N, 2)
    norms = np.sqrt(np.sum(centers**2, axis=1))
    if np.any(norms > 1.0 + 1e-9):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_edge_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_edge_dist)
    return max(0.0, r)

def _soft_min_objective(centers, beta=100.0):
    """
    Objective function for L-BFGS-B: minimize -soft_min(constraints)
    """
    centers = centers.reshape(N, 2)
    norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
    edge_constraints = 1.0 - norms
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    neg_beta_x = -beta * all_constraints
    max_val = np.max(neg_beta_x)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val

def _soft_min_gradient(centers, beta=100.0):
    """
    Gradient of the soft-min objective.
    """
    centers = centers.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
    edge_constraints = 1.0 - norms
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    neg_beta_x = -beta * all_constraints
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Edge gradient contribution
    edge_w = weights[:N]
    grad += edge_w[:, np.newaxis] * (centers / norms[:, np.newaxis])
    
    # Pair gradient contribution
    n_pairs = len(pair_constraints)
    i_idx = iu[0]
    j_idx = iu[1]
    w_pair = weights[N:N+n_pairs]
    
    diff_pairs = centers[i_idx] - centers[j_idx]
    dist_pairs = dists[i_idx, j_idx]
    dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
    
    np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_vecs)
    np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_vecs)
    
    return grad.reshape(-1)

def _refine(centers, beta=300.0, maxiter=200):
    """
    Local refinement using L-BFGS-B with soft-min objective.
    """
    try:
        result = minimize(
            _soft_min_objective,
            centers.flatten(),
            args=(beta,),
            jac=lambda x, b: _soft_min_gradient(x, b),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        refined = result.x.reshape(N, 2)
        # Ensure points stay inside
        norms = np.sqrt(np.sum(refined**2, axis=1))
        mask = norms > 1.0
        if np.any(mask):
            refined[mask] *= (1.0 - 1e-6) / norms[mask, np.newaxis]
        return refined
    except Exception:
        return centers

def _simulated_annealing(centers, steps=1500, T_init=0.015, T_final=0.0003):
    """
    Simulated Annealing for global exploration.
    """
    current_centers = centers.copy()
    current_score = _raw_score(current_centers)
    best_centers = current_centers.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, 1)
    
    for i in range(steps):
        if time.time() > time.time() + 10.0:  # placeholder, will be replaced
            break
            
        T = T_init * (T_final / T_init) ** (i / steps)
        
        # Move 1-3 random points
        num_moves = np.random.randint(1, 4)
        indices = np.random.choice(N, num_moves, replace=False)
        
        new_centers = current_centers.copy()
        step_size = max(T * 0.5, 0.003)
        moves = np.random.normal(0, step_size, size=(num_moves, 2))
        new_centers[indices] += moves
        
        # Keep inside unit circle
        norms = np.sqrt(np.sum(new_centers**2, axis=1))
        out_mask = norms > 0.9995
        if np.any(out_mask):
            new_centers[out_mask] *= 0.9995 / norms[out_mask, np.newaxis]
            
        new_score = _raw_score(new_centers)
        delta = new_score - current_score
        
        if delta > 0:
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = new_centers.copy()
        else:
            clamped_delta = max(delta, -10.0 * T)
            if np.random.rand() < np.exp(clamped_delta / T):
                current_centers = new_centers
                current_score = new_score
    
    return best_centers, best_score

def _make_rings(radii, counts, offsets=None):
    """Generate ring-based initial configuration."""
    centers = []
    centers.append([0.0, 0.0])
    start_idx = 1
    for r, c in zip(radii, counts):
        base_offset = offsets[start_idx-1] if offsets else 0.0
        for k in range(c):
            theta = 2 * np.pi * k / c + base_offset
            centers.append([r * np.cos(theta), r * np.sin(theta)])
        start_idx += 1
    return np.array(centers[:N])

def _hexagonal_seed(n, jitter=0.0):
    """Generate hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) >= N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        pts = points[idx]
        if jitter > 0:
            pts += np.random.normal(0, jitter, pts.shape)
            # Project back inside
            norms = np.sqrt(np.sum(pts**2, axis=1))
            out_mask = norms > 0.999
            if np.any(out_mask):
                pts[out_mask] *= 0.999 / norms[out_mask, np.newaxis]
        return pts
    return None

def construct():
    """Main entry point: find best packing of 50 circles in unit disk."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Ring-based initializations
    r1 = [0.0, 0.24, 0.46, 0.68, 0.88]
    c1 = [1, 6, 12, 18, 13]
    initializations.append(_make_rings(r1, c1))
    
    # Perturbed rings
    for seed in range(10):
        np.random.seed(seed)
        perturbed_radii = [r1[0]] + [r + np.random.uniform(-0.02, 0.02) for r in r1[1:]]
        offsets = [0.0] * len(r1)
        for s in range(1, len(r1)):
            offsets[s] = np.random.uniform(-0.15, 0.15)
        initializations.append(_make_rings(perturbed_radii, c1, offsets))
    
    # Different ring count distributions
    ring_configs = [
        ([0.0, 0.25, 0.5, 0.75, 0.9], [1, 6, 12, 18, 13]),
        ([0.0, 0.22, 0.44, 0.66, 0.85], [1, 7, 14, 14, 14]),
        ([0.0, 0.2, 0.4, 0.6, 0.8, 0.95], [1, 5, 10, 15, 10, 9]),
        ([0.0, 0.18, 0.38, 0.58, 0.78, 0.92], [1, 6, 11, 16, 10, 6]),
        ([0.0, 0.2, 0.45, 0.7, 0.9], [1, 8, 16, 14, 11]),
    ]
    for radii, counts in ring_configs:
        total = sum(counts)
        if total < N:
            counts[-1] += N - total
        elif total > N:
            counts[-1] -= total - N
            if counts[-1] <= 0:
                continue
        cfg = _make_rings(radii, counts)
        if cfg is not None and len(cfg) == N:
            initializations.append(cfg)
    
    # 2. Hexagonal lattice initializations
    hex_pts = _hexagonal_seed(N)
    if hex_pts is not None:
        initializations.append(hex_pts)
        # With jitter
        for j in [0.01, 0.02, 0.05]:
            hex_jit = _hexagonal_seed(N, jitter=j)
            if hex_jit is not None:
                initializations.append(hex_jit)
    
    # 3. Random initializations
    for seed in range(5):
        np.random.seed(100 + seed)
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)
    
    # Filter valid initializations
    valid_inits = []
    for init in initializations:
        if init is not None and len(init) == N and np.all(np.isfinite(init)):
            norms = np.sqrt(np.sum(init**2, axis=1))
            if np.all(norms <= 1.0):
                valid_inits.append(init)
    
    if not valid_inits:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        radii = np.sqrt(np.linspace(0, 1, N))
        valid_inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Optimize each initialization
    best_points = None
    best_score = 0.0
    
    for idx, init in enumerate(valid_inits):
        if time.time() > deadline - 20.0:
            break
        
        # Simulated annealing phase
        sa_time_budget = 30.0
        sa_steps = int(sa_time_budget * 100)  # rough estimate
        sa_points, sa_score = _simulated_annealing(init, steps=min(sa_steps, 2000), 
                                                    T_init=0.015, T_final=0.0003)
        
        # Refine with L-BFGS-B
        refined_points = _refine(sa_points, beta=300.0, maxiter=100)
        refined_score = _raw_score(refined_points)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
        
        # Try another refinement with different beta
        refined_points2 = _refine(refined_points, beta=500.0, maxiter=100)
        refined_score2 = _raw_score(refined_points2)
        
        if refined_score2 > best_score:
            best_score = refined_score2
            best_points = refined_points2.copy()
    
    # Final refinement on best
    if best_points is not None and time.time() < deadline - 5.0:
        final_points = _refine(best_points, beta=500.0, maxiter=200)
        final_score = _raw_score(final_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Ensure we return a valid result
    if best_points is None or len(best_points) != N:
        angles = 2 * np.pi * np.arange(N) / N
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Final validity check
    norms = np.sqrt(np.sum(best_points**2, axis=1))
    out_mask = norms > 1.0
    if np.any(out_mask):
        best_points[out_mask] *= (1.0 - 1e-6) / norms[out_mask, np.newaxis]
    
    return best_points
