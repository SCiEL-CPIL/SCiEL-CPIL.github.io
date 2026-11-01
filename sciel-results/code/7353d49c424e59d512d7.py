import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _compute_score(points):
    """Compute the actual packing score (largest common radius)."""
    n = len(points)
    if n < 2:
        return 1.0 - np.linalg.norm(points[0]) if n == 1 else 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n, r_init=0.05):
    """Initialize points on a hexagonal lattice clipped to the unit circle."""
    spacing = 2.0 * r_init
    R_est = spacing * np.sqrt(n * np.sqrt(3.0) / (2.0 * np.pi))
    R_est *= 1.15
    k = int(np.ceil(R_est / spacing)) + 2
    points = []
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        if abs(y) > R_est:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if np.hypot(x, y) <= R_est:
                points.append((x, y))
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 2))
    if len(points) > n:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:n]]
    elif len(points) < n:
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            rad = np.random.uniform(0, 0.95)
            points = np.vstack([points, [rad * np.cos(angle), rad * np.sin(angle)]])
    points += np.random.normal(0, 1e-4, points.shape)
    return points

def _ring_init(n):
    """Initialize points in concentric rings."""
    # Estimate number of rings
    # r ~ 0.05, spacing ~ 0.1
    # Max radius 1.0
    # rings ~ 1.0 / 0.08 ~ 12
    num_rings = 12
    points = []
    # Center point
    if n > 0:
        points.append([0.0, 0.0])
        n -= 1
    
    # Distribute remaining points among rings
    # More points in outer rings
    # Linear distribution of counts?
    # Let's try to make ring circumferences proportional to count * spacing
    # r_k * 2*pi ~ n_k * d
    # Sum n_k = n
    # Simple heuristic: n_k = k
    # Sum 1..K = K(K+1)/2 = n -> K ~ sqrt(2n)
    # For n=300, K ~ 24.
    # But radius grows linearly with k? r_k ~ k * d/2
    # Let's just assign counts roughly proportional to radius
    counts = []
    remaining = n
    for k in range(1, num_rings + 1):
        # Heuristic count
        c = int(2 * np.pi * (k * 0.08) / 0.1) 
        c = max(1, c)
        if remaining <= 0:
            break
        counts.append(min(c, remaining))
        remaining -= counts[-1]
    
    # If we didn't use all points, add to last ring or create new ones
    if remaining > 0:
        if counts:
            counts[-1] += remaining
        else:
            counts = [remaining]
            
    # If we used too few rings, adjust
    # Actually, let's just generate points in rings with radii 0.1, 0.2, ...
    # and let the optimizer fix it.
    
    # Better init:
    # Place points in rings with radius r_k = k * (1.0 / num_rings)
    # Count in ring k: proportional to r_k
    weights = np.arange(1, num_rings + 1, dtype=float)
    weights /= np.sum(weights)
    counts = (weights * n).astype(int)
    # Fix rounding errors
    diff = n - np.sum(counts)
    if diff > 0:
        for i in range(diff):
            counts[i % len(counts)] += 1
    elif diff < 0:
        for i in range(-diff):
            counts[-1 - (i % len(counts))] -= 1
            if counts[-1 - (i % len(counts))] < 1:
                 # move to next
                 pass # simple fix, might fail if count becomes 0, but unlikely for outer rings

    points = []
    for k, count in enumerate(counts):
        radius = (k + 1) * (1.0 / num_rings) * 0.95 # keep inside
        if count == 0: continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        # Stagger phases
        angles += (k % 2) * np.pi / count
        for a in angles:
            points.append([radius * np.cos(a), radius * np.sin(a)])
            
    points = np.array(points)
    if len(points) < n:
        # fill with random
        while len(points) < n:
            x, y = np.random.uniform(-0.9, 0.9, 2)
            if x*x+y*y < 0.8:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
        
    points += np.random.normal(0, 1e-4, points.shape)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s limit

    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    n_pairs = len(i_idx)

    # Generate diverse initializations
    inits = []
    inits.append(_hex_lattice_init(N, 0.05))
    inits.append(_hex_lattice_init(N, 0.048))
    inits.append(_hex_lattice_init(N, 0.052))
    inits.append(_ring_init(N))
    
    # Random init
    random_pts = np.random.uniform(-1, 1, (N, 2))
    norms = np.linalg.norm(random_pts, axis=1)
    mask = norms > 0.9
    random_pts[mask] *= 0.9 / norms[mask, np.newaxis]
    inits.append(random_pts)

    best_points = None
    best_score = -np.inf

    def make_objective_and_grad(beta):
        def objective_and_grad(x):
            p = x.reshape(-1, 2)
            vecs = p[i_idx] - p[j_idx]
            pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
            norms = np.linalg.norm(p, axis=1)
            bound_dists = 1.0 - norms

            z_pairs = -beta * pair_dists
            z_bounds = -beta * bound_dists
            max_z = max(np.max(z_pairs), np.max(z_bounds))

            exp_pairs = np.exp(z_pairs - max_z)
            exp_bounds = np.exp(z_bounds - max_z)
            sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
            log_sum_exp = max_z + np.log(sum_exp)

            val = log_sum_exp / beta

            w_pairs = exp_pairs / sum_exp
            w_bounds = exp_bounds / sum_exp

            grad = np.zeros_like(x)

            norms_safe = norms.copy()
            norms_safe[norms_safe == 0] = 1.0
            grad += -p / norms_safe[:, np.newaxis] * w_bounds[:, np.newaxis]

            pair_full_dists = np.sqrt(np.sum(vecs**2, axis=1))
            pair_full_dists_safe = np.where(pair_full_dists > 1e-12, pair_full_dists, 1.0)
            coeff = w_pairs / pair_full_dists_safe

            np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
            np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)

            return val, grad
        return objective_and_grad

    bounds = [(-1.05, 1.05)] * (2 * N)

    # Annealing schedule
    beta_schedule = [
        (10.0, 100),
        (30.0, 100),
        (100.0, 80),
        (300.0, 60),
        (1000.0, 50),
        (3000.0, 40),
    ]

    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        current_points = init_points.copy()
        
        for beta, max_iter in beta_schedule:
            if time.time() > deadline - 10:
                break
            obj_grad = make_objective_and_grad(beta)
            try:
                res = minimize(obj_grad, current_points.flatten(), jac=True,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': max_iter, 'ftol': 1e-14,
                                        'gtol': 1e-10, 'disp': False})
                if res.x is not None:
                    candidate = res.x.reshape(-1, 2)
                    radii = np.hypot(candidate[:, 0], candidate[:, 1])
                    if np.all(radii < 1.0) and np.all(np.isfinite(candidate)):
                        score = _compute_score(candidate)
                        if score > best_score:
                            best_score = score
                            best_points = candidate.copy()
                        current_points = candidate
            except Exception:
                pass

    # Final polish on the best found
    if best_points is not None and time.time() < deadline - 5:
         obj_grad = make_objective_and_grad(5000.0)
         try:
             res = minimize(obj_grad, best_points.flatten(), jac=True,
                            method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 50, 'ftol': 1e-16,
                                     'gtol': 1e-12, 'disp': False})
             if res.x is not None:
                 candidate = res.x.reshape(-1, 2)
                 radii = np.hypot(candidate[:, 0], candidate[:, 1])
                 if np.all(radii < 1.0) and np.all(np.isfinite(candidate)):
                     score = _compute_score(candidate)
                     if score > best_score:
                         best_points = candidate
                         best_score = score
         except Exception:
             pass

    # Validity check
    if best_points is None:
        best_points = _hex_lattice_init(N, 0.05)
        
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii >= 1.0):
        max_r = np.max(radii)
        if max_r > 0:
            scale = 0.9999 / max_r
            best_points *= scale

    return best_points
