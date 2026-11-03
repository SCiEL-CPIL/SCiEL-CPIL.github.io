"""
Packing 50 equal circles in a unit disk.
Uses ring-structured initialization with differential evolution for global
exploration, combined with multi-stage smooth annealing (log-sum-exp soft-min)
and constraint push polishing. The ring structure provides a strong geometric
prior for circle packing, while DE explores variations in ring radii and
angular offsets.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_MARGIN = 15.0  # seconds before hard limit


def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score: minimum of half pairwise distances and boundary distances."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return max(min(min_pair, min_boundary), 0.0)


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp for numerical stability."""
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum == 0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    """
    Smooth objective: minimize -softmin of all constraint margins.
    The softmin approximates the minimum, so minimizing -softmin maximizes the minimum.
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    
    # Get all finite values
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 1e10
    
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _hexagonal_init(n):
    """
    Initialize with a hexagonal lattice structure.
    This is a strong starting point for circle packing.
    """
    a = 1.0
    sqrt3 = np.sqrt(3)
    
    def hex_to_cart(q, r):
        x = a * 1.5 * q
        y = a * sqrt3 * (r + q / 2.0)
        return np.array([x, y])
    
    max_ring = 5
    candidates = []
    for q in range(-max_ring, max_ring + 1):
        for r in range(-max_ring, max_ring + 1):
            candidates.append((q, r))
            
    cart_points = []
    for q, r in candidates:
        p = hex_to_cart(q, r)
        d = np.linalg.norm(p)
        cart_points.append((d, p))
        
    cart_points.sort(key=lambda x: x[0])
    selected_points = [p for d, p in cart_points[:N]]
    points = np.array(selected_points)
    
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        scale = 0.8 / max_r
        points = points * scale
        
    return points


def _ring_to_points(ring_params):
    """
    Convert ring parameters to point positions.
    ring_params: array of shape (n_rings, 3) where each row is [radius, count, angle_offset]
    Returns: array of shape (N, 2)
    """
    centers = []
    for i in range(len(ring_params)):
        radius, count, angle_offset = ring_params[i]
        if count < 1:
            continue
        if count == 1 and radius < 1e-10:
            centers.append(np.array([0.0, 0.0]))
        else:
            angles = np.linspace(0, 2 * np.pi, int(count), endpoint=False) + angle_offset
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for j in range(int(count)):
                centers.append(np.array([xs[j], ys[j]]))
    
    pts = np.array(centers[:N])
    if len(pts) < N:
        # Pad with points near origin if needed
        while len(pts) < N:
            pts = np.vstack([pts, np.array([0.0, 0.0])])
    return pts[:N]


def _constraint_push(pts, r_target, max_iters=200):
    """
    Iteratively push violating constraints to satisfy minimum distance r_target.
    This is a simple coordinate descent on the constraint margins.
    """
    pts = pts.copy().astype(np.float64)
    
    for _ in range(max_iters):
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        # Find the most violated constraint
        min_pair_idx = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
        min_boundary_idx = np.argmin(boundary_margins)
        
        min_pair_margin = pair_margins[min_pair_idx]
        min_boundary_margin = boundary_margins[min_boundary_idx]
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            i, j = min_pair_idx
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = min_boundary_idx
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check if satisfied
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _random_init(n, seed=None):
    """Random initialization inside the unit circle."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0, 0.9, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _farthest_point_sample(n_points, max_iter=5000):
    """
    Sequential farthest-point sampling: place each new center at the location
    that maximizes the minimum distance to all previously placed centers and
    the container boundary.
    Uses a grid-based search for efficiency.
    """
    pts = np.zeros((n_points, 2))
    
    # First point at origin
    pts[0] = [0.0, 0.0]
    
    # Grid for candidate positions
    n_grid = 60
    theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    r_grid = np.linspace(0.05, 0.95, 35)
    
    for k in range(1, n_points):
        best_score = -np.inf
        best_pt = None
        
        for r in r_grid:
            for t in theta:
                candidate = np.array([r * np.cos(t), r * np.sin(t)])
                dist_to_boundary = 1.0 - np.linalg.norm(candidate)
                
                if dist_to_boundary <= 0:
                    continue
                
                dists_to_placed = np.linalg.norm(pts[:k] - candidate, axis=1)
                min_dist = np.min(dists_to_placed) if k > 0 else np.inf
                
                score = min(min_dist / 2.0, dist_to_boundary)
                
                if score > best_score:
                    best_score = score
                    best_pt = candidate
        
        if best_pt is None:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.1, 0.9)
            best_pt = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        
        pts[k] = best_pt
    
    return pts


def _de_ring_optimize(ring_counts, deadline, max_generations=50):
    """
    Differential Evolution optimization over ring parameters.
    ring_counts: list of circle counts per ring (must sum to N)
    Optimizes: [radius_0, radius_1, ..., angle_0, angle_1, ...]
    """
    n_rings = len(ring_counts)
    
    def ring_obj(params):
        radii = params[:n_rings]
        angles = params[n_rings:]
        
        # Ensure radii are in valid range
        radii = np.clip(radii, 0.0, 0.99)
        
        ring_params = np.column_stack([radii, np.array(ring_counts, dtype=float), angles])
        pts = _ring_to_points(ring_params)
        
        if not np.all(np.isfinite(pts)):
            return 1e10
        
        score = _score(pts)
        return -score
    
    # Initial guess: evenly spaced radii
    total = sum(ring_counts)
    if total != N:
        # Adjust to match N
        while sum(ring_counts) > N:
            idx = np.argmax(ring_counts)
            ring_counts[idx] -= 1
        while sum(ring_counts) < N:
            idx = np.argmin(ring_counts)
            ring_counts[idx] += 1
    
    # Bounds for DE
    bounds = [(0.0, 0.99)] * n_rings + [(0.0, 2 * np.pi)] * n_rings
    
    # Initial population
    rng = np.random.RandomState(42)
    x0 = np.array([0.2 + 0.1 * i for i in range(n_rings)] + [0.0] * n_rings)
    
    try:
        from scipy.optimize import differential_evolution
        result = differential_evolution(
            ring_obj,
            bounds,
            maxiter=max_generations,
            popsize=15,
            tol=1e-10,
            seed=42,
            polish=True,
            x0=x0
        )
        
        radii = result.x[:n_rings]
        angles = result.x[n_rings:]
        ring_params = np.column_stack([radii, np.array(ring_counts, dtype=float), angles])
        pts = _ring_to_points(ring_params)
        
        if np.all(np.isfinite(pts)):
            return pts, -result.fun
        else:
            return None, 0.0
            
    except Exception as e:
        return None, 0.0


def _repulsion_dynamics(pts, deadline, target_r=None, max_steps=5000, dt=0.002):
    """
    Particle-based repulsion dynamics with Lennard-Jones-like forces.
    Circles repel each other when too close and are attracted to the center.
    Uses simulated annealing: temperature decreases over time.
    
    This is a STRUCTURAL change: instead of directly optimizing the min-margin
    objective, we model the packing as a physical system and let dynamics find
    a low-energy configuration.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    # Target separation distance
    if target_r is None:
        target_r = _score(pts) + 0.005
    target_sep = 2.0 * target_r
    
    # Force parameters
    repulsion_strength = 50.0
    attraction_strength = 1.0
    boundary_strength = 100.0
    
    # Annealing schedule
    initial_temp = 0.05
    final_temp = 0.001
    n_anneal_steps = max_steps
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        # Temperature for this step
        t_frac = step / max(n_anneal_steps - 1, 1)
        temp = initial_temp * (final_temp / initial_temp) ** t_frac
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion forces
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = d[i, j]
                if dist < 1e-10:
                    # Random small perturbation
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                    continue
                
                # Repulsion: strong when closer than target_sep
                if dist < target_sep:
                    # Lennard-Jones-like: F ~ (target_sep/dist)^12 - (target_sep/dist)^6
                    ratio = target_sep / dist
                    force_mag = repulsion_strength * (ratio ** 12 - ratio ** 6)
                    # Cap the force to avoid instability
                    force_mag = np.clip(force_mag, -100, 100)
                    dir_ij = (pts[i] - pts[j]) / dist
                    forces[i] += force_mag * dir_ij
                    forces[j] -= force_mag * dir_ij
                elif dist < target_sep * 1.5:
                    # Gentle attraction to maintain spacing
                    ratio = target_sep / dist
                    force_mag = -repulsion_strength * 0.1 * (ratio ** 6)
                    dir_ij = (pts[i] - pts[j]) / dist
                    forces[i] += force_mag * dir_ij
                    forces[j] -= force_mag * dir_ij
        
        # Boundary attraction: push points inward if near boundary
        for i in range(n):
            r_i = np.linalg.norm(pts[i])
            if r_i > 0.95:
                # Strong push inward
                dir_in = -pts[i] / r_i
                force_mag = boundary_strength * (r_i - 0.95) ** 2
                forces[i] += force_mag * dir_in
            elif r_i > 0.85:
                # Gentle pull inward
                dir_in = -pts[i] / r_i
                force_mag = attraction_strength * (r_i - 0.85)
                forces[i] += force_mag * dir_in
        
        # Add thermal noise
        noise = temp * np.random.randn(n, 2)
        
        # Update positions
        new_pts = pts + dt * (forces + noise)
        
        # Project inside unit circle
        norms = np.linalg.norm(new_pts, axis=1)
        for i in range(n):
            if norms[i] > 0.99:
                new_pts[i] *= 0.99 / norms[i]
        
        # Accept or reject based on Metropolis criterion
        new_score = _score(new_pts)
        if new_score >= best_score:
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            # Metropolis acceptance
            delta = best_score - new_score
            if np.random.rand() < np.exp(-delta / (temp + 1e-10)):
                pts = new_pts
    
    return best_pts, best_score


def _optimize_from_init(pts, deadline, max_beta_stages=5):
    """Multi-stage annealing optimization from initial points."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Progressive beta values for smooth optimization
    betas = [10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # High-precision polish with larger betas
    if time.time() < deadline - 8:
        for beta in [3000.0, 10000.0]:
            if time.time() > deadline - 5:
                break
            x0 = current_pts.ravel()
            try:
                result = minimize(
                    _smooth_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000}
                )
                new_pts = result.x.reshape(N, 2)
                if not np.all(np.isfinite(new_pts)):
                    continue
                new_score = _score(new_pts)
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
            except Exception:
                pass
    
    return current_pts, current_score


def _binary_search_polish(pts, r_target, deadline, max_iters=300):
    """
    Polish by iteratively pushing points to achieve target radius r_target.
    """
    pts = pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _ring_initialization(ring_radii, ring_counts, angles_offset=None):
    """Generate points from a ring structure."""
    centers = []
    for idx, (count, radius) in enumerate(zip(ring_counts, ring_radii)):
        if count == 1 and radius < 1e-10:
            centers.append(np.array([0.0, 0.0]))
        else:
            if angles_offset is not None and idx < len(angles_offset):
                offset = angles_offset[idx]
            else:
                offset = 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for i in range(count):
                centers.append(np.array([xs[i], ys[i]]))
    
    pts = np.array(centers[:N])
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 135s effective deadline
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Ring-structured DE for global exploration (PRIMARY)
    # Try multiple ring configurations
    ring_configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 15, 16],
        [1, 6, 14, 16, 13],
        [1, 8, 12, 14, 15],
        [1, 6, 10, 16, 17],
    ]
    
    for ring_counts in ring_configs:
        if time.time() > deadline - 30:
            break
        de_pts, de_score = _de_ring_optimize(list(ring_counts), deadline, max_generations=30)
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_points = de_pts
    
    # Strategy 2: Polish DE results with multi-stage smooth annealing
    if best_points is not None and time.time() < deadline - 20:
        opt_pts, opt_score = _optimize_from_init(best_points, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Strategy 3: Hexagonal init + smooth annealing
    if time.time() < deadline - 20:
        hex_pts = _hexagonal_init(N)
        opt_pts, opt_score = _optimize_from_init(hex_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Strategy 4: Farthest-point sampling + smooth annealing
    if time.time() < deadline - 15:
        fpts = _farthest_point_sample(N)
        opt_pts, opt_score = _optimize_from_init(fpts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Strategy 5: Random init + smooth annealing (diversity)
    if time.time() < deadline - 12:
        rand_pts = _random_init(N, seed=42)
        opt_pts, opt_score = _optimize_from_init(rand_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Strategy 6: Final high-precision polish with very high beta
    if best_points is not None and time.time() < deadline - 10:
        for beta in [3000.0, 10000.0]:
            if time.time() > deadline - 5:
                break
            x0 = best_points.ravel()
            try:
                result = minimize(
                    _smooth_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 100000}
                )
                new_pts = result.x.reshape(N, 2)
                if np.all(np.isfinite(new_pts)):
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts
            except Exception:
                pass
    
    if best_points is None:
        best_points = _hexagonal_init(N)
        best_score = _score(best_points)
    
    # Final constraint push polish
    if time.time() < deadline - 3:
        target_r = best_score + 1e-6
        polished = _constraint_push(best_points, target_r, max_iters=200)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points
