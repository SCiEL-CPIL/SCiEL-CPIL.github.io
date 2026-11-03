"""
Packing 50 equal circles in a unit disk.
Uses sequential farthest-point sampling with iterative re-optimization,
combined with smooth surrogate annealing and constraint polishing.
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


def _farthest_point_sample(pts_placed, n_points, max_iter=10000):
    """
    Sequential farthest-point sampling: place each new center at the location
    that maximizes the minimum distance to all previously placed centers and
    the container boundary (i.e., 1 - |p|).
    
    Uses a grid-based search for efficiency.
    """
    pts = np.zeros((n_points, 2))
    
    # First point at origin (symmetry)
    pts[0] = [0.0, 0.0]
    
    # Grid for candidate positions
    n_grid = 80
    theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    r_grid = np.linspace(0.05, 0.95, 40)
    
    for k in range(1, n_points):
        best_score = -np.inf
        best_pt = None
        
        for r in r_grid:
            for t in theta:
                candidate = np.array([r * np.cos(t), r * np.sin(t)])
                dist_to_boundary = 1.0 - np.linalg.norm(candidate)
                
                if dist_to_boundary <= 0:
                    continue
                
                # Minimum distance to placed points
                dists_to_placed = np.linalg.norm(pts[:k] - candidate, axis=1)
                min_dist = np.min(dists_to_placed) if k > 0 else np.inf
                
                # Combined score: min of half-pair-dist and boundary dist
                score = min(min_dist / 2.0, dist_to_boundary)
                
                if score > best_score:
                    best_score = score
                    best_pt = candidate
        
        if best_pt is None:
            # Fallback: random placement
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.1, 0.9)
            best_pt = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        
        pts[k] = best_pt
    
    return pts


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
    # Handle case where all values are -inf or similar
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
    # Project inside unit circle
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def _hexagonal_init(n):
    """
    Initialize with a hexagonal lattice structure.
    This is a strong starting point for circle packing.
    """
    # Generate hexagonal grid points
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


def _sequential_init_with_optimization(deadline):
    """
    Structural change: Use sequential farthest-point sampling as the primary
    initialization, then iteratively re-optimize the sequence by adjusting
    each point's position to improve the global minimum margin.
    
    This transforms the static geometry into a sequential construction problem
    where each placement decision is informed by all prior placements.
    """
    # Phase 1: Generate initial sequence via farthest-point sampling
    pts = _farthest_point_sample(None, N)
    current_score = _score(pts)
    
    # Phase 2: Iteratively improve by re-optimizing individual points
    # Each iteration, pick the point with the worst margin and try to move it
    max_rounds = 30
    for round_idx in range(max_rounds):
        if time.time() > deadline - 10:
            break
        
        # Compute margins for each point
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        # For each point, compute its "bottleneck" margin
        point_margins = np.zeros(N)
        for i in range(N):
            # Minimum pairwise distance involving point i
            min_pair_i = np.min(pair_margins[i])
            # Boundary margin for point i
            bound_i = boundary_margins[i]
            point_margins[i] = min(min_pair_i, bound_i)
        
        # Find the point with the worst margin
        worst_idx = np.argmin(point_margins)
        worst_margin = point_margins[worst_idx]
        
        # Try to improve this point using local optimization
        # Define a local objective: maximize the minimum margin when this point moves
        def local_obj(x_local):
            new_pts = pts.copy()
            new_pts[worst_idx] = x_local
            if not np.all(np.isfinite(new_pts)):
                return 1e10
            d_local = _pairwise_distances(new_pts)
            np.fill_diagonal(d_local, np.inf)
            min_pair = np.min(d_local) / 2.0
            min_bound = np.min(1.0 - np.sqrt(np.sum(new_pts ** 2, axis=1)))
            return -min(min_pair, min_bound)
        
        x0 = pts[worst_idx].copy()
        try:
            result = minimize(
                local_obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
            )
            new_pt = result.x
            if np.all(np.isfinite(new_pt)):
                # Check if this improvement helps globally
                test_pts = pts.copy()
                test_pts[worst_idx] = new_pt
                test_score = _score(test_pts)
                if test_score > current_score + 1e-12:
                    pts = test_pts
                    current_score = test_score
        except Exception:
            pass
    
    # Phase 3: Full smooth optimization from the sequentially-built init
    if time.time() < deadline - 15:
        opt_pts, opt_score = _optimize_from_init(pts, deadline)
        if opt_score > current_score:
            pts = opt_pts
            current_score = opt_score
    
    return pts, current_score


def _optimize_from_init(pts, deadline, max_beta_stages=5):
    """Multi-stage annealing optimization from initial points."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Progressive beta values for smooth optimization
    # Start with lower beta for exploration, higher for exploitation
    betas = [10.0, 30.0, 100.0, 300.0]
    
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
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
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
    # Only do this if we have enough time
    if time.time() < deadline - 10:
        for beta in [1000.0, 3000.0]:
            if time.time() > deadline - 5:
                break
            x0 = current_pts.ravel()
            try:
                result = minimize(
                    _smooth_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
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


def _binary_search_polish(pts, r_target, deadline, max_iters=500):
    """
    Polish by iteratively pushing points to achieve target radius r_target.
    This is a direct constraint satisfaction approach.
    """
    pts = pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Compute current margins
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        # Find the most violated constraint
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            # Pair constraint is binding
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
            # Boundary constraint is binding
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
        
        # Check convergence
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _de_ring_optimize(deadline):
    """
    Differential evolution over ring parameters to find good ring configurations.
    """
    ring_counts = [1, 6, 12, 18, 13]
    
    def ring_to_pts(params):
        radii = params[:4]
        offsets = params[4:]
        radii = np.maximum(radii, 0.05)
        radii = np.sort(radii)
        max_allowed = 1.0 - 0.12
        radii = np.minimum(radii, max_allowed)
        return _ring_initialization(radii, ring_counts, offsets)
    
    def neg_score(params):
        pts = ring_to_pts(params)
        return -_score(pts)
    
    bounds = [
        (0.10, 0.40),   # r1
        (0.25, 0.60),   # r2
        (0.40, 0.75),   # r3
        (0.55, 0.95),   # r4
        (0, 2*np.pi),   # offset1
        (0, 2*np.pi),   # offset2
        (0, 2*np.pi),   # offset3
        (0, 2*np.pi),   # offset4
    ]
    
    rng = np.random.RandomState(42)
    pop_size = 40
    max_gen = 60
    F = 0.7
    CR = 0.8
    
    pop = []
    for _ in range(pop_size):
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        pop.append(params)
    pop = np.array(pop)
    
    scores = np.array([neg_score(p) for p in pop])
    best_idx = np.argmin(scores)
    best_params = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(max_gen):
        if time.time() > deadline - 25:  # Leave more time for polish
            break
        
        for i in range(pop_size):
            # Select three distinct random individuals
            a_idx, b_idx, c_idx = rng.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = pop[a_idx] + F * (pop[b_idx] - pop[c_idx])
            
            # Clip to bounds
            for j in range(len(v)):
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(len(v))
            for j in range(len(v)):
                if rng.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Selection
            trial_score = neg_score(trial)
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_params = trial.copy()
    
    best_pts = ring_to_pts(best_params)
    
    # Polish the best ring configuration with local optimization
    # Use a shorter deadline for this polish to ensure we return in time
    polish_deadline = deadline - 5.0
    if time.time() < polish_deadline:
        opt_pts, opt_score = _optimize_from_init(best_pts, polish_deadline)
        if opt_score > -best_score:
            return opt_pts, opt_score
            
    return best_pts, -best_score


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 135s effective deadline
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Sequential farthest-point sampling + iterative optimization (NEW PRIMARY)
    # This is a structural change: the problem is formulated as a sequential
    # construction where each circle placement maximizes the minimum distance
    # to all previously placed circles and the boundary, then the entire sequence
    # is refined.
    if time.time() < deadline - 30:
        seq_pts, seq_score = _sequential_init_with_optimization(deadline)
        if seq_pts is not None and seq_score > best_score:
            best_score = seq_score
            best_points = seq_pts
    
    # Strategy 2: Ring-based DE + optimization (Secondary)
    if time.time() < deadline - 25:
        de_pts, de_score = _de_ring_optimize(deadline)
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_points = de_pts
    
    # Strategy 3: Hexagonal initialization + optimization (Tertiary)
    if time.time() < deadline - 15:
        hex_pts = _hexagonal_init(N)
        opt_pts, opt_score = _optimize_from_init(hex_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts

    # Strategy 4: If we have time, try to refine the best point found with a very tight local search
    if best_points is not None and time.time() < deadline - 8:
        # Use a very high beta polish on the current best
        x0 = best_points.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(5000.0,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 50000}
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
        # Fallback: return hexagonal init
        best_points = _hexagonal_init(N)
        best_score = _score(best_points)
    
    # Final binary search polish
    if time.time() < deadline - 2:
        # Try to improve by targeting a slightly higher radius
        target_r = best_score + 1e-5
        polished = _binary_search_polish(best_points, target_r, deadline, max_iters=100)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points
