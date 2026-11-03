"""Packing 50 equal circles in unit disk - Enhanced multi-stage optimization.

Strategy:
1. Multi-start initialization: ring-based and hexagonal patterns.
2. Global search via Differential Evolution on ring parameters.
3. Local refinement via multi-stage L-BFGS-B annealing with soft-min objective.
4. Final high-precision polish via constraint projection and binary search.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DEADLINE_MARGIN = 18.0  # seconds before hard limit


def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score: largest common radius."""
    if pts is None or len(pts) != N:
        return -1.0
    pts = np.asarray(pts, dtype=float)
    if not np.all(np.isfinite(pts)):
        return -1.0
    
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0:
        return -1.0
    return val


def _softmin_logsumexp(values, beta):
    """Numerically-stable soft-min via log-sum-exp."""
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0.0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta):
    """Smooth objective for optimization: minimize negative soft-min of margins."""
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    # Combine all constraints
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


def _ring_to_points(radii, angles_list):
    """Convert ring parameters (radii, list of angle arrays) to point coordinates."""
    pts = [np.array([0.0, 0.0])]  # Center
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _hexagonal_init(scale=0.12):
    """Generate hexagonal lattice points and filter to fit in unit disk."""
    points = []
    # Generate hexagonal lattice
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x**2 + y**2 <= 0.81:  # Keep points well inside
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        return None
    
    # Sort by distance from center and take first 50
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def _anneal_points(pts, deadline, max_betas=12):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _score(current)
    
    # Progressive beta values for annealing
    betas = [3.0, 8.0, 20.0, 50.0, 120.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0]
    betas = betas[:max_betas]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={
                    'maxiter': 1000, 
                    'ftol': 1e-16, 
                    'gtol': 1e-14,
                    'maxfun': 30000
                }
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
            
    return current, current_score


def _constraint_push(pts, r_target, deadline, max_iters=500):
    """Polish solution by iteratively pushing violating constraints."""
    pts = pts.copy()
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        min_pair = np.min(pair_margins)
        min_bound = np.min(boundary_margins)
        
        if min_pair < min_bound - 1e-14:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0
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


def _binary_search_polish(pts, deadline):
    """Binary search for maximum feasible radius with projection polish."""
    current = pts.copy()
    low = _score(current) - 1e-8
    high = _score(current) + 1e-4
    
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=200)
        s = _score(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    
    return current, _score(current)


def _simulated_annealing_polish(pts, deadline, max_iters=2000):
    """Simulated annealing on full point coordinates for final polish."""
    current = pts.copy()
    current_score = _score(current)
    
    # Temperature schedule
    T0 = 0.01
    T_final = 1e-8
    alpha = (T_final / T0) ** (1.0 / max_iters)
    T = T0
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if time.time() > deadline - 4:
            break
        
        # Propose random perturbation
        proposal = current.copy()
        # Perturb 2-5 random points
        n_perturb = rng.randint(2, 6)
        idx = rng.choice(N, n_perturb, replace=False)
        noise_scale = T * 0.5
        proposal[idx] += rng.normal(0, noise_scale, size=(n_perturb, 2))
        
        # Ensure points stay in disk
        dists = np.sqrt(np.sum(proposal ** 2, axis=1))
        out_of_bounds = dists > 0.99
        if np.any(out_of_bounds):
            proposal[out_of_bounds] *= 0.99 / dists[out_of_bounds]
        
        new_score = _score(proposal)
        
        # Accept or reject
        if new_score >= current_score:
            current = proposal
            current_score = new_score
        else:
            delta = new_score - current_score
            if delta > 0:
                current = proposal
                current_score = new_score
            else:
                prob = np.exp(delta / (T + 1e-15))
                if rng.random() < prob:
                    current = proposal
                    current_score = new_score
        
        T *= alpha
    
    return current, current_score


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_MARGIN
    
    best_points = None
    best_score = -1.0
    
    # Standard ring topology for 50 circles: 1 + 6 + 12 + 18 + 13
    ring_sizes = [1, 6, 12, 18, 13]
    
    # Initial guess for radii
    init_radii = np.array([0.0, 0.25, 0.45, 0.65, 0.85])
    
    # Initialize angles
    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))
    
    # ---- Stage 1: Multiple initializations ----
    candidates = []
    
    # Ring-based initialization
    init_pts = _ring_to_points(init_radii, angles_list)
    init_score = _score(init_pts)
    if init_score > 0:
        candidates.append((init_pts.copy(), init_score))
    
    # Hexagonal initialization
    hex_pts = _hexagonal_init(scale=0.12)
    if hex_pts is not None and len(hex_pts) == N:
        hex_score = _score(hex_pts)
        if hex_score > 0:
            candidates.append((hex_pts.copy(), hex_score))
    
    # Try different hexagonal scales
    for scale in [0.11, 0.13]:
        hex_pts2 = _hexagonal_init(scale=scale)
        if hex_pts2 is not None and len(hex_pts2) == N:
            s2 = _score(hex_pts2)
            if s2 > 0:
                candidates.append((hex_pts2.copy(), s2))
    
    # Select best initial candidate
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_points = candidates[0][0]
        best_score = candidates[0][1]
    
    # ---- Stage 2: Differential Evolution on Ring Parameters ----
    if time.time() < deadline - 45:
        bounds = [(0.0, 0.95)] * len(ring_sizes)
        for i in range(1, len(ring_sizes)):
            bounds.extend([(0.0, 2.0 * np.pi)] * ring_sizes[i])
        
        def ring_obj(vec):
            if time.time() > deadline - 45:
                return 1e10
            radii = vec[:len(ring_sizes)].copy()
            angles_list_local = [np.array([0.0])]
            idx = len(ring_sizes)
            for i in range(1, len(ring_sizes)):
                n = ring_sizes[i]
                angles_list_local.append(vec[idx:idx + n].copy())
                idx += n
            
            radii = np.clip(radii, 0.0, 0.95)
            pts = _ring_to_points(radii, angles_list_local)
            d = _pairwise_distances(pts)
            np.fill_diagonal(d, np.inf)
            boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
            pair_half = d / 2.0
            all_vals = np.concatenate([pair_half.ravel(), boundary])
            all_vals = all_vals[np.isfinite(all_vals)]
            if len(all_vals) == 0:
                return 1e10
            return -_softmin_logsumexp(all_vals, 30.0)
        
        try:
            res_de = differential_evolution(
                ring_obj, 
                bounds, 
                maxiter=120, 
                popsize=25, 
                tol=1e-9,
                mutation=(0.4, 1.0), 
                recombination=0.8, 
                seed=42, 
                polish=False
            )
            if time.time() < deadline - 35:
                ring_x = res_de.x
                radii_opt = np.clip(ring_x[:len(ring_sizes)], 0.0, 0.95)
                angles_list_opt = [np.array([0.0])]
                idx = len(ring_sizes)
                for i in range(1, len(ring_sizes)):
                    n = ring_sizes[i]
                    angles_list_opt.append(ring_x[idx:idx + n].copy())
                    idx += n
                de_pts = _ring_to_points(radii_opt, angles_list_opt)
                de_score = _score(de_pts)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_pts.copy()
        except Exception:
            pass
    
    # ---- Stage 3: Multi-stage Annealing Refinement ----
    if best_points is not None and time.time() < deadline - 15:
        refined_pts, refined_score = _anneal_points(best_points, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # ---- Stage 3.5: Simulated Annealing Polish ----
    if best_points is not None and time.time() < deadline - 8:
        sa_pts, sa_score = _simulated_annealing_polish(best_points, deadline, max_iters=1500)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
    
    # ---- Stage 4: Binary Search Polish ----
    if best_points is not None and time.time() < deadline - 3:
        polished_pts, polished_score = _binary_search_polish(best_points, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # Fallback if something went wrong
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _ring_to_points(init_radii, angles_list)
        best_score = _score(best_points)
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
