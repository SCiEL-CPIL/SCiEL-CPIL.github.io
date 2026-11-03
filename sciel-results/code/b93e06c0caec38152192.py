"""Combined packing solver: ring-structured init + smooth surrogate annealing + local polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# Best known packing radius (Graham et al. 1998)
R_KNOWN = 0.1258254895304


def _pairwise_distances(pts):
    """Return matrix of pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score: min(min pairwise dist/2, min(1 - |c_i|))."""
    if len(pts) != N:
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return min(min_pair, min_boundary)


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp: approximates min(values) smoothly."""
    neg = -values * beta
    neg_max = np.max(neg)
    exp_sum = np.sum(np.exp(neg - neg_max))
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    """
    Smooth objective to maximize: we minimize the negative of the soft-min score.
    x is flat array of 2N coordinates.
    """
    pts = x.reshape(N, 2)
    # Pairwise distances
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    # Boundary distances
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    # Combine all constraints into one soft-min
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 0.0
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _energy(P, r):
    """
    Smooth energy function: sum of squared overlaps.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))


def _energy_grad(P, r):
    """
    Gradient of the energy function.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    V_cc = np.maximum(2.0 * r - dist, 0.0)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    W = V_cc * inv_dist
    
    sum_W = np.sum(W, axis=1) 
    term1 = (P.T @ W).T 
    term2 = P * sum_W[:, np.newaxis] 
    
    grad += term1 - term2
    
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _ring_initialization():
    """
    Initialize with a ring structure matching the known optimal packing:
    1 center + rings of 6, 12, 18, 13 (total 50).
    """
    centers = []
    # Center
    centers.append([0.0, 0.0])
    
    # Ring radii and counts
    r_est = 0.125
    ring1_r = 2.0 * r_est
    ring1_n = 6
    for i in range(ring1_n):
        angle = 2.0 * np.pi * i / ring1_n
        centers.append([ring1_r * np.cos(angle), ring1_r * np.sin(angle)])
    
    ring2_r = 2.0 * np.sqrt(3.0) * r_est
    ring2_n = 12
    for i in range(ring2_n):
        angle = 2.0 * np.pi * i / ring2_n
        centers.append([ring2_r * np.cos(angle), ring2_r * np.sin(angle)])
    
    ring3_r = 4.0 * r_est
    ring3_n = 18
    for i in range(ring3_n):
        angle = 2.0 * np.pi * i / ring3_n
        centers.append([ring3_r * np.cos(angle), ring3_r * np.sin(angle)])
    
    ring4_r = 5.5 * r_est
    ring4_n = 13
    for i in range(ring4_n):
        angle = 2.0 * np.pi * i / ring4_n + np.pi / ring4_n
        centers.append([ring4_r * np.cos(angle), ring4_r * np.sin(angle)])
    
    pts = np.array(centers[:N])
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 1.0 - r_est:
            pts[i] *= (1.0 - r_est) / norms[i]
    
    return pts


def _hex_lattice_initialization():
    """Hexagonal lattice as fallback initialization."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    best_pts = None
    best_score = -1.0
    
    while spacing > 0.05:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            order = np.argsort(radius[mask])
            pts = points[order[:N]]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        spacing *= 0.97
    
    if best_pts is None:
        best_pts = np.random.rand(N, 2) * 0.8 - 0.4
    return best_pts


def _gradient_polish(pts, r_target, time_budget=10.0):
    """
    Use gradient-based optimization with the overlap energy to polish positions.
    Returns improved points if found.
    """
    start_time = time.time()
    P_curr = pts.copy()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
    
    try:
        result = minimize(objective, P_curr.ravel(), jac=grad, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
        P_new = result.x.reshape(N, 2)
        
        # Validate
        norms = np.sqrt(np.sum(P_new ** 2, axis=1))
        if np.any(~np.isfinite(P_new)) or np.any(norms >= 1.0):
            return P_curr
        
        new_score = _score(P_new)
        if new_score > _score(P_curr):
            return P_new
    except Exception:
        pass
    
    return P_curr


def _manual_force_refine(pts, r_target, time_budget=5.0):
    """
    Manual force-directed refinement: push overlapping circles apart.
    """
    start_time = time.time()
    P_curr = pts.copy()
    
    max_iter = 200
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        forces = np.zeros_like(P_curr)
        
        # Circle-circle overlaps
        overlap_cc = np.maximum(2.0 * r_target - dist, 0.0)
        mask_cc = overlap_cc > 0
        if np.any(mask_cc):
            inv_dist = np.zeros_like(dist)
            np.divide(1.0, dist, out=inv_dist, where=dist > 0)
            weight = np.zeros((N, N))
            weight[mask_cc] = overlap_cc[mask_cc] * inv_dist[mask_cc]
            sum_w = np.sum(weight, axis=1)
            term1 = P_curr * sum_w[:, np.newaxis]
            term2 = (P_curr.T @ weight).T
            forces += 2.0 * (term1 - term2)
        
        # Circle-boundary overlaps
        dist_origin = np.linalg.norm(P_curr, axis=1)
        overlap_cb = np.maximum(dist_origin + r_target - 1.0, 0.0)
        mask_cb = overlap_cb > 0
        if np.any(mask_cb):
            unit_vec = P_curr[mask_cb] / dist_origin[mask_cb, np.newaxis]
            forces[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb, np.newaxis]
        
        norm_forces = np.linalg.norm(forces)
        if norm_forces < 1e-15:
            break
        
        # Adaptive step size
        best_step = 0
        best_score = _score(P_curr)
        for factor in [0.1, 0.05, 0.02, 0.01, 0.005]:
            step = factor * norm_forces / (N * N + 1)
            P_new = P_curr + step * forces
            new_score = _score(P_new)
            if new_score > best_score:
                best_score = new_score
                best_step = step
        
        if best_step > 0:
            P_curr = P_curr + best_step * forces
        else:
            break
    
    return P_curr


def construct():
    """Construct 50 equal circles inside a unit circle to maximize the common radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Try multiple initialization strategies and pick the best
    inits = []
    
    # Ring initialization
    ring_pts = _ring_initialization()
    inits.append(ring_pts)
    
    # Hex lattice initialization
    hex_pts = _hex_lattice_initialization()
    inits.append(hex_pts)
    
    # Random initialization
    rng = np.random.default_rng(42)
    rand_pts = rng.uniform(-0.8, 0.8, size=(N, 2))
    norms = np.sqrt(np.sum(rand_pts ** 2, axis=1))
    rand_pts = rand_pts * np.minimum(1.0, 0.8 / norms[:, np.newaxis])
    inits.append(rand_pts)
    
    # Evaluate initial scores
    best_init = max(inits, key=_score)
    current_pts = best_init.copy()
    current_score = _score(current_pts)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return current_pts
    
    # Phase 1: Smooth surrogate annealing with progressive beta
    betas = [5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
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
            
            # Validate
            norms = np.sqrt(np.sum(new_pts ** 2, axis=1))
            if np.any(~np.isfinite(new_pts)) or np.any(norms >= 1.0):
                continue
            
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return current_pts
    
    # Phase 2: Gradient-based polish using overlap energy
    # Try multiple target radii around current best
    for r_target in [current_score, current_score * 1.001, current_score * 1.002]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        polished = _gradient_polish(current_pts, r_target, time_budget=5.0)
        new_score = _score(polished)
        if new_score > current_score:
            current_pts = polished
            current_score = new_score
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 8:
        return current_pts
    
    # Phase 3: Manual force-directed refinement
    for r_target in [current_score, current_score * 1.001]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        refined = _manual_force_refine(current_pts, r_target, time_budget=3.0)
        new_score = _score(refined)
        if new_score > current_score:
            current_pts = refined
            current_score = new_score
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 3:
        return current_pts
    
    # Phase 4: Final high-precision polish with very high beta
    for beta in [5000.0, 10000.0]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 2:
            break
        
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 20000}
            )
            new_pts = result.x.reshape(N, 2)
            
            norms = np.sqrt(np.sum(new_pts ** 2, axis=1))
            if np.any(~np.isfinite(new_pts)) or np.any(norms >= 1.0):
                continue
            
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # Final validation: ensure all circles are within the unit circle
    dists = np.linalg.norm(current_pts, axis=1)
    for i in range(N):
        if dists[i] >= 1.0:
            current_pts[i] *= 0.99 / dists[i]
    
    return current_pts
