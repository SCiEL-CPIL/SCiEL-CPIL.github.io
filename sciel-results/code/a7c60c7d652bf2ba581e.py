"""Improved packing: ring-structured init + smooth surrogate + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# Best known packing radius (Graham et al. 1998) as a reference upper bound
R_KNOWN = 0.1258254895304

# Global best found across all attempts
_best_result = None


def _pairwise_distances(pts):
    """Return matrix of pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def _pairwise_distances_sq(pts):
    """Return matrix of squared pairwise distances (cheaper for objective)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=-1)


def _score(pts):
    """Compute the packing score: min(min pairwise dist/2, min(1 - |c_i|))."""
    if len(pts) != N:
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return min(min_pair, min_boundary)

def _score_fast(pts):
    """Faster score computation using squared distances where possible."""
    if len(pts) != N:
        return 0.0
    d_sq = _pairwise_distances_sq(pts)
    np.fill_diagonal(d_sq, np.inf)
    min_pair = np.sqrt(np.min(d_sq)) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return min(min_pair, min_boundary)


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp: approximates min(values) smoothly."""
    # Use the identity: softmin(x) ≈ -logsumexp(-x * beta) / beta
    neg = -values * beta
    neg_max = np.max(neg)
    # For numerical stability
    exp_sum = np.sum(np.exp(neg - neg_max))
    return -(neg_max + np.log(exp_sum)) / beta

def _softmin_logsumexp_sq(values, beta):
    """Soft-min on squared values for efficiency, returns soft-min of sqrt values."""
    # We want softmin(sqrt(values)), but work with values directly
    # softmin(sqrt(v)) ≈ -1/beta * logsumexp(-sqrt(v)*beta)
    # For efficiency, use the fact that if v_i are close, sqrt is smooth
    # Simpler: just compute sqrt first, then softmin
    vals = np.sqrt(values)
    return _softmin_logsumexp(vals, beta)


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
    # We want the minimum of all pairwise/2 and boundary values
    pair_half = d / 2.0
    # Flatten and concatenate
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    # Remove inf values
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 0.0
    soft_min = _softmin_logsumexp(all_vals, beta)
    # We minimize -soft_min (to maximize the soft-min)
    return -soft_min

def _smooth_objective_grad(x, beta=50.0):
    """
    Gradient of the smooth objective for faster L-BFGS-B convergence.
    """
    pts = x.reshape(N, 2)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary = 1.0 - norms
    
    # Combine all constraints
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft-min: -1/beta * logsumexp(-beta * values)
    # Derivative w.r.t. values: -1/beta * d/dv [logsumexp(-beta*v)]
    # = -1/beta * [-beta * exp(-beta*v) / sum(exp(-beta*v))]
    # = exp(-beta*v) / sum(exp(-beta*v))
    
    neg_beta_vals = -beta * all_vals
    neg_max = np.max(neg_beta_vals)
    exp_vals = np.exp(neg_beta_vals - neg_max)
    exp_sum = np.sum(exp_vals)
    weights = exp_vals / exp_sum  # These are the soft-min weights
    
    # Now we need to map weights back to the original structure
    n_pairs = N * (N - 1) // 2
    n_boundary = N
    total = n_pairs + n_boundary
    
    # Reconstruct the full weight array
    full_weights = np.zeros(total)
    full_weights[:n_pairs] = weights[:n_pairs]
    full_weights[n_pairs:] = weights[n_pairs:]
    
    # Gradient of -softmin w.r.t. each value is -weight_i
    # So gradient w.r.t. x is sum over i of (-weight_i) * d(value_i)/dx
    
    grad = np.zeros_like(x)
    
    # Pairwise contributions
    # value_i = d_ij / 2, so d(value_i)/dx = d(d_ij)/dx / 2
    # d(d_ij)/dx_k = (x_k - x_j) / d_ij for i != k, j
    # This is complex, so we use finite differences as fallback or skip gradient
    
    # For simplicity and robustness, return None for gradient to let L-BFGS-B use numerical
    return -_softmin_logsumexp(all_vals, beta), None


def _ring_initialization():
    """
    Initialize with a ring structure matching the known optimal packing:
    1 center + rings of 6, 12, 18, 13 (total 50).
    """
    centers = []
    # Center
    centers.append([0.0, 0.0])
    
    # Ring radii and counts
    # Approximate ring radii for r ≈ 0.1258
    # Ring 1: 6 circles at radius ~ 2*r = 0.2516
    # Ring 2: 12 circles at radius ~ 2*sqrt(3)*r ≈ 0.4358
    # Ring 3: 18 circles at radius ~ 4*r = 0.5033 (approx)
    # Ring 4: 13 circles at radius ~ 5*r ≈ 0.6291 (approx)
    
    # Let's use more careful estimates
    # For 6 circles around center: radius = 2r
    r_est = 0.125
    ring1_r = 2.0 * r_est  # ~0.25
    ring1_n = 6
    for i in range(ring1_n):
        angle = 2.0 * np.pi * i / ring1_n
        centers.append([ring1_r * np.cos(angle), ring1_r * np.sin(angle)])
    
    # Ring 2: 12 circles
    ring2_r = 2.0 * np.sqrt(3.0) * r_est  # ~0.433
    ring2_n = 12
    for i in range(ring2_n):
        angle = 2.0 * np.pi * i / ring2_n
        centers.append([ring2_r * np.cos(angle), ring2_r * np.sin(angle)])
    
    # Ring 3: 18 circles
    ring3_r = 4.0 * r_est  # ~0.50
    ring3_n = 18
    for i in range(ring3_n):
        angle = 2.0 * np.pi * i / ring3_n
        centers.append([ring3_r * np.cos(angle), ring3_r * np.sin(angle)])
    
    # Ring 4: 13 circles
    ring4_r = 5.5 * r_est  # ~0.6875, need to fit within 1 - r
    ring4_n = 13
    for i in range(ring4_n):
        angle = 2.0 * np.pi * i / ring4_n + np.pi / ring4_n  # offset
        centers.append([ring4_r * np.cos(angle), ring4_r * np.sin(angle)])
    
    pts = np.array(centers[:N])
    # Ensure all points are within the unit circle (with margin)
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 1.0 - r_est:
            pts[i] *= (1.0 - r_est) / norms[i]
    
    return pts

def _ring_initialization_v2():
    """
    Alternative ring initialization with slightly different parameters.
    Uses r_est closer to known optimum.
    """
    centers = []
    centers.append([0.0, 0.0])
    
    r_est = 0.1258
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
            # Sort by distance from origin, take closest N
            order = np.argsort(radius[mask])
            pts = points[order[:N]]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        spacing *= 0.97
    
    if best_pts is None:
        # Fallback: uniform random
        best_pts = np.random.rand(N, 2) * 0.8 - 0.4
    return best_pts

def _random_initialization(seed=42):
    """Random initialization with projection into unit disk."""
    rng = np.random.default_rng(seed)
    rand_pts = rng.uniform(-1.0, 1.0, size=(N, 2))
    norms = np.sqrt(np.sum(rand_pts ** 2, axis=1))
    # Project points that are outside into the disk
    for i in range(N):
        if norms[i] > 0.85:
            rand_pts[i] *= 0.85 / norms[i]
    return rand_pts


def construct():
    global _best_result
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Try multiple initialization strategies and pick the best
    inits = []
    
    # Ring initialization
    ring_pts = _ring_initialization()
    inits.append(ring_pts)
    
    # Ring initialization v2
    ring_pts_v2 = _ring_initialization_v2()
    inits.append(ring_pts_v2)
    
    # Hex lattice initialization
    hex_pts = _hex_lattice_initialization()
    inits.append(hex_pts)
    
    # Random initialization with different seeds
    for seed in [42, 123, 456]:
        rand_pts = _random_initialization(seed)
        inits.append(rand_pts)
    
    # Evaluate initial scores
    best_init = max(inits, key=_score)
    best_score = _score(best_init)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_init
    
    # Optimize using L-BFGS-B with smooth surrogate
    # Use multiple beta values for annealing effect
    betas = [10.0, 25.0, 50.0, 100.0, 200.0]
    
    current_pts = best_init.copy()
    current_score = best_score
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        x0 = current_pts.ravel()
        
        # Run optimization
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
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
    if elapsed > deadline - 5:
        return current_pts
    
    # Final polish: high-precision local optimization with very high beta
    # This refines the solution with tighter tolerances
    for beta in [500.0, 1000.0, 2000.0]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 3:
            break
        
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 50000}
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
    
    # Final validation
    final_pts = current_pts
    norms = np.sqrt(np.sum(final_pts ** 2, axis=1))
    if np.any(~np.isfinite(final_pts)) or np.any(norms >= 1.0):
        # Fallback to best valid init
        return best_init
    
    return final_pts
