"""
Packing 50 equal circles in a unit circle.
Combines soft-min L-BFGS optimization with force-directed pre-relaxation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
# Evaluator kills at 150s. We aim to finish by 140s.
_DEADLINE = _START_TIME + 140.0

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Compute the maximum radius r for which circles centered at 'centers' 
    are disjoint and inside the unit circle.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    """
    if centers.size == 0:
        return 0.0
    n = centers.shape[0]
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound_dist = np.min(1.0 - norms)
    
    # Check validity: centers must be inside
    if np.any(norms > 1.0):
        return 0.0
        
    return min(min_pair_dist / 2.0, min_bound_dist)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: -soft_min(constraints)
    Constraints: d_ij / 2 for pairs, 1 - |c_i| for boundary.
    We want to maximize the minimum of these, so we minimize the negative soft-min.
    """
    centers = params.reshape(n, 2)
    
    # 1. Compute pair distances (vectorized)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # 2. Compute boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    
    # 3. Combine all constraints into a single vector
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # 4. Compute Soft Min
    m = np.min(vals)
    shifted = vals - m
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    
    obj_val = -(m - (1.0 / beta) * np.log(S))
    
    # 5. Compute Gradient (vectorized)
    weights = exp_terms / S
    
    grad_centers = np.zeros_like(centers)
    
    # Pair contributions (vectorized)
    pair_weights = weights[:n*(n-1)//2]
    # Only include pairs with significant weight to save time and improve stability
    mask = pair_weights > 1e-12
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        w_m = pair_weights[mask]
        d_m = dists[i_m, j_m]
        d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
        vecs = (centers[i_m] - centers[j_m]) / d_safe[:, np.newaxis]
        contrib = w_m[:, np.newaxis] * 0.5 * vecs
        # Scatter add for i indices (repulsion from j pushes i away, so gradient of dist is (ci-cj)/d)
        # We are minimizing -soft_min, so we maximize soft_min.
        # d(soft_min)/d(ci) ~ w_i * (ci-cj)/d
        # d(-soft_min)/d(ci) ~ -w_i * (ci-cj)/d
        # Wait, the objective is -soft_min.
        # soft_min = m - 1/beta log(sum exp(-beta (v-m)))
        # d(soft_min)/d(v_k) = - exp(-beta(v_k-m)) / sum(exp(...)) = -w_k
        # d(-soft_min)/d(v_k) = w_k
        # v_k for pair (i,j) is d_ij/2.
        # d(v_k)/d(c_i) = 0.5 * (c_i - c_j) / d_ij
        # So d(-soft_min)/d(c_i) = w_k * 0.5 * (c_i - c_j) / d_ij
        # This means the gradient points AWAY from the other circle?
        # If c_i is close to c_j, d is small. v is small. w is large.
        # Gradient is proportional to (c_i - c_j).
        # Minimizing the objective (which is -max_min) means we want to increase the min.
        # The gradient of the objective function tells us the direction of steepest ascent of the objective.
        # We want to MINIMIZE the objective. So we move AGAINST the gradient.
        # If gradient is (c_i - c_j), we move c_i - (c_i - c_j) = c_j? No, that would collapse them.
        # Let's re-verify signs.
        # f = -soft_min(vals)
        # df/dc_i = - d(soft_min)/d(vals_k) * d(vals_k)/d(c_i)
        # d(soft_min)/d(vals_k) = -w_k (where w_k = exp(-beta(vals_k-m))/S)
        # So df/dc_i = - (-w_k) * d(vals_k)/d(c_i) = w_k * d(vals_k)/d(c_i)
        # d(vals_k)/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # So df/dc_i = w_k * 0.5 * unit_vec(i->j)
        # This vector points FROM j TO i.
        # If we minimize f, we update c_i = c_i - lr * df/dc_i.
        # c_i moves AGAINST the vector pointing from j to i.
        # So c_i moves TOWARDS j?
        # This is wrong. We want to maximize the distance.
        # Let's check the definition of soft_min again.
        # soft_min(x) approximates min(x).
        # If x_k is the minimum, soft_min is close to x_k.
        # If we increase x_k, soft_min increases.
        # So d(soft_min)/d(x_k) should be positive.
        # Formula: soft_min = m - 1/beta log(sum exp(-beta (x-m)))
        # Let y = x-m. sum exp(-beta y).
        # d/dx_k [ -1/beta log(sum exp(-beta (x-m))) ]
        # = -1/beta * 1/sum * d/dx_k (sum exp(-beta (x-m)))
        # = -1/beta * 1/sum * (-beta) exp(-beta (x_k-m))
        # = exp(-beta (x_k-m)) / sum = w_k.
        # So d(soft_min)/d(x_k) = +w_k.
        # Therefore d(-soft_min)/d(x_k) = -w_k.
        # So df/dc_i = -w_k * d(vals_k)/d(c_i) = -w_k * 0.5 * (c_i - c_j) / d
        # This vector points FROM i TO j (negative of i->j).
        # Minimizing f: c_i -= lr * (-w_k * 0.5 * unit(i->j))
        # c_i += lr * w_k * 0.5 * unit(i->j)
        # c_i moves AWAY from j. Correct.
        
        # So the gradient returned should be df/dc.
        # df/dc_i = -w_k * 0.5 * (c_i - c_j) / d
        # df/dc_j = -w_k * 0.5 * (c_j - c_i) / d = +w_k * 0.5 * (c_i - c_j) / d
        
        # In the code above:
        # contrib = w_m[:, np.newaxis] * 0.5 * vecs  where vecs = (ci - cj)/d
        # This is w * 0.5 * unit(i->j).
        # We need to add -contrib to grad[i] and +contrib to grad[j]?
        # Let's check boundary.
        # v_b = 1 - ||c||. d(v_b)/d(c) = -c/||c||.
        # d(-soft_min)/d(v_b) = -w_b.
        # df/dc = -w_b * (-c/||c||) = w_b * c/||c||.
        # Vector points OUTWARD from center.
        # Minimizing f: c -= lr * (w_b * c/||c||).
        # c moves INWARD. Correct.
        
        # So for pairs:
        # grad[i] += -w * 0.5 * unit(i->j)
        # grad[j] += +w * 0.5 * unit(i->j)
        
        np.add.at(grad_centers, i_m, -contrib)
        np.add.at(grad_centers, j_m, contrib)

    # Boundary contributions (vectorized)
    bound_weights = weights[n*(n-1)//2:]
    mask_b = bound_weights > 1e-12
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        w_b = bound_weights[idx_b]
        norm_b = norms[idx_b]
        norm_safe = np.where(norm_b > 1e-10, norm_b, 1e-10)
        vecs_b = centers[idx_b] / norm_safe[:, np.newaxis]
        contrib_b = w_b[:, np.newaxis] * vecs_b
        # grad += w_b * c/||c||
        np.add.at(grad_centers, idx_b, contrib_b)
            
    return obj_val, grad_centers.flatten()

def _get_ring_initialization(n):
    """
    Generate a ring-based initialization tailored for N=50.
    Center: 1
    Ring 1: 6
    Ring 2: 12
    Ring 3: 12
    Ring 4: 9
    Ring 5: 10
    Total: 50
    """
    centers = []
    centers.append([0.0, 0.0])
    
    # Radii estimated based on packing density
    rings = [
        (6, 0.22),
        (12, 0.44),
        (12, 0.66),
        (9, 0.82),
        (10, 0.92)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
            
    return np.array(centers[:n])

def _get_hex_lattice_initialization(n):
    """
    Generate a hexagonal lattice initialization.
    """
    r_est = 0.12
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 3
    
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        return _get_ring_initialization(n)
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """Generate random initialization with rejection sampling for minimum distance."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(10000):
        if len(centers) >= n:
            break
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y > 0.95**2:
            continue
        if centers:
            pts = np.array(centers)
            dists = np.hypot(pts[:,0] - x, pts[:,1] - y)
            if np.min(dists) < 0.15:
                continue
        centers.append([x, y])
    
    while len(centers) < n:
        x = rng.uniform(-0.9, 0.9)
        y = rng.uniform(-0.9, 0.9)
        centers.append([x, y])
        
    return np.array(centers[:n])

def _force_directed_relax(centers, time_budget=3.0):
    """
    Pre-processing step: use force-directed relaxation to resolve overlaps
    and push points to a valid configuration before L-BFGS.
    """
    centers = centers.copy()
    start_time = time.time()
    step = 0.01
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    n = len(centers)
    
    while time.time() - start_time < time_budget:
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion (vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Force magnitude: 1/(d^2 + eps), direction: diff/d
        # We want to push apart if d is small.
        force_mag = 1.0 / (dists * dists + 1e-6)
        np.fill_diagonal(force_mag, 0.0)
        
        # Scale diff by force_mag/d to get force vector
        # force_vec = diff * force_mag[:, :, np.newaxis] / np.maximum(dists, 1e-10)[:, :, np.newaxis]
        # This results in a vector pointing away from the other point.
        force_vec = diff * (force_mag / np.maximum(dists, 1e-10))[:, :, np.newaxis]
        forces = np.sum(force_vec, axis=1)
        
        # Boundary constraint: push inward if close to boundary
        radii = np.sqrt(np.sum(centers * centers, axis=1))
        boundary_mask = radii > 0.90
        if np.any(boundary_mask):
            r_safe = np.maximum(radii, 1e-10)
            # Force is proportional to (1 - r) or just push inward
            # Simple push inward: -center / r
            forces[boundary_mask] -= 0.5 * centers[boundary_mask] / r_safe[boundary_mask][:, np.newaxis]
        
        new_centers = centers + step * forces
        new_radii = np.sqrt(np.sum(new_centers * new_centers, axis=1))
        mask = new_radii > 1.0
        if np.any(mask):
            new_centers[mask] = new_centers[mask] / new_radii[mask][:, None] * 0.999
        
        new_score = _compute_score(new_centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers
            step *= 1.05
        else:
            step *= 0.9
            centers = best_centers.copy()
            
        if step < 1e-8:
            break
            
    return best_centers

def _optimize_lbfgs(centers, beta, maxiter, gtol):
    """
    Optimize using L-BFGS-B.
    """
    try:
        res = minimize(
            _soft_min_obj_and_grad, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _polish(centers, start_time, deadline):
    """
    Progressively polish the solution with increasing beta.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    
    # Start with lower beta for global landscape, increase for local precision
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        maxiter = 200
        gtol = 1e-8
        
        polished = _optimize_lbfgs(current, beta, maxiter, gtol)
        
        if np.all(np.isfinite(polished)):
            # Ensure points are inside
            radii = np.hypot(polished[:, 0], polished[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                polished[mask] *= (0.999 / radii[mask])[:, None]
            
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score

def construct():
    """
    Construct positions for N circles in the unit disk.
    Returns an array of shape (N, 2).
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    # 1. Generate diverse initializations
    inits = []
    inits.append(_get_ring_initialization(N))
    inits.append(_get_hex_lattice_initialization(N))
    
    for seed in [1, 2, 3, 4, 5, 6, 7]:
        inits.append(_get_random_initialization(N, seed))
    
    best_pts = None
    best_score = 0.0
    
    # 2. Pre-relax and Optimize each initialization
    # We allocate time based on number of inits
    # 8 inits. ~15s each for relax + initial polish?
    # Total budget ~140s.
    # Let's do 5s relax + 10s polish per init = 15s * 8 = 120s.
    # Leaves 20s for final deep polish.
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Pre-relaxation to escape bad local minima / resolve overlaps
        relaxed = _force_directed_relax(init, time_budget=3.0)
        
        # Quick polish
        optimized, score = _polish(relaxed, start_time, deadline)
        
        if score > best_score:
            best_score = score
            best_pts = optimized.copy()
            
    if best_pts is None:
        best_pts = inits[0]
        best_score = _compute_score(best_pts)
    
    # 3. Deep optimization on the best candidate
    if time.time() < deadline - 10:
        current = best_pts.copy()
        
        # Higher betas for final precision
        betas = [500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            maxiter = 500
            gtol = 1e-10
            
            optimized = _optimize_lbfgs(current, beta, maxiter, gtol)
            
            if np.all(np.isfinite(optimized)):
                radii = np.hypot(optimized[:, 0], optimized[:, 1])
                mask = radii > 1.0
                if np.any(mask):
                    optimized[mask] *= (0.9999 / radii[mask])[:, None]
                
                score = _compute_score(optimized)
                if score > best_score:
                    best_score = score
                    best_pts = optimized
                    current = optimized
    
    # Final validation and cleanup
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    max_allowed_radius = 1.0 - best_score - 1e-12
    
    mask = radii > max_allowed_radius
    if np.any(mask):
        best_pts[mask] *= (max_allowed_radius / radii[mask])[:, None]
        
    return best_pts

# EVOLVE-BLOCK-END
