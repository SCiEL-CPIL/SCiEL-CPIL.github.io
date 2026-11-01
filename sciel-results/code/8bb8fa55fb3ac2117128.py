"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_bound = np.min(np.minimum(dist_x, dist_y))
    
    return min(min_pair, min_bound)

def _soft_min_obj(x, n, beta):
    """Soft-min objective for optimization. x is flattened centers."""
    centers = x.reshape(n, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Combine all values
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    
    # Soft min using log-sum-exp for stability
    m = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - log_sum / beta
    
    return -soft_min_val

def _soft_min_grad(x, n, beta):
    """Gradient of the soft-min objective."""
    centers = x.reshape(n, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Combine all values
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    
    # Soft min weights
    m = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of soft_min w.r.t. all_vals is -weights
    # We need d(soft_min)/d(x)
    # soft_min = m - log_sum/beta where log_sum = log(sum(exp(-beta*(all_vals - m))))
    # d(soft_min)/d(all_vals_k) = -weights_k (approximately, since m depends on all_vals)
    # More precisely: d(soft_min)/d(v_k) = -exp(-beta*(v_k - m)) / sum(exp(-beta*(v_j - m)))
    # = -weights_k
    
    grad_all = -weights
    
    # Now compute gradient w.r.t. x
    grad_x = np.zeros_like(x)
    
    # Pairwise part
    num_pairs = n * (n - 1) // 2
    pair_grad = grad_all[:num_pairs]
    
    # For each pair (i, j) with i < j, contribution to gradient
    # pair_vals[k] = dist(i,j)/2
    # d(dist(i,j)/2)/d(centers[i]) = (centers[i] - centers[j]) / dist(i,j)
    # d(dist(i,j)/2)/d(centers[j]) = (centers[j] - centers[i]) / dist(i,j)
    
    for k in range(num_pairs):
        # Find which pair this is
        i = k // (n - 1)
        j_offset = k % (n - 1)
        if j_offset >= i:
            j = j_offset + 1
        else:
            j = j_offset
        
        d = dists[i, j]
        if d < 1e-12:
            continue
        
        direction = (centers[i] - centers[j]) / d
        w = pair_grad[k]
        
        grad_x[2*i] += w * direction[0]
        grad_x[2*i+1] += w * direction[1]
        grad_x[2*j] += w * (-direction[0])
        grad_x[2*j+1] += w * (-direction[1])
    
    # Boundary part
    bound_grad = grad_all[num_pairs:]
    for i in range(n):
        # dist_x = 0.5 - |centers[i,0]|
        # d(dist_x)/d(centers[i,0]) = -sign(centers[i,0])
        # dist_y = 0.5 - |centers[i,1]|
        # d(dist_y)/d(centers[i,1]) = -sign(centers[i,1])
        # bound_vals[i] = min(dist_x, dist_y)
        # The gradient is with respect to the min, so we need to handle the subgradient
        
        dx = 0.5 - np.abs(centers[i, 0])
        dy = 0.5 - np.abs(centers[i, 1])
        
        if dx <= dy:
            # Gradient comes from dist_x
            gx = -np.sign(centers[i, 0]) if centers[i, 0] != 0 else 1.0
            gy = 0.0
        else:
            gx = 0.0
            gy = -np.sign(centers[i, 1]) if centers[i, 1] != 0 else 1.0
        
        grad_x[2*i] += bound_grad[i] * gx
        grad_x[2*i+1] += bound_grad[i] * gy
    
    # We're minimizing -soft_min, so gradient is -d(soft_min)/dx
    return -grad_x

def _smart_init(n):
    """Create a smart initial configuration: 7x7 grid + optimal 50th point."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    if n == 49:
        return grid_49
    
    # Find best spot for the 50th point
    sample_res = 100
    xs = np.linspace(-0.5, 0.5, sample_res)
    ys = np.linspace(-0.5, 0.5, sample_res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Distance to existing grid points
    diff = pts[:, np.newaxis, :] - grid_49[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = np.min(dists, axis=1)
    
    # Distance to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    d_b = np.minimum(d_x, d_y)
    
    # Maximize the minimum clearance
    clearance = np.minimum(min_dists, d_b)
    idx = np.argmax(clearance)
    new_pt = pts[idx]
    
    centers = np.vstack([grid_49, new_pt])
    
    # Small perturbation to break symmetry
    np.random.seed(42)
    centers = centers + np.random.normal(0, 0.005, size=centers.shape)
    
    return centers[:n]

def _feasibility_check(centers, r):
    """Check if all circles of radius r fit without overlap or boundary violation."""
    if np.any(centers < -0.5 + r) or np.any(centers > 0.5 - r):
        return False, 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    if min_pair < r:
        return False, min_pair - r
    return True, 0.0

def _push_apart(centers, r, n_steps=500, lr=0.001):
    """Push overlapping circles apart using gradient descent on violations."""
    centers = centers.copy()
    
    for step in range(n_steps):
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 2*r and dist > 1e-12:
                    # Push apart
                    force = (2*r - dist) / dist * 0.5
                    forces[i] += force * d
                    forces[j] -= force * d
        
        # Boundary repulsion
        for i in range(N):
            if centers[i, 0] < -0.5 + r:
                forces[i, 0] += (-0.5 + r - centers[i, 0]) * 2.0
            elif centers[i, 0] > 0.5 - r:
                forces[i, 0] += (0.5 - r - centers[i, 0]) * 2.0
            if centers[i, 1] < -0.5 + r:
                forces[i, 1] += (-0.5 + r - centers[i, 1]) * 2.0
            elif centers[i, 1] > 0.5 - r:
                forces[i, 1] += (0.5 - r - centers[i, 1]) * 2.0
        
        centers += lr * forces
        centers = np.clip(centers, -0.5 + r, 0.5 - r)
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_centers = None
    best_r = 0.0
    
    # Initial configurations
    initial_configs = [
        _smart_init(N),
        _smart_init(N) + np.random.normal(0, 0.02, size=(N, 2)),
    ]
    
    for init_centers in initial_configs:
        if time.time() > deadline - 15.0:
            break
        
        centers = init_centers.copy()
        r = _compute_radius(centers)
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Binary search on radius with feasibility check
        r_low = r
        r_high = r * 1.5
        
        # First, try to increase r using push-apart
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            
            r_mid = (r_low + r_high) / 2
            if r_mid <= r_low + 1e-8:
                break
            
            # Try to find feasible configuration with radius r_mid
            trial_centers = _push_apart(centers, r_mid, n_steps=200, lr=0.002)
            feasible, violation = _feasibility_check(trial_centers, r_mid)
            
            if feasible:
                r_low = r_mid
                centers = trial_centers
                r = r_mid
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
            else:
                r_high = r_mid
        
        # Now do soft-min optimization to polish
        if time.time() > deadline - 5.0:
            continue
        
        current_centers = centers.copy()
        current_r = r
        
        betas = [50, 100, 200]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            
            x0 = current_centers.flatten()
            
            try:
                result = minimize(
                    _soft_min_obj,
                    x0,
                    args=(N, beta),
                    method='L-BFGS-B',
                    bounds=[(-0.49, 0.49)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                refined_centers = result.x.reshape(N, 2)
                refined_r = _compute_radius(refined_centers)
                
                if refined_r > current_r:
                    current_r = refined_r
                    current_centers = refined_centers
            except Exception:
                pass
        
        if current_r > best_r:
            best_r = current_r
            best_centers = current_centers.copy()
    
    # Final polish with high beta
    if time.time() < deadline - 2.0 and best_centers is not None:
        try:
            x0 = best_centers.flatten()
            result = minimize(
                _soft_min_obj,
                x0,
                args=(N, 500),
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (2 * N),
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            refined_centers = result.x.reshape(N, 2)
            refined_r = _compute_radius(refined_centers)
            if refined_r > best_r:
                best_r = refined_r
                best_centers = refined_centers
        except Exception:
            pass

    # Ensure validity
    if best_centers is not None:
        eps = 1e-9
        best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    else:
        best_centers = _smart_init(N)
        best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

# EVOLVE-BLOCK-END
