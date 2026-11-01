import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: Minimize the log-sum-exp soft minimum of clearances.
    This is equivalent to maximizing the approximate minimum clearance.
    Uses Möbius-transformed coordinates for a different geometric metric.
    """
    centers = params.reshape(n, 2)
    
    # Calculate all clearances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    pair_clearances = pair_dists / 2.0
    
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_clearances = 1.0 - norms
    
    all_clearances = np.concatenate([pair_clearances, bound_clearances])
    
    # Numerical stability: shift by min clearance
    c_min = np.min(all_clearances)
    shifted = -beta * (all_clearances - c_min)
    
    exp_terms = np.exp(shifted)
    S = np.sum(exp_terms)
    
    if S == 0 or not np.isfinite(S):
        return 0.0, np.zeros_like(params)
        
    # Objective value: -c_min + (1/beta)*ln(S)
    obj_val = -c_min + (1.0 / beta) * np.log(S)
    
    # Weights w_k = exp(-beta * c_k) / S
    weights = exp_terms / S
    
    # Gradient calculation
    grad = np.zeros_like(centers)
    
    # Pairwise contributions
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        w = weights[k]
        if w < 1e-10:
            continue
        d = pair_dists[k]
        if d < 1e-10:
            continue
        vec = (centers[i] - centers[j]) / d
        grad[i] -= w * 0.5 * vec
        grad[j] += w * 0.5 * vec
        
    # Boundary contributions
    for i in range(n):
        w = weights[len(i_idx) + i]
        if w < 1e-10:
            continue
        norm = norms[i]
        if norm < 1e-10:
            continue
        vec = centers[i] / norm
        grad[i] += w * vec
        
    return obj_val, grad

def _vectorized_force_relax(centers, n_steps=200, dt=0.002):
    """
    Vectorized force-directed relaxation using numpy operations.
    """
    centers = centers.copy()
    n = len(centers)
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Vectorized pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Repulsion strength: strong when close
        repulsion = 1.0 / (dists ** 2)
        # Boost repulsion for very close pairs
        close_mask = dists < 0.3
        repulsion[close_mask] *= 10.0
        np.fill_diagonal(repulsion, 0.0)
        
        # Force vector: sum over all other particles
        force_vec = diff * (repulsion / dists)[:, :, np.newaxis]
        forces = np.sum(force_vec, axis=1)
        
        # Boundary force: push back if outside 0.95
        norms = np.linalg.norm(centers, axis=1)
        boundary_mask = norms > 0.95
        if np.any(boundary_mask):
            boundary_forces = -5.0 * centers[boundary_mask] / norms[boundary_mask, np.newaxis]
            forces[boundary_mask] += boundary_forces
        
        # Update positions
        centers += dt * forces
        
        # Project to disk if needed
        norms = np.linalg.norm(centers, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            centers[mask] *= 0.99 / norms[mask]
    
    return centers

def _get_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
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
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, seed=0):
    """Generate initial points on concentric rings matching known optimal structure."""
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 circles at r=0.24
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.24
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 2: 12 circles at r=0.48
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.48
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 3: 18 circles at r=0.72
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.72
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 4: 14 circles at r=0.92 (filling remaining)
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1)
        r = 0.92
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    return np.array(points[:n])

def _get_random_initialization(n, seed=0):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def _optimize_from_start(centers, start_time, deadline, max_betas=4):
    """
    Optimize from a starting configuration using soft-min with increasing beta.
    Returns best centers found.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    betas = [200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0][:max_betas]
    
    for beta in betas:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, beta),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-10, 'maxfun': 5000}
            )
            new_centers = res.x.reshape(N, 2)
            
            # Ensure centers are inside the disk
            norms = np.linalg.norm(new_centers, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                new_centers[mask] *= 0.999 / norms[mask]
            
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    return best_centers, best_score

def _binary_search_radius(centers, start_time, deadline):
    """
    Binary search on radius to find the maximum feasible packing radius.
    For each candidate radius r, check if we can place N circles of radius r.
    """
    best_centers = centers.copy()
    best_r = _compute_score(best_centers)
    
    # Initial bounds
    r_low = best_r
    r_high = 0.15  # Upper bound estimate
    
    # Binary search iterations
    max_iters = 15
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to optimize with target radius r_mid
        # Use soft-min with high beta to push toward uniform spacing
        beta = 5000.0
        
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, beta),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-10, 'maxfun': 10000}
            )
            new_centers = res.x.reshape(N, 2)
            
            # Ensure centers are inside the disk
            norms = np.linalg.norm(new_centers, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                new_centers[mask] *= 0.999 / norms[mask]
            
            new_score = _compute_score(new_centers)
            
            if new_score >= r_mid - 1e-6:
                # Feasible: can do better
                r_low = r_mid
                if new_score > best_r:
                    best_r = new_score
                    best_centers = new_centers.copy()
            else:
                # Not feasible: need smaller radius
                r_high = r_mid
        except Exception:
            r_high = r_mid
    
    return best_centers, best_r

def _final_polish(centers, start_time, deadline):
    """
    Final high-precision polish with multiple optimization passes.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Pass 1: Soft-min with high beta
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, 10000.0),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-12, 'maxfun': 20000}
            )
            new_centers = res.x.reshape(N, 2)
            norms = np.linalg.norm(new_centers, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                new_centers[mask] *= 0.999 / norms[mask]
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Pass 2: Another round with slightly different beta
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, 5000.0),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-12, 'maxfun': 10000}
            )
            new_centers = res.x.reshape(N, 2)
            norms = np.linalg.norm(new_centers, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                new_centers[mask] *= 0.999 / norms[mask]
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Pass 3: Force-directed relaxation for fine-tuning
    if time.time() < deadline - 1:
        try:
            relaxed = _vectorized_force_relax(best_centers, n_steps=100, dt=0.0005)
            relaxed_score = _compute_score(relaxed)
            if relaxed_score > best_score:
                best_score = relaxed_score
                best_centers = relaxed.copy()
        except Exception:
            pass
    
    return best_centers, best_score

def _construct():
    """Main construction function."""
    start_time = time.time()
    
    # Generate diverse initializations
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N),  # Different seed via internal randomness
        _get_random_initialization(N, seed=999),
    ]
    
    # Relax all initializations
    relaxed_inits = []
    for init in initializations:
        if time.time() > DEADLINE - 40:
            break
        relaxed = _vectorized_force_relax(init, n_steps=150, dt=0.002)
        relaxed_inits.append(relaxed)
    
    if not relaxed_inits:
        return initializations[0]
    
    # Optimize each initialization
    best_centers = relaxed_inits[0].copy()
    best_score = _compute_score(best_centers)
    
    # Allocate time: 50% for multi-start optimization, 30% for binary search, 20% for polish
    opt_deadline = start_time + 0.50 * (DEADLINE - start_time)
    bs_deadline = start_time + 0.80 * (DEADLINE - start_time)
    
    for i, init in enumerate(relaxed_inits):
        if time.time() > opt_deadline:
            break
        
        remaining_inits = len(relaxed_inits) - i
        time_per_init = (opt_deadline - time.time()) / max(remaining_inits, 1)
        init_deadline = time.time() + min(time_per_init, 20)
        
        centers, score = _optimize_from_start(init, start_time, init_deadline, max_betas=3)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Binary search on radius
    if time.time() < bs_deadline:
        best_centers, best_score = _binary_search_radius(best_centers, start_time, bs_deadline)
    
    # Final polish
    if time.time() < DEADLINE - 2:
        best_centers, best_score = _final_polish(best_centers, start_time, DEADLINE)

    return best_centers

def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
