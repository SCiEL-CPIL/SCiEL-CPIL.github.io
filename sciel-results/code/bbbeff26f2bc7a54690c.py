"""Improved packing: FCC init + L-BFGS-B with soft-min surrogate + final polish."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_radius(c):
    """Compute the true radius for a set of centres."""
    # Distance to walls
    wall_dist = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall_dist)
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(c, metric='euclidean')
    min_pair = np.min(d) if len(d) > 0 else np.inf
    return min(min_wall, min_pair / 2.0)


def _soft_min_radius(c, beta):
    """
    Smooth surrogate for the true radius using log-sum-exp.
    We maximize the soft-min of (pairwise_dist/2, wall_dists).
    
    The true radius r = min(min_pair/2, min_wall).
    We approximate -log-sum-exp(-beta * values) / beta as a smooth min.
    """
    c = np.asarray(c).reshape(-1, 3)
    n = len(c)
    
    # Wall distances: 0.5 - |c_ik| for each coordinate
    # min_wall = min over all i,k of (0.5 - |c_ik|)
    # = 0.5 - max_{i,k} |c_ik|
    # So values for walls: 0.5 - |c_ik| for all i,k
    
    # Pairwise distances / 2
    # We need min over i<j of |c_i - c_j|/2
    
    # Build all "constraint values" whose min gives the radius
    # Wall constraints: for each i, k: v_ik = 0.5 - |c_ik|
    # Pair constraints: for each i<j: v_ij = |c_i - c_j| / 2
    
    # For efficiency, compute soft-min directly
    
    # Wall part: soft_min of (0.5 - |c_ik|) over all i,k
    # = 0.5 - soft_max(|c_ik|) over all i,k
    # soft_max(x) = (1/beta) * log(sum(exp(beta * x)))
    # So soft_min_wall = 0.5 - (1/beta)*log(sum_{i,k} exp(beta*(0.5 - |c_ik|)))
    #                = 0.5 - (1/beta)*log(exp(beta*0.5) * sum_{i,k} exp(-beta*|c_ik|))
    #                = 0.5 - 0.5 - (1/beta)*log(sum_{i,k} exp(-beta*|c_ik|))
    #                = -(1/beta)*log(sum_{i,k} exp(-beta*|c_ik|))
    
    abs_c = np.abs(c)  # (n, 3)
    # sum_{i,k} exp(-beta * |c_ik|)
    wall_sum = np.sum(np.exp(-beta * abs_c))
    soft_min_wall = -(1.0 / beta) * np.log(wall_sum)
    
    # Pair part: soft_min of (|c_i - c_j|/2) over i<j
    # = (1/(2*beta)) * (-log(sum_{i<j} exp(-beta * |c_i - c_j|/2)))
    # = -(1/(2*beta)) * log(sum_{i<j} exp(-beta/2 * |c_i - c_j|))
    
    if n > 1:
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        d = pdist(c, metric='euclidean')
        pair_sum = np.sum(np.exp(-0.5 * beta * d))
        soft_min_pair = -(1.0 / (2.0 * beta)) * np.log(pair_sum)
    else:
        soft_min_pair = 0.5
    
    # The true radius is min(soft_min_wall, soft_min_pair)
    # But we want a smooth approximation of min(a, b).
    # soft_min(a, b) = -(1/beta)*log(exp(-beta*a) + exp(-beta*b))
    # However, a and b here are already soft approximations.
    # Better: combine all terms into one log-sum-exp.
    
    # Actually, let's just take the min of the two soft values as a simpler approach,
    # or better, combine them properly.
    
    # The true r = min(min_wall, min_pair/2)
    # min_wall = min_{i,k} (0.5 - |c_ik|)
    # min_pair/2 = min_{i<j} |c_i-c_j|/2
    
    # Combined soft-min over all constraints:
    # r_soft = -(1/beta) * log( sum_{i,k} exp(-beta*(0.5-|c_ik|)) + sum_{i<j} exp(-beta*|c_i-c_j|/2) )
    
    if n > 1:
        total_sum = wall_sum + pair_sum
    else:
        total_sum = wall_sum
    
    r_soft = -(1.0 / beta) * np.log(total_sum)
    
    return r_soft


def _soft_min_radius_grad(c, beta):
    """
    Gradient of the soft-min radius surrogate.
    """
    c = np.asarray(c).reshape(-1, 3)
    n = len(c)
    
    abs_c = np.abs(c)
    
    # Wall part
    wall_terms = np.exp(-beta * abs_c)  # (n, 3)
    wall_sum = np.sum(wall_terms)
    
    # Pair part
    if n > 1:
        from scipy.spatial.distance import pdist
        d = pdist(c, metric='euclidean')
        pair_terms = np.exp(-0.5 * beta * d)
        pair_sum = np.sum(pair_terms)
    else:
        d = np.array([])
        pair_terms = np.array([])
        pair_sum = 0.0
    
    total_sum = wall_sum + pair_sum
    
    if total_sum <= 0:
        return np.zeros_like(c)
    
    # d/dc of r_soft where r_soft = -(1/beta)*log(total_sum)
    # d r_soft / d c = -(1/beta) * (1/total_sum) * d total_sum / d c
    
    # d wall_sum / d c_i_k = -beta * sign(c_ik) * exp(-beta*|c_ik|)
    # For c_ik = 0, sign is 0 (subgradient), we use 0.
    wall_grad = -beta * np.sign(c) * wall_terms  # (n, 3)
    
    # d pair_sum / d c_i for each i
    # pair_sum = sum_{i<j} exp(-beta/2 * |c_i - c_j|)
    # d/d c_i of exp(-beta/2 * |c_i - c_j|) = exp(-beta/2*|c_i-c_j|) * (-beta/2) * (c_i - c_j)/|c_i - c_j|
    
    pair_grad = np.zeros_like(c)
    if n > 1:
        # Build full distance matrix
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (n, n, 3)
        dist = np.sqrt(np.sum(diff**2, axis=2))  # (n, n)
        
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-12, dist, 1.0)
        
        # Unit vectors
        unit = diff / safe_dist[:, :, np.newaxis]  # (n, n, 3)
        unit[dist < 1e-12] = 0.0
        
        # exp terms for all pairs (including i=j which we'll zero out)
        exp_terms = np.exp(-0.5 * beta * dist)  # (n, n)
        np.fill_diagonal(exp_terms, 0.0)
        
        # Gradient contribution from pair (i,j) to c_i:
        # exp(-beta/2*|c_i-c_j|) * (-beta/2) * (c_i-c_j)/|c_i-c_j|
        # = -beta/2 * exp_terms[i,j] * unit[i,j]
        
        # For each i, sum over j != i
        # grad_i = -beta/2 * sum_j exp_terms[i,j] * unit[i,j]
        
        weighted_unit = exp_terms[:, :, np.newaxis] * unit  # (n, n, 3)
        pair_grad = -0.5 * beta * np.sum(weighted_unit, axis=1)
    
    # Total gradient
    total_grad = wall_grad + pair_grad
    grad_r_soft = -(1.0 / beta) * total_grad / total_sum
    
    return grad_r_soft


def _fcc_lattice_init():
    """
    Generate an FCC lattice packing and select 100 points.
    FCC has a packing density of pi/(3*sqrt(2)) ≈ 0.7405.
    In a unit cube, we can fit multiple layers.
    
    For FCC in a cube of side 1, the conventional cubic cell has side a = 2*sqrt(2)*r
    where r is the sphere radius. But we want to place centres.
    
    Let's think differently: FCC lattice points in [0,1]^3.
    The FCC lattice can be described as:
    - Simple cubic points at (i, j, k) * a for integer i,j,k
    - Plus face-center points at (i+0.5, j+0.5, k)*a, etc.
    
    Actually, let's just create a dense grid and use FCC-like offsets.
    
    A simpler approach: create a cubic lattice with spacing s, then add FCC offsets.
    For 100 points in a unit cube, we need roughly 5x5x4 or similar.
    
    Let's use a 5x5x5 grid = 125 points, take the first 100.
    But to make it FCC-like, we can interleave.
    
    Actually, let's just use a well-spaced initial guess. The key insight from the
    observations is that FCC initialization gives a denser starting configuration.
    
    Let me create an FCC lattice properly:
    FCC basis vectors: a1 = (0, 0.5, 0.5), a2 = (0.5, 0, 0.5), a3 = (0.5, 0.5, 0)
    times some scale factor.
    
    For a unit cube, let's scale so points fit.
    """
    # Generate FCC lattice points in [0, 1]^3
    # FCC can be viewed as a BCC lattice rotated, or we can use the standard description.
    
    # Standard FCC in cubic cell of side a:
    # Corners: (0,0,0), (a,0,0), (0,a,0), (0,0,a), (a,a,0), (a,0,a), (0,a,a), (a,a,a)
    # Face centers: (a/2, a/2, 0), (a/2, 0, a/2), (0, a/2, a/2), (a, a/2, a/2), (a/2, a, a/2), (a/2, a/2, a)
    
    # The nearest neighbor distance in FCC is a/sqrt(2).
    # We want to fit as many as possible in [0,1]^3.
    
    # Let's try a few values of a and generate points, keeping those in [0,1]^3.
    
    best_points = None
    best_count = 0
    
    for a in np.linspace(0.15, 0.5, 50):
        n_div = int(np.ceil(1.0 / a))
        points = []
        for i in range(n_div + 1):
            for j in range(n_div + 1):
                for k in range(n_div + 1):
                    # Corner points
                    x, y, z = i * a, j * a, k * a
                    if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= z <= 1:
                        points.append([x, y, z])
                    # Face center points
                    for (dx, dy, dz) in [(0.5*a, 0.5*a, 0), (0.5*a, 0, 0.5*a), (0, 0.5*a, 0.5*a)]:
                        x2, y2, z2 = x + dx, y + dy, z + dz
                        if 0 <= x2 <= 1 and 0 <= y2 <= 1 and 0 <= z2 <= 1:
                            points.append([x2, y2, z2])
        
        if len(points) >= N:
            points = np.array(points)
            # Center in [-0.5, 0.5]^3
            points = points - 0.5
            if len(points) > N:
                # Take a subset - use greedy selection to maximize min distance
                # Or just take first N
                selected = points[:N]
            else:
                selected = points
            r = _compute_radius(selected)
            if best_points is None or r > _compute_radius(best_points):
                best_points = selected
                best_count = len(points)
    
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def construct():
    """Construct 100 sphere centres in [-0.5, 0.5]^3 maximizing the minimum radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with FCC lattice
    c = _fcc_lattice_init()
    c = c.reshape(-1, 3).astype(np.float64)
    
    # Ensure valid
    c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # Progressive beta schedule for soft-min
    betas = [10, 20, 50, 100, 200, 400, 800]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        # Define objective (we minimize negative soft-min radius)
        def obj(x):
            return -_soft_min_radius(x, beta)
        
        def obj_grad(x):
            return -_soft_min_radius_grad(x, beta)
        
        # Bounds: keep centres in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        try:
            result = minimize(
                obj,
                c,
                jac=obj_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50,
                }
            )
            
            if result.success or result.fun < obj(c):
                c_new = result.x.reshape(-1, 3)
                c_new = np.clip(c_new, -0.5 + 1e-10, 0.5 - 1e-10)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                c = c_new
        except Exception:
            pass
    
    # Final polish with very high beta or direct optimization
    if time.time() < deadline - 10:
        # Try a few more optimization steps with increasing beta
        for beta in [1000, 2000, 5000]:
            if time.time() > deadline - 5:
                break
            
            def obj(x):
                return -_soft_min_radius(x, beta)
            
            def obj_grad(x):
                return -_soft_min_radius_grad(x, beta)
            
            bounds = [(-0.5, 0.5)] * (3 * N)
            
            try:
                result = minimize(
                    obj,
                    best_c,
                    jac=obj_grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 30,
                    }
                )
                
                if result.success:
                    c_new = result.x.reshape(-1, 3)
                    c_new = np.clip(c_new, -0.5 + 1e-10, 0.5 - 1e-10)
                    r_new = _compute_radius(c_new)
                    if r_new > best_r:
                        best_r = r_new
                        best_c = c_new.copy()
            except Exception:
                pass
    
    # Final validation and return
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Ensure no coincident points
    # If any points are too close, nudge them apart slightly
    for _ in range(10):
        from scipy.spatial.distance import pdist
        d = pdist(best_c, metric='euclidean')
        if len(d) == 0:
            break
        min_d = np.min(d)
        if min_d > 1e-8:
            break
        # Find the closest pair and separate them
        idx = np.argmin(d)
        # Convert flat index to pair
        n_pts = len(best_c)
        # pdist index: for i<j, index = i*(2*n_pts - i - 1)//2 + j - i - 1
        # This is complex; just add small random perturbation
        best_c += np.random.randn(*best_c.shape) * 1e-10
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
