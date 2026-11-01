"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_radius(c, beta):
    """
    Compute a soft approximation of the minimum radius.
    Uses log-sum-exp to approximate the minimum of:
    - distances to boundaries
    - half pairwise distances
    """
    # Distances to boundaries
    # We want min(0.5 - |x|, 0.5 - |y|) for each point, which is min over 4 boundaries
    # Actually, min(0.5 - |c_i|) is the distance to the nearest boundary.
    # Let's construct the vector of all constraints:
    # 1. For each point i, distance to left, right, bottom, top boundaries.
    # 2. For each pair (i,j), half the distance between them.
    
    # Boundary terms: 0.5 - c[i,0], 0.5 + c[i,0], 0.5 - c[i,1], 0.5 + c[i,1]
    # Pair terms: 0.5 * ||c_i - c_j||
    
    # For efficiency, we can compute the soft min of the boundary distances and the soft min of the pair distances separately?
    # No, the min is over the union of all these constraints.
    # However, constructing the full vector of size 4N + N(N-1)/2 is expensive for N=200 (approx 40k terms).
    # Log-sum-exp on 40k terms is doable but might be slow if done repeatedly.
    
    # Alternative: Use a smaller set of active constraints or a different soft min.
    # Let's stick to the full set but optimize the calculation.
    
    # Boundary distances
    b_left = 0.5 - c[:, 0]
    b_right = 0.5 + c[:, 0]
    b_bottom = 0.5 - c[:, 1]
    b_top = 0.5 + c[:, 1]
    
    # Pairwise distances
    # This is the bottleneck. pdist is fast in C.
    d = pdist(c, metric='euclidean')
    p_terms = 0.5 * d
    
    # Combine all terms into one array
    # b_terms shape (4N), p_terms shape (N(N-1)/2)
    b_terms = np.concatenate([b_left, b_right, b_bottom, b_top])
    all_terms = np.concatenate([b_terms, p_terms])
    
    # Soft min: -1/beta * log(sum(exp(-beta * terms)))
    # To avoid overflow/underflow, we can use the log-sum-exp trick:
    # m = min(terms)
    # log_sum_exp(-beta * terms) = -beta * m + log(sum(exp(-beta * (terms - m))))
    # soft_min = -1/beta * (-beta * m + log(sum(exp(-beta * (terms - m)))))
    #           = m - 1/beta * log(sum(exp(-beta * (terms - m))))
    
    m = np.min(all_terms)
    # If m is very small, terms - m is small, exp is fine.
    # If beta is large, -beta * (terms - m) is <= 0, so exp is <= 1.
    # Sum of exps is <= N_terms.
    
    # We want to minimize the negative of the soft min, so we return -soft_min.
    # Or we can just return the soft min and minimize -soft_min.
    
    # For numerical stability with large beta:
    # If beta is very large, the term with the smallest value dominates.
    # exp(-beta * (terms - m)) will be 1 for the min terms and 0 for others.
    
    # Let's compute the log sum exp part
    # To avoid underflow for non-min terms when beta is large, we rely on the fact that they are > 0.
    # If terms - m > 1e-10, and beta=1000, exp(-100) is ~0.
    
    # Optimization: We don't need the exact sum if we are far from the minimum.
    # But for L-BFGS-B, we need a smooth gradient.
    
    # A cheaper approximation: only consider the k-smallest terms?
    # No, that introduces non-smoothness.
    
    # Let's use the full sum but be careful with performance.
    # N=200, terms ~ 40k. exp and sum on 40k floats is fast (~0.1ms).
    
    exp_terms = np.exp(-beta * (all_terms - m))
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min_val = m - (1.0 / beta) * log_sum
    
    return soft_min_val


def _objective(c_flat, beta):
    """Objective function for minimization: -soft_min_radius."""
    c = c_flat.reshape(-1, 2)
    # Ensure points are within bounds for the objective to be well-behaved?
    # L-BFGS-B handles bounds, but the objective should be smooth.
    # If a point goes outside, the boundary distance becomes negative.
    # The soft min will be negative. This is fine, it penalizes going outside.
    r = _soft_min_radius(c, beta)
    return -r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a few spacings
    for d in np.linspace(0.055, 0.075, 10):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _md_relax(c, r, time_limit=2.0):
    """
    Relaxes the configuration using Molecular Dynamics to resolve overlaps
    for circles of radius r.
    Returns the relaxed configuration.
    """
    c = c.copy()
    v = np.zeros_like(c)
    
    # Parameters
    dt = 0.001
    damping = 0.99
    k_rep = 100.0 # Spring constant for repulsion
    k_wall = 1000.0 # Spring constant for boundaries
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Calculate forces
        F = np.zeros_like(c)
        
        # Pairwise forces using broadcasting
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d_mat, 1.0)
        
        # Unit vectors
        unit_vec = diff / d_mat[:, :, np.newaxis]
        
        # Overlap
        overlap = np.maximum(0.0, 2*r - d_mat)
        
        # Force contribution from each pair j on i
        force_mat = k_rep * overlap[:, :, np.newaxis] * unit_vec
        F = np.sum(force_mat, axis=1)
        
        # Boundary forces
        # Push back if close to boundary (distance < r)
        F[:, 0] += k_wall * np.maximum(0.0, (-0.5 + r) - c[:, 0])
        F[:, 0] -= k_wall * np.maximum(0.0, c[:, 0] - (0.5 - r))
        F[:, 1] += k_wall * np.maximum(0.0, (-0.5 + r) - c[:, 1])
        F[:, 1] -= k_wall * np.maximum(0.0, c[:, 1] - (0.5 - r))
        
        # Velocity Verlet (simplified: Euler with damping)
        v = v * damping + F * dt
        c = c + v * dt
        
        # Hard clip to ensure they don't fly out
        c = np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Adaptive time step
        max_v = np.max(np.linalg.norm(v, axis=1))
        if max_v > 0.05:
            dt *= 0.5
        elif max_v < 0.01 and dt < 0.005:
            dt *= 1.5

    return c

def _find_max_radius_md(c0, time_limit=30.0):
    """
    Finds the maximum radius by iteratively increasing r and relaxing.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    
    best_c = c.copy()
    best_r = r
    
    dr = 0.0001
    max_tries = 500
    
    for _ in range(max_tries):
        if time.time() - start_time > time_limit:
            break
            
        r_try = r + dr
        c_try = _md_relax(c, r_try, time_limit=2.0)
        
        # Check validity of c_try with radius r_try
        d_mat = cdist(c_try, c_try)
        np.fill_diagonal(d_mat, 1.0)
        min_d = np.min(d_mat)
        min_b = np.min(0.5 - np.abs(c_try))
        
        if min_d >= 2*r_try - 1e-6 and min_b >= r_try - 1e-6:
            # Valid
            c = c_try
            r = r_try
            best_c = c.copy()
            best_r = r
        else:
            # Invalid, stop increasing
            break
            
    return best_c, best_r

def _polish_lbfgs(c0, max_time=10.0):
    """
    Final polish using L-BFGS-B with high beta.
    """
    start_time = time.time()
    c0_flat = c0.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Use high beta for precision
    b_val = 500.0
    def objective_wrapper(x):
        return _objective(x, b_val)
        
    try:
        res = minimize(
            objective_wrapper, 
            best_c.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 300}
        )
        
        c_opt = res.x.reshape(-1, 2)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    except Exception:
        pass

    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (120s limit)
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # MD Optimization
    if time.time() < deadline - 15:
        max_time_md = min(60.0, deadline - time.time() - 10)
        c_md, r_md = _find_max_radius_md(c_hex, time_limit=max_time_md)
        if r_md > best_radius:
            best_radius = r_md
            best_c = c_md.copy()
    
    # If we have time, try a second start from a slightly perturbed hexagonal grid
    # This helps if the first start got stuck in a local minimum.
    if time.time() < deadline - 25:
        np.random.seed(42)
        c_pert = best_c + np.random.normal(0, 0.005, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5, 0.5)
        
        max_time_md2 = min(40.0, deadline - time.time() - 10)
        c_md2, r_md2 = _find_max_radius_md(c_pert, time_limit=max_time_md2)
        if r_md2 > best_radius:
            best_radius = r_md2
            best_c = c_md2.copy()
            
    # Final Polish with L-BFGS-B
    if time.time() < deadline - 5:
        max_time_polish = min(15.0, deadline - time.time() - 2)
        c_pol, r_pol = _polish_lbfgs(best_c, max_time=max_time_polish)
        if r_pol > best_radius:
            best_radius = r_pol
            best_c = c_pol.copy()

    if best_c is None:
        best_c = _initialize_hexagonal()
    
    # Final projection to ensure strict validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
