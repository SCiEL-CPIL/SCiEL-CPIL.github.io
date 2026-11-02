"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import Delaunay, cKDTree


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


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, trying multiple spacings."""
    best_c = None
    best_r = 0.0
    for d in np.linspace(0.054, 0.076, 15):
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
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _delaunay_edges(c):
    """Get Delaunay triangle edges as a list of (i, j) pairs."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = int(simplex[a]), int(simplex[b])
                    if i != j:
                        edges.add((min(i, j), max(i, j)))
        return list(edges)
    except Exception:
        # Fallback: use k-nearest neighbors
        tree = cKDTree(c)
        pairs, _ = tree.query_pairs(0.08, output_type='ndarray')
        return [(int(min(p)), int(max(p))) for p in pairs]


def _local_energy_and_grad(c, r):
    """
    Compute energy (violation) and gradient for local optimization.
    Energy = sum of squared violations for active constraints.
    We want to push r up while keeping all constraints satisfied.
    """
    n = c.shape[0]
    # Get Delaunay edges for efficient pairwise computation
    edges = _delaunay_edges(c)
    
    # Compute all relevant distances
    # Boundary violations
    b_left = 0.5 - c[:, 0] - r
    b_right = 0.5 + c[:, 0] - r
    b_bottom = 0.5 - c[:, 1] - r
    b_top = 0.5 + c[:, 1] - r
    
    # Pairwise violations for Delaunay edges
    pair_violations = np.zeros(len(edges))
    for k, (i, j) in enumerate(edges):
        d = np.linalg.norm(c[i] - c[j])
        pair_violations[k] = d - 2.0 * r
    
    # Energy: sum of squared negative parts (violations)
    # Use a penalty approach: energy = sum(max(0, -violation)^2)
    all_violations = np.concatenate([
        b_left, b_right, b_bottom, b_top, pair_violations
    ])
    
    # Only count violations (negative values)
    neg_viol = np.maximum(0.0, -all_violations)
    energy = np.sum(neg_viol ** 2)
    
    # Gradient w.r.t. c
    grad_c = np.zeros_like(c)
    
    # Boundary gradients
    # d(b_left)/dc = [-1, 0], d(b_right)/dc = [1, 0], etc.
    # d(energy)/dc = 2 * sum(-violation) * d(violation)/dc for violated constraints
    
    mask_left = b_left < 0
    mask_right = b_right < 0
    mask_bottom = b_bottom < 0
    mask_top = b_top < 0
    
    # For each violated boundary constraint, add gradient contribution
    # energy += (-b)^2 where b < 0, so d(energy)/dc = 2*(-b)*d(b)/dc = -2*b*d(b)/dc
    # Since b < 0, -2*b > 0, so gradient points in direction to increase b
    
    for idx in np.where(mask_left)[0]:
        grad_c[idx, 0] += -2.0 * b_left[idx] * (-1.0)  # d(b_left)/dx = -1
    for idx in np.where(mask_right)[0]:
        grad_c[idx, 0] += -2.0 * b_right[idx] * (1.0)   # d(b_right)/dx = 1
    for idx in np.where(mask_bottom)[0]:
        grad_c[idx, 1] += -2.0 * b_bottom[idx] * (-1.0) # d(b_bottom)/dy = -1
    for idx in np.where(mask_top)[0]:
        grad_c[idx, 1] += -2.0 * b_top[idx] * (1.0)     # d(b_top)/dy = 1
    
    # Pairwise gradients
    for k, (i, j) in enumerate(edges):
        if pair_violations[k] < 0:
            d = np.linalg.norm(c[i] - c[j])
            if d < 1e-12:
                continue
            # violation = d - 2r, d(violation)/d(c_i) = (c_i - c_j) / d
            # d(energy)/dc_i = -2 * violation * d(violation)/dc_i = -2 * (d-2r) * (c_i-c_j)/d
            # Since violation < 0, d-2r < 0, so -2*(d-2r) > 0
            # Gradient on c_i: -2*(d-2r) * (c_i-c_j)/d
            # Gradient on c_j: -2*(d-2r) * (c_j-c_i)/d
            factor = -2.0 * pair_violations[k] / d
            diff = c[i] - c[j]
            grad_c[i] += factor * diff
            grad_c[j] -= factor * diff
    
    return energy, grad_c


def _local_optimize(c0, max_time=30.0, r_target=None):
    """
    Local optimization using gradient descent on the violation energy.
    Strategy: increase r gradually, and at each step, minimize violations.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    best_r = r
    best_c = c.copy()
    
    if r_target is None:
        r_target = r + 0.002  # Try to increase by 0.002
    
    # Phase 1: Increase r in small steps, minimizing violations at each step
    dr = 0.0002
    max_steps = 50
    
    for step in range(max_steps):
        if time.time() - start_time > max_time - 5.0:
            break
        
        r_try = r + dr
        if r_try > r_target:
            r_try = r_target
        
        # Minimize violations at this r
        for inner_iter in range(20):
            if time.time() - start_time > max_time - 5.0:
                break
            
            energy, grad_c = _local_energy_and_grad(c, r_try)
            
            if energy < 1e-10:
                break
            
            # Gradient descent step
            lr = 0.01
            # Adaptive step size
            c_new = c - lr * grad_c
            c_new = np.clip(c_new, -0.5, 0.5)
            
            # Check if this improves
            energy_new, _ = _local_energy_and_grad(c_new, r_try)
            if energy_new < energy:
                c = c_new
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        # Check if we can accept this r
        final_energy, _ = _local_energy_and_grad(c, r_try)
        if final_energy < 1e-8:
            r = r_try
            # Update best
            actual_r = _compute_radius(c)
            if actual_r > best_r:
                best_r = actual_r
                best_c = c.copy()
        else:
            # Try smaller increment
            dr *= 0.5
            if dr < 1e-6:
                break
    
    return best_c, best_r


def _delaunay_refine(c0, max_time=40.0):
    """
    Delaunay-based refinement: use the Delaunay graph to identify
    critical pairs and optimize only those constraints.
    """
    start_time = time.time()
    c = c0.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    # Iterative improvement
    for iteration in range(20):
        if time.time() - start_time > max_time - 5.0:
            break
        
        r_current = _compute_radius(c)
        
        # Get Delaunay edges
        edges = _delaunay_edges(c)
        
        # Compute minimum distance among Delaunay edges
        min_d = float('inf')
        min_i, min_j = 0, 1
        for i, j in edges:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
                min_i, min_j = i, j
        
        # Also check boundary
        min_boundary = np.min(0.5 - np.abs(c))
        
        if min_d / 2.0 < min_boundary:
            r_current = min_d / 2.0
        else:
            r_current = min_boundary
        
        # Try to increase r by pushing the critical pair apart
        # and adjusting others
        dr = 0.0005
        
        for sub_iter in range(10):
            if time.time() - start_time > max_time - 5.0:
                break
            
            r_try = r_current + dr
            energy, grad_c = _local_energy_and_grad(c, r_try)
            
            if energy < 1e-10:
                break
            
            # Gradient step
            lr = 0.005
            c_new = c - lr * grad_c
            c_new = np.clip(c_new, -0.5, 0.5)
            
            energy_new, _ = _local_energy_and_grad(c_new, r_try)
            if energy_new < energy:
                c = c_new
            else:
                break
        
        # Check if improvement
        new_r = _compute_radius(c)
        if new_r > best_r:
            best_r = new_r
            best_c = c.copy()
        
        # If no progress, break
        if new_r <= r_current + 1e-8:
            break
    
    return best_c, best_r


def _annealed_local_search(c0, max_time=50.0):
    """
    Annealed local search: combine local optimization with random perturbations
    to escape local minima.
    """
    start_time = time.time()
    c = c0.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    # Temperature schedule
    temps = [0.01, 0.008, 0.006, 0.004, 0.003, 0.002, 0.001]
    
    for temp in temps:
        if time.time() - start_time > max_time - 10.0:
            break
        
        # Try multiple perturbations at this temperature
        for restart in range(3):
            if time.time() - start_time > max_time - 10.0:
                break
            
            # Perturb
            c_pert = c + np.random.normal(0, temp, size=c.shape)
            c_pert = np.clip(c_pert, -0.5, 0.5)
            
            # Local optimize
            max_time_sub = min(10.0, max_time - (time.time() - start_time) - 8.0)
            if max_time_sub < 3.0:
                break
            
            c_opt, r_opt = _local_optimize(c_pert, max_time=max_time_sub)
            
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
                c = c_opt.copy()
            else:
                # Accept with probability based on temperature (simulated annealing)
                delta_r = best_r - r_opt
                if delta_r > 0 and np.random.random() < np.exp(-delta_r / (temp * 0.1)):
                    c = c_opt.copy()
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Phase 1: Delaunay-based refinement (structural change from soft-min)
    if time.time() < deadline - 40:
        max_time_p1 = min(40.0, deadline - time.time() - 35)
        c_opt, r_opt = _delaunay_refine(c_hex, max_time=max_time_p1)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Phase 2: Local optimization with r-increase strategy
    if time.time() < deadline - 30:
        max_time_p2 = min(30.0, deadline - time.time() - 25)
        c_opt2, r_opt2 = _local_optimize(best_c, max_time=max_time_p2)
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
    
    # Phase 3: Annealed local search to escape local minima
    if time.time() < deadline - 15:
        max_time_p3 = min(35.0, deadline - time.time() - 12)
        c_opt3, r_opt3 = _annealed_local_search(best_c, max_time=max_time_p3)
        if r_opt3 > best_radius:
            best_radius = r_opt3
            best_c = c_opt3.copy()
    
    # Phase 4: Final polish with another Delaunay refinement
    if time.time() < deadline - 5:
        max_time_p4 = min(15.0, deadline - time.time() - 4)
        c_pol, r_pol = _delaunay_refine(best_c, max_time=max_time_p4)
        if r_pol > best_radius:
            best_radius = r_pol
            best_c = c_pol.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    # Final projection to ensure strict validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
