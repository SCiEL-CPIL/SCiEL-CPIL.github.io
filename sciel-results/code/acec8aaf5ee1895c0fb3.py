"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """Feasible common radius: min of boundary clearance and half min pairwise distance."""
    if points is None or len(points) < 2:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dist_boundary))
    d = pdist(points)
    min_pair = float(np.min(d)) / 2.0
    return min(min_boundary, min_pair)


def _hexagonal_lattice(spacing):
    """Hexagonal lattice points inside the unit disk (with margin), closest to centre."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _concentric_rings(n_rings):
    """Concentric rings packing; returns points or None."""
    if n_rings <= 0:
        return None
    if n_rings == 1:
        ring_counts = [N]
    else:
        remaining = N - 1
        base = remaining // (n_rings - 1)
        extra = remaining % (n_rings - 1)
        ring_counts = [1] + [base + (1 if i < extra else 0) for i in range(n_rings - 1)]
    pts = []
    if ring_counts[0] > 0:
        pts.append([0.0, 0.0])
    for ring_idx in range(1, n_rings):
        count = ring_counts[ring_idx]
        if count <= 0:
            continue
        ring_radius = ring_idx / n_rings * 0.95
        for i in range(count):
            angle = 2 * np.pi * i / count
            pts.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
    if len(pts) != N:
        return None
    return np.array(pts)


def _square_grid(spacing):
    """Square grid points inside the unit disk, closest to centre."""
    k = int(1.0 / spacing) + 3
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing
            y = j * spacing
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _soft_min_objective(points, t=50.0):
    """
    Smooth surrogate for the objective: maximize soft-min of all constraints.
    Returns a scalar value (higher is better).
    Uses log-sum-exp for soft-min of pairwise distances and boundary distances.
    """
    # Boundary distances
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    
    # Pairwise distances using KDTree for efficiency
    tree = cKDTree(points)
    # Get nearest neighbor distances (excluding self)
    d, _ = tree.query(points, k=2)
    min_pairwise = d[:, 1]  # distance to nearest neighbor
    
    # Soft-min of boundary distances
    # soft-min(a) = -logsumexp(-a) / 1 ... actually soft-min = -soft-max(-a)
    # soft-max(a) = log(sum(exp(a/t))) * t ... let's use the standard formula
    # soft-min of set S = -t * log(sum(exp(-s_i / t)))
    
    # Combine all constraints: boundary distances and pairwise distances
    # We want soft-min of all (dist_boundary_i) and (min_pairwise_j)
    # Actually, the true constraint is min(min_boundary, min_pairwise/2)
    # Let's define the "constraint values" as:
    #   c_i = dist_boundary_i for i in 0..N-1
    #   d_j = min_pairwise_j / 2.0 for j in 0..N-1 (each point's nearest neighbor distance / 2)
    # The feasible radius is min(all c_i, all d_j)
    
    constraints = np.concatenate([dist_boundary, min_pairwise / 2.0])
    
    # soft-min via log-sum-exp
    # soft_min = -t * log(sum(exp(-c / t)))
    max_c = np.max(constraints)
    exp_terms = np.exp(-(constraints - max_c) / t)
    sum_exp = np.sum(exp_terms)
    soft_min_val = max_c - t * np.log(sum_exp)
    
    return soft_min_val


def _gradient_descent_smooth(points, deadline, max_iters=2000, lr=0.001, t=50.0):
    """
    Gradient ascent on the soft-min surrogate using numerical gradients.
    """
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    best_soft = _soft_min_objective(points, t)
    rng = np.random.default_rng(999)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Numerical gradient of soft-min objective
        grad = np.zeros_like(points)
        eps = 1e-4
        f0 = _soft_min_objective(points, t)
        
        for i in range(N):
            for d in range(2):
                old = points[i, d]
                points[i, d] = old + eps
                f_plus = _soft_min_objective(points, t)
                points[i, d] = old - eps
                f_minus = _soft_min_objective(points, t)
                points[i, d] = old
                grad[i, d] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient ascent step
        points += lr * grad
        
        # Project back into feasible region approximately
        # Ensure points stay within unit circle
        norms = np.linalg.norm(points, axis=1)
        for i in range(N):
            if norms[i] > 0.99:
                points[i] *= 0.99 / norms[i]
        
        new_soft = _soft_min_objective(points, t)
        new_r = _compute_radius(points)
        
        if new_soft > best_soft:
            best_soft = new_soft
            best_r = new_r
            best_points = points.copy()
    
    return best_points, best_r


def _refine_local(points, deadline, max_iters=2000, step_size=0.02):
    """Hill-climbing local refinement; respects the deadline."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    rng = np.random.default_rng(12345)
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, len(points))
        original = points[idx].copy()
        # Try several perturbations of this point; keep the best improving one.
        best_local = original
        best_local_r = best_r
        for _ in range(6):
            if time.time() > deadline:
                break
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            points[idx] = original + np.array([dx, dy])
            r = _compute_radius(points)
            if r > best_local_r:
                best_local_r = r
                best_local = points[idx].copy()
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
    return best_points, best_r


def _refine_simulated_annealing(points, deadline, max_iters=5000, T_start=0.01, T_end=0.0001):
    """Simulated annealing local refinement; respects the deadline."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    current_r = best_r
    rng = np.random.default_rng(777)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (i / max_iters)
        
        idx = rng.integers(0, len(points))
        original = points[idx].copy()
        
        # Random perturbation
        dx = rng.uniform(-0.05, 0.05)
        dy = rng.uniform(-0.05, 0.05)
        points[idx] = original + np.array([dx, dy])
        
        new_r = _compute_radius(points)
        delta_r = new_r - current_r
        
        # Accept if better or with probability exp(delta_r / T)
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            points[idx] = original
            
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s wall-clock limit

    candidates = []

    # Hexagonal lattice: try a range of spacings and keep the best feasible.
    for trial in range(40):
        spacing = 0.05 + trial * 0.005
        pts = _hexagonal_lattice(spacing)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    # Concentric rings: a few ring counts.
    for n_rings in range(3, 20):
        pts = _concentric_rings(n_rings)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    # Square grid: a range of spacings.
    for trial in range(30):
        spacing = 0.05 + trial * 0.005
        pts = _square_grid(spacing)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        # Fallback: single ring of N points.
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Refine the best candidates while time permits.
    # First pass: hill climbing
    for cand_idx in range(min(3, len(candidates))):
        if time.time() > deadline - 40.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _refine_local(cand_pts, deadline - 30.0,
                                               max_iters=1000, step_size=0.02)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Second pass: simulated annealing on the best result so far
    if time.time() < deadline - 20.0:
        refined_pts, refined_r = _refine_simulated_annealing(best_points, deadline - 15.0,
                                                             max_iters=2000, T_start=0.01, T_end=0.0001)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Third pass: smooth gradient ascent on soft-min surrogate
    if time.time() < deadline - 5.0:
        refined_pts, refined_r = _gradient_descent_smooth(best_points, deadline - 2.0,
                                                          max_iters=500, lr=0.0005, t=50.0)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Final polish: very tight hill climbing with small steps
    if time.time() < deadline - 1.0:
        refined_pts, refined_r = _refine_local(best_points, deadline,
                                               max_iters=500, step_size=0.005)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    return best_points

# EVOLVE-BLOCK-END
