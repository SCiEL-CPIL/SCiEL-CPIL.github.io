"""Fifty equal circles in a circle – SQP-style optimization with multi-start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
R_KNOWN = 0.1258254895304


def _score(pts):
    """Compute the packing radius (min of half-min-distance and min-distance-to-boundary)."""
    pts = np.asarray(pts)
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Distance to boundary
    radii = np.sqrt(np.sum(pts**2, axis=-1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)


def _objective_and_grad(x, r):
    """
    Objective: maximize the minimum of all pairwise distances/2 and (1 - |c_i|).
    We minimize the negative, but use a smooth approximation.
    Actually, for SQP we want to maximize r subject to constraints.
    Instead, let's use a different approach: minimize the maximum violation.
    
    Better approach: minimize sum of soft violations for a target radius r,
    but also try to push r up.
    
    Let's use: minimize f(x) = -r where r is the actual achievable radius,
    but that's non-smooth. 
    
    Alternative: For a fixed target radius r_target, minimize the sum of squared
    violations. Then do a binary search on r_target.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Boundary distances
    radii = np.sqrt(np.sum(pts**2, axis=-1))
    boundary_dist = 1.0 - radii
    
    # Violations: for pairs, violation = max(0, 2*r - dist)
    # For boundary, violation = max(0, r - boundary_dist)
    
    # For smoothness, use softplus approximation or just use the raw values
    # and rely on the optimizer to handle it
    
    # Energy: sum of squared violations
    energy = 0.0
    grad = np.zeros_like(pts)
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist_ij = np.sqrt(np.dot(d, d))
            if dist_ij < 1e-12:
                dist_ij = 1e-12
                d = np.array([1e-12, 0.0])
            
            if dist_ij < 2.0 * r:
                violation = 2.0 * r - dist_ij
                energy += violation**2
                factor = 2.0 * violation / dist_ij
                grad[i] += factor * d
                grad[j] -= factor * d
    
    # Boundary constraints
    for i in range(n):
        radius_i = np.sqrt(np.dot(pts[i], pts[i]))
        if radius_i < 1e-12:
            continue
        if radius_i > 1.0 - r:
            violation = radius_i - (1.0 - r)
            energy += violation**2
            factor = -2.0 * violation / radius_i
            grad[i] += factor * pts[i]
    
    return energy, grad.flatten()


def _hex_lattice_init(n):
    """Generate hexagonal lattice points inside unit circle."""
    points = []
    for spacing in np.linspace(0.25, 0.08, 30):
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x * x + y * y) <= 1.0:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n:
            radii = np.sqrt(np.sum(pts**2, axis=1))
            idx = np.argsort(radii, kind='stable')[:n]
            return pts[idx]
    return np.random.randn(n, 2) * 0.8


def _ring_init(n):
    """Generate ring-based initial configuration."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    
    # Rings
    ring_radii = [0.15, 0.35, 0.55, 0.75, 0.9]
    counts_per_ring = [6, 12, 18, 12, 2]
    
    for r_ring, count in zip(ring_radii, counts_per_ring):
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            pts.append([r_ring * np.cos(a), r_ring * np.sin(a)])
    
    pts = np.array(pts[:n])
    return pts


def _random_init(n):
    """Generate random initial configuration."""
    return np.random.randn(n, 2) * 0.9


def _optimize_for_radius(pts_init, r_target, max_iter=3000):
    """Optimize to find a valid configuration with radius >= r_target."""
    x0 = pts_init.flatten()
    
    def objective(x):
        e, g = _objective_and_grad(x, r_target)
        return e, g
    
    res = minimize(objective, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
    pts = res.x.reshape(-1, 2)
    return pts, res.fun


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    np.random.seed(42)
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_hex_lattice_init(N))
    inits.append(_ring_init(N))
    inits.append(_random_init(N))
    # A few more random starts
    for _ in range(3):
        inits.append(_random_init(N))
    
    best_pts = inits[0].copy()
    best_r = _score(best_pts)
    
    # Binary search on radius
    r_low = 0.10
    r_high = 0.13
    
    # First, find a good starting radius
    for init in inits:
        r_test = 0.12
        pts_opt, energy = _optimize_for_radius(init, r_test, max_iter=1500)
        actual_r = _score(pts_opt)
        if actual_r > best_r:
            best_r = actual_r
            best_pts = pts_opt.copy()
            r_low = max(r_low, actual_r - 0.001)
            r_high = min(r_high, actual_r + 0.005)
        
        if time.time() > deadline - 30:
            break
    
    # Binary search for max feasible radius
    for iteration in range(15):
        if time.time() > deadline - 25:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try the best initial point and a couple others
        success = False
        for init_idx in [0, 1, 2]:
            if time.time() > deadline - 20:
                break
            
            pts_opt, energy = _optimize_for_radius(inits[init_idx], r_mid, max_iter=2000)
            actual_r = _score(pts_opt)
            
            if actual_r >= r_mid - 1e-7:
                r_low = r_mid
                if actual_r > best_r:
                    best_r = actual_r
                    best_pts = pts_opt.copy()
                success = True
                break
        
        if not success:
            r_high = r_mid
        
        if r_high - r_low < 1e-7:
            break
    
    # Final polish: try to improve the best configuration
    if time.time() < deadline - 15:
        # Try a few more optimizations from the best point
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            # Slightly perturb the best point
            pts_perturbed = best_pts + np.random.randn(N, 2) * 0.001
            r_target = best_r + 0.0005
            pts_opt, energy = _optimize_for_radius(pts_perturbed, r_target, max_iter=2000)
            actual_r = _score(pts_opt)
            if actual_r > best_r:
                best_r = actual_r
                best_pts = pts_opt.copy()
    
    # Ensure validity
    best_pts = np.asarray(best_pts)
    radii = np.sqrt(np.sum(best_pts**2, axis=1))
    mask = radii > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / radii[mask, None] * 0.999
    
    return best_pts
