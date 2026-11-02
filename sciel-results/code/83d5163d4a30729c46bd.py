"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
import time
from itertools import combinations

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) -
                                  (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute convex hull area using Andrew's monotone chain."""
    pts = np.array(pts, dtype=float)
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _fourier_boundary(coeffs, n_samples=200):
    """
    Generate points on a smooth convex boundary from Fourier coefficients.
    coeffs: array of shape (2*(K+1),) where K is max harmonic.
    The radius is r(theta) = a0 + sum_{k=1}^{K} (ak*cos(k*theta) + bk*sin(k*theta))
    We ensure positivity by adding a large constant offset.
    """
    K = (len(coeffs) - 1) // 2
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
    r = np.full(n_samples, coeffs[0])  # a0
    for k in range(1, K + 1):
        ak = coeffs[2 * k - 1]
        bk = coeffs[2 * k]
        r += ak * np.cos(k * thetas) + bk * np.sin(k * thetas)
    
    # Ensure strict positivity for convexity
    r_min = r.min()
    if r_min <= 0.1:
        r = r + (0.1 - r_min + 1e-6)
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return np.column_stack([x, y])


def _place_points_on_boundary(boundary_pts, n_points, phase_shifts=None):
    """
    Place n_points at roughly equal arc-length intervals along the boundary,
    with optional phase shifts.
    """
    n = len(boundary_pts)
    if phase_shifts is None:
        phase_shifts = np.zeros(n_points)
    
    # Compute cumulative arc length
    diffs = np.diff(boundary_pts, axis=0, append=boundary_pts[:1])
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_length = np.cumsum(seg_lengths)
    total_length = cum_length[-1]
    
    # Target positions along the boundary
    target_fracs = (np.arange(n_points) + phase_shifts) / n_points
    target_fracs = target_fracs % 1.0
    
    # Find positions on the boundary
    points = np.zeros((n_points, 2))
    for i in range(n_points):
        target_len = target_fracs[i] * total_length
        # Binary search for segment
        idx = np.searchsorted(cum_length, target_len)
        if idx == 0:
            points[i] = boundary_pts[0]
        elif idx >= n:
            points[i] = boundary_pts[-1]
        else:
            seg_start = cum_length[idx - 1]
            seg_end = cum_length[idx]
            t = (target_len - seg_start) / (seg_end - seg_start + 1e-15)
            points[i] = (1 - t) * boundary_pts[idx - 1] + t * boundary_pts[idx]
    
    return points


def _optimize_phase_shifts(boundary_pts, n_points, initial_shifts, max_iter=300, lr=0.01):
    """
    Optimize phase shifts to maximize the minimum triangle area.
    Uses simple gradient-free local search.
    """
    best_shifts = initial_shifts.copy()
    best_score = _score(_place_points_on_boundary(boundary_pts, n_points, best_shifts))
    
    for iteration in range(max_iter):
        # Try perturbing each shift
        improved = False
        for i in range(n_points):
            for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                new_shifts = best_shifts.copy()
                new_shifts[i] += delta
                new_shifts[i] = new_shifts[i] % 1.0
                pts = _place_points_on_boundary(boundary_pts, n_points, new_shifts)
                s = _score(pts)
                if s > best_score + 1e-10:
                    best_score = s
                    best_shifts = new_shifts
                    improved = True
        if not improved:
            break
    
    return best_shifts, best_score


def construct():
    """
    Construct 14 points maximizing min triangle area / convex hull area.
    Uses Fourier-parameterized convex boundary + optimized point placement.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Regular polygon baseline
    theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(theta_reg), np.sin(theta_reg)])
    reg_score = _score(reg_pts)
    best_points = reg_pts.copy()
    best_score = reg_score
    
    # Strategy 2: Fourier boundary with various harmonic structures
    n_samples = 300  # samples along boundary
    
    # Try different numbers of harmonics and random coefficient sets
    rng = np.random.RandomState(42)
    
    for K in [1, 2, 3, 4]:
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Generate random Fourier coefficients
            # a0 should be large enough to ensure convexity
            a0 = 1.0 + rng.random() * 0.5
            coeffs = np.zeros(2 * (K + 1))
            coeffs[0] = a0
            for k in range(1, K + 1):
                # Smaller coefficients for higher harmonics
                scale = 0.3 / k
                coeffs[2 * k - 1] = rng.uniform(-scale, scale)
                coeffs[2 * k] = rng.uniform(-scale, scale)
            
            boundary = _fourier_boundary(coeffs, n_samples)
            
            # Check if boundary is convex (all points should be on convex hull)
            hull_area = _convex_hull_area(boundary)
            boundary_area = _convex_hull_area(boundary)
            if hull_area < 0.95 * boundary_area:
                continue  # Not convex enough
            
            # Place points and optimize
            initial_shifts = np.zeros(N)
            pts = _place_points_on_boundary(boundary, N, initial_shifts)
            s = _score(pts)
            
            if s > best_score:
                best_score = s
                best_points = pts.copy()
            
            # Local optimization of phase shifts
            if time.time() < deadline - 5:
                opt_shifts, opt_score = _optimize_phase_shifts(
                    boundary, N, initial_shifts, max_iter=100, lr=0.01
                )
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = _place_points_on_boundary(boundary, N, opt_shifts).copy()
    
    # Strategy 3: Try slightly perturbed regular polygon
    for trial in range(50):
        if time.time() > deadline:
            break
        
        theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Add small angular perturbations
        perturbations = rng.uniform(-0.05, 0.05, N)
        theta_perturbed = theta_reg + perturbations
        theta_perturbed = np.sort(theta_perturbed)
        
        # Vary radius slightly
        radii = 1.0 + rng.uniform(-0.1, 0.1, N)
        
        pts = np.column_stack([radii * np.cos(theta_perturbed), 
                               radii * np.sin(theta_perturbed)])
        s = _score(pts)
        
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Strategy 4: Direct local optimization from best found
    if best_points is not None and time.time() < deadline - 10:
        # Try small random perturbations
        for trial in range(200):
            if time.time() > deadline:
                break
            
            perturbed = best_points + rng.normal(0, 0.05, (N, 2))
            s = _score(perturbed)
            
            if s > best_score:
                best_score = s
                best_points = perturbed.copy()
    
    return best_points

# EVOLVE-BLOCK-END
