import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute triangle indices for vectorization
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    
    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
        try:
            pts = points - points.mean(axis=0)
            # Check for degenerate hull
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        
        if min_area < 0:
            return 0.0
        return min_area / area_hull

    def points_from_fourier(coeffs):
        """
        Generate 13 points from Fourier coefficients.
        coeffs: array of shape (2*K + 1) or similar, but let's use a fixed set of harmonics.
        Let's use K=3 harmonics.
        r(theta) = 1 + a1 cos(theta) + b1 sin(theta) + a2 cos(2theta) + b2 sin(2theta) + a3 cos(3theta) + b3 sin(3theta)
        Total 6 coefficients.
        Angles are fixed: 2*pi*k/13.
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        # coeffs: [a1, b1, a2, b2, a3, b3]
        r = np.ones(N)
        r += coeffs[0] * np.cos(1 * angles) + coeffs[1] * np.sin(1 * angles)
        r += coeffs[2] * np.cos(2 * angles) + coeffs[3] * np.sin(2 * angles)
        r += coeffs[4] * np.cos(3 * angles) + coeffs[5] * np.sin(3 * angles)
        
        # Ensure r > 0 to maintain convexity/validity
        r = np.maximum(r, 0.1)
        
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        return np.column_stack([xs, ys])

    def objective_fourier(coeffs):
        pts = points_from_fourier(coeffs)
        score = compute_score(pts)
        return -score

    def objective_full(points_flat):
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        return -score

    best_score = -float('inf')
    best_points = None

    # Phase 1: Optimize in the low-dimensional Fourier subspace
    # This reduces the search space significantly and enforces a smooth convex boundary.
    # Initial guess: Regular polygon (all coeffs = 0)
    initial_coeffs = np.zeros(6)
    
    # Try a few perturbations to kickstart
    inits = [
        initial_coeffs,
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    ]
    
    for init_c in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(objective_fourier, init_c, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = points_from_fourier(res.x).copy()
        except Exception:
            pass

    # If Fourier search didn't find anything good (shouldn't happen, but just in case)
    if best_points is None:
        best_points = points_from_fourier(initial_coeffs).copy()
        best_score = compute_score(best_points)

    # Phase 2: Polish in full 26D space
    # Start from the best Fourier-based configuration
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective_full, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            candidate_pts = res.x.reshape(-1, 2)
            candidate_score = compute_score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_pts.copy()
        except Exception:
            pass

    # Phase 3: Random restarts in full space if time permits
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(42)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Random points on a circle with small noise
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.randn(N) * 0.1
            radii = 1.0 + rng.randn(N) * 0.1
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            try:
                res = minimize(objective_full, pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
                candidate_pts = res.x.reshape(-1, 2)
                candidate_score = compute_score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts.copy()
            except Exception:
                continue

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Center the points
    best_points = best_points - np.mean(best_points, axis=0)
    
    return best_points
