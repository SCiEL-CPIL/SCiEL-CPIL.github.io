# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import sys

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    Uses SDP-inspired Gram matrix formulation for global structure,
    then soft-min surrogate and hard-min polish.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s limit
    
    # Precompute triangle index pairs
    tri_idx = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_idx.append((i, j, k))
    tri_idx = np.array(tri_idx)  # shape (286, 3)
    n_tris = len(tri_idx)
    
    def compute_triangle_areas(points):
        """Compute all 286 triangle areas efficiently."""
        i, j, k = tri_idx[:, 0], tri_idx[:, 1], tri_idx[:, 2]
        p1 = points[i]  # (286, 2)
        p2 = points[j]
        p3 = points[k]
        # Cross product: (p2-p1) x (p3-p1)
        d1 = p2 - p1
        d2 = p3 - p1
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        areas = 0.5 * np.abs(cross)
        return areas
    
    def compute_hull_area(points):
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return np.inf
    
    def hard_min_score(points):
        """True score: min triangle area / hull area."""
        if not np.all(np.isfinite(points)):
            return -np.inf
        a_hull = compute_hull_area(points)
        if a_hull <= 1e-10:
            return -np.inf
        areas = compute_triangle_areas(points)
        return np.min(areas) / a_hull
    
    def soft_min_objective(x, beta):
        """
        Soft-min surrogate: minimize -softmin(areas) / hull_area
        """
        points = x.reshape(n, 2)
        if not np.all(np.isfinite(points)):
            return 1e6
        a_hull = compute_hull_area(points)
        if a_hull <= 1e-10:
            return 1e6
        areas = compute_triangle_areas(points)
        m = np.min(areas)
        exp_terms = np.exp(-beta * (areas - m))
        log_sum = np.log(np.sum(exp_terms))
        softmin_val = m - (1.0/beta) * log_sum
        val = -softmin_val / a_hull
        if not np.isfinite(val):
            return 1e6
        return val
    
    def hard_min_objective(x):
        """Objective for final polish: -hard_min_score."""
        points = x.reshape(n, 2)
        score = hard_min_score(points)
        if score == -np.inf:
            return 1e6
        return -score
    
    # SDP-inspired initialization: use Gram matrix approach
    def sdp_inspired_init():
        """
        Use a Gram matrix approach to find a good configuration.
        We maximize the minimum squared triangle area subject to
        fixed total hull area, using iterative convex relaxations.
        """
        rng = np.random.default_rng(2024)
        
        # Start with a regular polygon as base
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Iteratively improve by solving for better positions
        for iteration in range(50):
            if time.time() > deadline - 120:
                break
                
            # Compute current triangle areas
            areas = compute_triangle_areas(points)
            min_area_idx = np.argmin(areas)
            
            # Get the three points forming the minimum area triangle
            i, j, k = tri_idx[min_area_idx]
            
            # Try to move these points to increase their triangle area
            # while maintaining overall structure
            p_i = points[i].copy()
            p_j = points[j].copy()
            p_k = points[k].copy()
            
            # Compute centroid of the triangle
            centroid = (p_i + p_j + p_k) / 3.0
            
            # Move each point away from centroid slightly
            for idx in [i, j, k]:
                direction = points[idx] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    # Small step away from centroid
                    points[idx] += 0.02 * direction
            
            # Also try moving points that are too close to others
            for a in range(n):
                for b in range(a+1, n):
                    dist = np.linalg.norm(points[a] - points[b])
                    if dist < 0.3:
                        # Push apart
                        mid = (points[a] + points[b]) / 2
                        dir_a = points[a] - mid
                        dir_b = points[b] - mid
                        norm_a = np.linalg.norm(dir_a)
                        norm_b = np.linalg.norm(dir_b)
                        if norm_a > 1e-10:
                            points[a] += 0.01 * dir_a / norm_a
                        if norm_b > 1e-10:
                            points[b] += 0.01 * dir_b / norm_b
            
            # Normalize to unit scale
            center = np.mean(points, axis=0)
            points -= center
            max_norm = np.max(np.linalg.norm(points, axis=1))
            if max_norm > 1e-10:
                points /= max_norm
        
        return points.flatten()
    
    def init_regular_polygon():
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 1.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.column_stack((x, y)).flatten()
    
    def init_ring_structure():
        """Two concentric rings: outer 7, inner 6, offset."""
        rng = np.random.default_rng(42)
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        outer_r = 1.0
        inner_r = 0.5
        ox = outer_r * np.cos(outer_angles)
        oy = outer_r * np.sin(outer_angles)
        ix = inner_r * np.cos(inner_angles)
        iy = inner_r * np.sin(inner_angles)
        pts = np.column_stack([np.concatenate([ox, ix]), np.concatenate([oy, iy])])
        pts += rng.normal(0, 0.05, pts.shape)
        return pts.flatten()
    
    def init_alternating_radii():
        """Points on a circle with alternating radii."""
        rng = np.random.default_rng(43)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = np.ones(n)
        radii[::2] = 0.6  # even indices get smaller radius
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        pts = np.column_stack([x, y])
        pts += rng.normal(0, 0.05, pts.shape)
        return pts.flatten()
    
    def init_triangular_lattice():
        """Triangular lattice with 13 points."""
        pts = []
        for i in range(5):
            for j in range(5 - i):
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2) * j
                pts.append([x, y])
        pts = np.array(pts)
        com = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - com, axis=1)
        indices_to_remove = np.argsort(dists)[:2]
        mask = np.ones(15, dtype=bool)
        mask[indices_to_remove] = False
        pts_13 = pts[mask]
        # Normalize to unit scale
        center = np.mean(pts_13, axis=0)
        pts_13 -= center
        max_norm = np.max(np.linalg.norm(pts_13, axis=1))
        pts_13 /= max_norm
        return pts_13.flatten()
    
    def init_random_cluster():
        """Random points in a disk."""
        rng = np.random.default_rng(44)
        angles = rng.uniform(0, 2*np.pi, n)
        radii = rng.uniform(0.3, 1.0, n)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y]).flatten()
    
    def init_sdp_relaxed():
        """
        SDP relaxation approach: maximize minimum squared distance
        subject to Gram matrix constraints.
        """
        # Use a simple iterative approach inspired by SDP
        # Start from regular polygon and iteratively improve
        points = init_regular_polygon().reshape(n, 2)
        rng = np.random.default_rng(2025)
        
        for it in range(100):
            if time.time() > deadline - 100:
                break
            
            # Compute pairwise distances
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(points[i] - points[j])
                    dists[i, j] = d
                    dists[j, i] = d
            
            # Find minimum distance pair
            min_dist = np.min(dists)
            
            # For each point, try to increase its minimum distance to others
            for i in range(n):
                # Find nearest neighbor
                nearest = np.argmin(dists[i])
                if nearest == i:
                    continue
                
                # Direction away from nearest neighbor
                direction = points[i] - points[nearest]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    # Small step
                    points[i] += 0.01 * direction
            
            # Normalize
            center = np.mean(points, axis=0)
            points -= center
            max_norm = np.max(np.linalg.norm(points, axis=1))
            if max_norm > 1e-10:
                points /= max_norm
        
        return points.flatten()
    
    # Generate diverse initializations
    inits = []
    inits.append(init_regular_polygon())
    inits.append(init_ring_structure())
    inits.append(init_alternating_radii())
    inits.append(init_triangular_lattice())
    inits.append(init_random_cluster())
    inits.append(sdp_inspired_init())
    inits.append(init_sdp_relaxed())
    
    # Add perturbed versions
    rng = np.random.default_rng(123)
    for base_init in [init_regular_polygon(), init_ring_structure(), sdp_inspired_init()]:
        for scale in [0.05, 0.1, 0.2]:
            base = base_init.reshape(n, 2)
            perturbed = base + rng.normal(0, scale, (n, 2))
            inits.append(perturbed.flatten())
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Optimize with soft-min surrogate using L-BFGS-B
    beta_values = [10, 20, 50, 100, 200, 400]
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 60:
            break
        
        # For each init, do a sequence of soft-min optimizations with increasing beta
        current_x = init.copy()
        for beta in beta_values:
            if time.time() > deadline - 30:
                break
            
            try:
                res = minimize(
                    lambda x: soft_min_objective(x, beta),
                    current_x,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < 1e5:
                    current_x = res.x
            except Exception:
                pass
        
        # Evaluate hard-min score
        points = current_x.reshape(n, 2)
        score = hard_min_score(points)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 2: Polish the best result with hard-min objective using Nelder-Mead
    if best_points is not None and time.time() < deadline - 30:
        try:
            res = minimize(
                hard_min_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.success or res.fun < -best_score:
                polish_score = -res.fun
                if polish_score > best_score:
                    best_score = polish_score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass
    
    # Phase 3: Try multiple local optimizations from best points
    if best_points is not None and time.time() < deadline - 20:
        rng2 = np.random.default_rng(999)
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            perturbed = best_points + rng2.normal(0, 0.005 + 0.005*attempt, (n, 2))
            try:
                res = minimize(
                    hard_min_objective,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.success or res.fun < -best_score:
                    s = -res.fun
                    if s > best_score:
                        best_score = s
                        best_points = res.x.reshape(n, 2)
            except Exception:
                pass
    
    # Phase 4: Final refinement with very small steps
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                hard_min_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.success or res.fun < -best_score:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = init_regular_polygon().reshape(n, 2)
    
    # Ensure validity
    if not np.all(np.isfinite(best_points)):
        best_points = init_regular_polygon().reshape(n, 2)
    
    return best_points

# EVOLVE-BLOCK-END
