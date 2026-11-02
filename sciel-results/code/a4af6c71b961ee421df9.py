"""
Strategy: Symmetric Convex Parameterization + Stochastic Global Search.

The current approach uses unconstrained coordinates and smooth surrogates.
This approach:
1. Enforces convexity by construction: 7 outer points form a convex polygon (the hull).
2. Uses high symmetry: Points are arranged in concentric regular-like polygons.
   - Layer 0: 7 points on the outer hull (radius R0, angles theta_i).
   - Layer 1: 6 points on an inner hexagon (radius R1, angles phi_i).
   - Center: 0 points (to keep total 13).
   Wait, 7+6=13.
   Actually, regular heptagon (7) + regular hexagon (6) is a good start.
   Or 6 outer + 7 inner?
   Let's stick to 7 outer (hull) + 6 inner.
   
   To maximize flexibility while keeping convexity:
   - Outer points: Angles are fixed at regular heptagon angles (or optimized slightly), Radii are equal (R0).
   - Inner points: Angles are fixed at regular hexagon angles (or optimized), Radii are equal (R1).
   - Rotation: Relative rotation between layers.
   - Scaling: Overall scale doesn't matter due to normalization, but relative radii do.
   
   Parameters to optimize:
   - r: Ratio of inner radius to outer radius (0 < r < 1).
   - alpha: Angular offset of inner hexagon relative to outer heptagon.
   - (Optional) Slight perturbations to angles to break perfect symmetry if needed, 
     but symmetry is often optimal for these problems. Let's start with pure symmetry parameters.
   
   If pure symmetry is too restrictive, we can add "wobble" parameters.
   However, for N=13, the "Heilbronn number" is often achieved by configurations close to regular polygons.
   
   Let's use a more flexible parameterization:
   - 7 outer points: Angles are regular heptagon angles + small noise? No, let's keep them fixed to ensure convexity and stability.
   - 6 inner points: Angles are regular hexagon angles + rotation.
   - Radii: Outer = 1.0, Inner = r.
   
   This reduces the problem to 2 parameters (r, alpha). We can scan these densely.
   Then, we can use local optimization on the full coordinates starting from the best symmetric config, 
   but constrained to remain convex (or just check hull area).
   
   Actually, the prompt says "normalized by the point set convex hull area".
   If we place 7 points on the convex hull, the hull area is the area of that heptagon.
   The inner points do not change the hull area.
   So we just need to maximize min_triangle_area / area_heptagon.
   
   Let's try a hybrid:
   1. Global search over symmetric parameters (r, alpha) to find a good starting point.
   2. Local Nelder-Mead on full coordinates, but with a penalty if convexity is lost?
      No, better: Use the symmetric config as the seed for a local search that allows slight deviations.
      
   Key Insight: The optimal configuration for small N is often highly symmetric.
   For N=13, a regular 13-gon is a candidate, but we found that concentric polygons might be better.
   Let's test:
   - Config A: Regular 13-gon.
   - Config B: 7-outer (heptagon) + 6-inner (hexagon).
   - Config C: 6-outer (hexagon) + 6-inner (hexagon) + 1-center.
   
   I will implement a search over these symmetric families and then polish.
"""

import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)
NUM_TRI = len(TRI_IDX)

def min_triangle_area(points):
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score(points):
    a_min = min_triangle_area(points)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # --- Phase 1: Symmetric Global Search ---
    # We explore different symmetric topologies.
    
    # Topology 1: Regular 13-gon
    angles_13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts_13 = np.column_stack([np.cos(angles_13), np.sin(angles_13)])
    update_best(pts_13)
    
    # Topology 2: 7-outer (Heptagon) + 6-inner (Hexagon)
    # Parameters: r (inner radius), alpha (rotation of inner)
    # Outer angles: fixed regular heptagon
    angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    # Inner angles: regular hexagon + alpha
    
    # We will scan r and alpha
    r_values = np.linspace(0.1, 0.9, 20)
    alpha_values = np.linspace(0, 2 * np.pi / 6, 20) # Hexagon symmetry period is 60 deg
    
    for r in r_values:
        if time.time() > deadline - 20.0:
            break
        for alpha in alpha_values:
            if time.time() > deadline - 20.0:
                break
            
            outer = np.column_stack([np.cos(angles_7), np.sin(angles_7)])
            inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + alpha
            inner = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            
            # Interleave or just concatenate?
            # The indices don't matter for the set, but for "convex hull", 
            # we need to ensure the outer points are indeed the hull.
            # Since r < 1, the heptagon is the hull.
            pts = np.vstack([outer, inner])
            update_best(pts)
            
            # Also try with a slight rotation of the outer polygon
            # The relative angle between 7-gon and 6-gon is what matters.
            # Since 7 and 6 are coprime, the pattern repeats.
            # But rotating the outer by small amounts might help if the optimum isn't at alpha=0.
            # Actually, rotating outer by d is equivalent to rotating inner by -d.
            # So the (r, alpha) scan covers all relative rotations.

    # Topology 3: 6-outer (Hexagon) + 6-inner (Hexagon) + 1-center
    # Parameters: r_inner, alpha_inner
    angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    center = np.array([[0.0, 0.0]])
    
    r_values_6 = np.linspace(0.2, 0.8, 15)
    alpha_values_6 = np.linspace(0, np.pi / 6, 15) # 30 deg period for relative rotation
    
    for r in r_values_6:
        if time.time() > deadline - 10.0:
            break
        for alpha in alpha_values_6:
            if time.time() > deadline - 10.0:
                break
            
            outer = np.column_stack([np.cos(angles_6), np.sin(angles_6)])
            inner_angles = angles_6 + alpha
            inner = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            
            pts = np.vstack([outer, inner, center])
            update_best(pts)

    # Topology 4: 5-outer (Pentagon) + 8-inner?
    # 5 is prime, might be less symmetric. Let's skip for now to save time.
    # 4-outer (Square) + 9-inner?
    # Let's stick to the most promising ones.
    
    # --- Phase 2: Local Optimization from Best Symmetric Config ---
    # Use Nelder-Mead on the full coordinates, but start from the best symmetric config.
    # This allows the points to "breathe" from the perfect symmetry if needed.
    
    if best_points is not None:
        if time.time() < deadline - 5.0:
            x0 = best_points.flatten()
            
            def objective(x):
                pts = x.reshape(-1, 2)
                return -score(pts)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                pts = res.x.reshape(-1, 2)
                update_best(pts)
            except Exception:
                pass
            
            # Try a second polish with tighter tolerances
            if time.time() < deadline - 2.0 and best_points is not None:
                x0 = best_points.flatten()
                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                    pts = res.x.reshape(-1, 2)
                    update_best(pts)
                except Exception:
                    pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
