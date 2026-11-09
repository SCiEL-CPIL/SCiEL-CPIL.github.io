import numpy as np
import math
import time
from scipy.optimize import minimize_scalar
from itertools import combinations

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _compute_min_s_for_cluster(centers, inner_angle_deg=0.0):
    """
    Compute the minimum outer side length s for a given set of centers 
    (assumed to be unit hexagons with given inner_angle_deg).
    Optimizes the outer hexagon angle.
    """
    # Get all vertices
    all_verts = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angle_deg)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Convex hull to speed up support function calculation
    # Simple hull implementation or just use all points (72 is small)
    # Using all points is safe and fast enough for 792 * (grid+opt)
    
    def objective(phi_deg):
        phi = np.deg2rad(phi_deg)
        # Normals of the outer hexagon
        # The outer hexagon has side s. The distance from center to edge is a = s * sqrt(3)/2.
        # The condition for a point p to be inside is p . n_k <= a for all k.
        # So a >= max(p . n_k).
        # n_k angles are phi + 30 + 60k
        n_angles = phi + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
        normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
        
        # Project all vertices onto all normals
        # dots shape: (72, 6)
        dots = all_verts @ normals.T
        max_dot = dots.max()
        
        # a = max_dot
        # s = 2 * a / sqrt(3)
        return 2.0 * max_dot / math.sqrt(3.0)

    # Minimize over phi in [0, 60)
    # Use a coarse grid first to find the basin
    best_s = float('inf')
    best_phi = 0.0
    
    # Coarse grid
    for phi in np.linspace(0, 60, 600):
        s_val = objective(phi)
        if s_val < best_s:
            best_s = s_val
            best_phi = phi
            
    # Fine tune with Nelder-Mead or minimize_scalar
    try:
        res = minimize_scalar(objective, bounds=(best_phi - 5, best_phi + 5), method='bounded', options={'xatol': 1e-10})
        if res.success and res.fun < best_s:
            best_s = res.fun
            best_phi = res.x
    except:
        pass
        
    return best_s, best_phi

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0
    
    # Lattice parameters
    d = 2 * math.sqrt(3) # Distance between centers
    a1 = np.array([d, 0.0])
    a2 = np.array([d * math.cos(math.pi/3), d * math.sin(math.pi/3)])
    
    # Generate lattice points in a 3x3 grid around origin (indices -2 to 2)
    # This is enough to cover the 7-point flower and the next ring
    lattice_points = {}
    for u in range(-3, 4):
        for v in range(-3, 4):
            pos = u * a1 + v * a2
            lattice_points[(u, v)] = pos
            
    # Core: 7 points (0,0) and its 6 neighbors
    core_coords = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    core_centers = np.array([lattice_points[c] for c in core_coords])
    
    # Ring 2: 12 points
    # Neighbors of the core that are not in the core
    ring2_coords = []
    for c in core_coords:
        u, v = c
        neighbors = [(u+1, v), (u, v+1), (u-1, v+1), (u-1, v), (u, v-1), (u+1, v-1)]
        for n in neighbors:
            if n not in core_coords and n not in ring2_coords:
                ring2_coords.append(n)
                
    # Ensure we have 12 points in ring 2
    # The logic above should find them. Let's verify count.
    # If not, expand search range.
    if len(ring2_coords) < 12:
        # Fallback: just pick 12 random-ish points from a larger set
        # But for a standard lattice, the 12 neighbors of the 7-point flower are well-defined.
        pass
        
    ring2_centers = np.array([lattice_points[c] for c in ring2_coords])
    
    best_s = float('inf')
    best_config = None
    best_phi = 0.0
    
    # We need to choose 5 points from the 12 in ring2
    # Number of combinations: C(12, 5) = 792
    # This is small. We can evaluate all.
    
    # To ensure symmetry and compactness, we might want to restrict to "connected" or "convex" sets,
    # but evaluating all is safe and fast.
    
    indices = list(range(len(ring2_centers)))
    
    for combo in combinations(indices, 5):
        if time.time() > deadline - 10:
            break
            
        # Construct the 12 centers
        # Core is 7, plus 5 from ring 2
        selected_ring2 = ring2_centers[list(combo)]
        centers = np.vstack([core_centers, selected_ring2])
        
        # Center the cluster to minimize s (WLOG, we can translate the cluster)
        # The outer hexagon is centered at (0,0).
        # We should shift the inner centers so their "centroid" or "center of mass" is at (0,0)?
        # Actually, for minimizing the enclosing hexagon, the cluster should be centered.
        # The "center" of the cluster in terms of the enclosing hexagon is the point that minimizes s.
        # For a fixed shape, the optimal translation is such that the support functions are balanced.
        # However, for a regular hexagon, the optimal center is the centroid of the vertices?
        # Let's assume the cluster is centered at the origin.
        # Since the lattice is symmetric, the 7-point flower is centered at (0,0).
        # Adding 5 points from ring 2 might shift the "center of mass".
        # We should optimize the translation (x, y) as well?
        # No, the outer hexagon is fixed at (0,0).
        # We can translate the inner cluster.
        # Let the translation be T.
        # We want to minimize s(T, phi).
        # This is a 3D optimization (Tx, Ty, phi).
        # Given the symmetry, the optimal T is likely the negative of the centroid of the cluster vertices?
        # Let's just compute the centroid of the 12 centers and shift them so the centroid is at (0,0).
        # This is a good heuristic.
        
        centroid = centers.mean(axis=0)
        centers_shifted = centers - centroid
        
        # Compute min s
        s_val, phi_val = _compute_min_s_for_cluster(centers_shifted, 0.0)
        
        if s_val < best_s:
            best_s = s_val
            best_phi = phi_val
            best_centers = centers_shifted.copy()
            
    # Construct the final result
    # inner: (12, 3)
    # outer: (3,)
    # s: scalar
    
    if best_centers is None:
        # Fallback
        centers = core_centers[:7]
        # Add 5 arbitrary from ring 2
        if len(ring2_centers) >= 5:
            centers = np.vstack([centers, ring2_centers[:5]])
        else:
            # Fill with zeros? No, must be valid.
            # Just use the 7 core + 5 duplicates? No, invalid.
            # Use a simple valid config.
            pass
            
        centroid = centers.mean(axis=0)
        best_centers = centers - centroid
        best_s, best_phi = _compute_min_s_for_cluster(best_centers, 0.0)
        
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = best_centers[i, 0]
        inner[i, 1] = best_centers[i, 1]
        inner[i, 2] = 0.0 # Inner angle
        
    outer = np.array([0.0, 0.0, best_phi])
    
    return inner, outer, best_s

# EVOLVE-BLOCK-END
