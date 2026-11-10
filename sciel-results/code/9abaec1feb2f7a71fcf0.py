# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a hybrid strategy: 
    1. Generate candidate "snap points" on the hexagonal lattice.
    2. Use Combinatorial Branch-and-Bound (with pruning) to select 12 points that minimize 
       the bounding hexagon size.
    3. Refine the best combinatorial solution using L-BFGS-B to fine-tune positions.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    sqrt3 = math.sqrt(3)
    N = 12
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: Check if two convex polygons are separated using SAT
    def are_polygons_separated(poly1, poly2):
        # poly1, poly2 are (6, 2) arrays of vertices
        for poly in [poly1, poly2]:
            edges = np.roll(poly, -1, axis=0) - poly
            normals = np.column_stack((edges[:, 1], -edges[:, 0]))
            for n in normals:
                proj1 = poly1 @ n
                proj2 = poly2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                    return True
        return False

    # Pre-compute normals for the outer hexagon (angle 0)
    # Outer hex vertices are at angles 0, 60, ...
    # Edge normals are at 30, 90, ...
    normal_angles = [math.radians(30 + i*60) for i in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in normal_angles])
    
    # Vertex offsets for unit hexagon centered at origin, angle 0
    v_offsets = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def calc_min_s(centers):
        # centers: (N, 2) array
        # For each center, compute max projection of its vertices onto each normal
        # s = (max_proj * 2) / sqrt(3)
        # Actually, apothem = s * sqrt(3)/2. 
        # max_proj <= apothem => s >= max_proj * 2 / sqrt(3)
        max_proj = 0.0
        for c in centers:
            verts = c + v_offsets
            projs = verts @ normals.T
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        return max_proj * 2.0 / sqrt3

    def is_valid(centers, s):
        # Check containment and non-overlap
        # 1. Containment
        apothem = s * sqrt3 / 2
        for c in centers:
            verts = c + v_offsets
            projs = verts @ normals.T
            if np.any(np.max(projs, axis=0) > apothem + 1e-6):
                return False
        
        # 2. Non-overlap
        all_verts = []
        for c in centers:
            all_verts.append(c + v_offsets)
            
        for i in range(N):
            for j in range(i+1, N):
                if not are_polygons_separated(all_verts[i], all_verts[j]):
                    return False
        return True

    # --- Phase 1: Combinatorial Search on Lattice ---
    
    # Generate candidate points on the hexagonal lattice
    # Basis vectors: u = (1.5, sqrt(3)/2), v = (0, sqrt(3))
    # Wait, standard hex grid basis:
    # v1 = (1.5, sqrt(3)/2)
    # v2 = (0.5, sqrt(3)/2) ? No.
    # Let's use:
    # e1 = (1.5, sqrt(3)/2)
    # e2 = (0.5, sqrt(3)/2)
    # Any point is i*e1 + j*e2
    # We want a set of points that likely contain a good packing.
    # The "radius" of the packing is roughly sqrt(12) ~ 3.5.
    # So we need lattice points within a radius of ~4 from origin.
    
    e1 = np.array([1.5, sqrt3/2])
    e2 = np.array([0.5, sqrt3/2])
    
    candidates = []
    max_coord = 4
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            p = i * e1 + j * e2
            if np.linalg.norm(p) < 5.0:
                candidates.append(p)
    
    candidates = np.array(candidates)
    
    # We need to pick 12 points from candidates such that they are non-overlapping.
    # Non-overlap condition for lattice points:
    # Distance between centers must be >= 2 (for angle 0 hexes, min distance is 2? 
    # Actually, for two unit hexes, min distance is 2 if they share an edge? 
    # Center to center distance for touching hexes is 2 * apothem = sqrt(3) ~ 1.732?
    # No. Two hexes touching at a vertex: dist = 2.
    # Two hexes touching at an edge: dist = sqrt(3).
    # So we need dist >= sqrt(3) - epsilon?
    # Let's check: Hex A at (0,0), Hex B at (sqrt(3), 0).
    # Vertices of A: (1,0), (0.5, sqrt3/2)...
    # Vertices of B: (sqrt3+1, 0)...
    # Edge of A is from (0.5, sqrt3/2) to (-0.5, sqrt3/2). Midpoint (0, sqrt3/2).
    # Edge of B is from (sqrt3+0.5, -sqrt3/2) to (sqrt3-0.5, -sqrt3/2). Midpoint (sqrt3, -sqrt3/2).
    # This doesn't seem to touch.
    # Let's use the SAT check to be safe, but for pruning in B&B, we need a fast distance check.
    # Minimum distance between centers of two non-overlapping unit hexes is sqrt(3).
    
    min_dist_sq = 3.0 - 1e-9
    
    # Sort candidates by some heuristic? Or just iterate?
    # To make B&B efficient, we should order candidates.
    # Let's just use a simple recursive search with pruning.
    # State: current list of selected points, current max projection (lower bound on s).
    
    best_s = float('inf')
    best_points = None
    
    # To speed up B&B, we can pre-compute pairwise distances or just check on the fly.
    # Given ~100-200 candidates, B&B might be slow if not pruned well.
    # Pruning: If current_max_proj * 2/sqrt3 >= best_s, prune.
    
    # We also need to ensure we can complete the set of 12.
    # This is a "choose 12 from K" problem with constraints.
    
    # Let's limit the candidate set to a smaller radius to keep search space manageable
    # and rely on the refinement step to fix any minor misalignments.
    # Actually, the lattice points *are* the optimal positions for angle-0 hexes if the outer hex is also angle-0.
    # So if we find a valid set of 12 lattice points, it's a very strong candidate.
    
    def bnb_search(idx, selected, current_max_proj):
        nonlocal best_s, best_points
        
        if time.time() > deadline - 10:
            return
            
        if len(selected) == N:
            s = current_max_proj * 2.0 / sqrt3
            if s < best_s:
                best_s = s
                best_points = selected.copy()
            return
        
        # Pruning: Even if we add points that don't increase projection, 
        # we need to check if remaining points can fit. 
        # Simple bound: current_max_proj * 2/sqrt3 >= best_s -> prune
        if current_max_proj * 2.0 / sqrt3 >= best_s:
            return
            
        # Heuristic: Try candidates that are "central" first?
        # Or just iterate.
        
        for i in range(idx, len(candidates)):
            p = candidates[i]
            
            # Check if p overlaps with any selected point
            overlap = False
            for sp in selected:
                if np.linalg.norm(p - sp) < math.sqrt(min_dist_sq):
                    overlap = True
                    break
            if overlap:
                continue
                
            # Calculate new max projection
            verts = p + v_offsets
            projs = verts @ normals.T
            new_max_p = np.max(projs)
            
            new_current_max_proj = max(current_max_proj, new_max_p)
            
            # Prune if this point alone exceeds best_s bound
            if new_current_max_proj * 2.0 / sqrt3 >= best_s:
                continue
                
            selected.append(p)
            bnb_search(i + 1, selected, new_current_max_proj)
            selected.pop()
            
            # Optional: If we found a very good solution, we could stop early?
            # But let's let it run for a bit.

    # Initialize with a known valid solution to set a good upper bound for B&B
    # Use the lattice 3x4 block
    init_centers = []
    for i in range(3):
        for j in range(4):
            x = j * 2.5
            y = i * sqrt3
            if i % 2 == 1:
                x += 1.25
            init_centers.append([x, y])
    init_centers = np.array(init_centers)
    init_centers -= np.mean(init_centers, axis=0)
    
    # Snap init_centers to lattice? They are already on the lattice.
    # Let's compute their s to initialize best_s
    if is_valid(init_centers, calc_min_s(init_centers)):
        best_s = calc_min_s(init_centers)
        # Find these points in candidates? 
        # The candidates are generated from 0. The init centers are centered.
        # We need to shift candidates or init centers to match.
        # Let's just run B&B from scratch with best_s = inf, but the init provides a fallback.
        # Actually, let's shift the candidates to be centered?
        # No, the lattice is infinite. The "center" of the packing is not necessarily a lattice point.
        # However, the relative positions are lattice points.
        # My B&B assumes the outer hex is centered at (0,0) and axis-aligned.
        # The lattice points I generated are centered at (0,0).
        # The init_centers are also centered at (0,0).
        # So they should be compatible.
        
        # Let's find the exact lattice points for init_centers to set best_points?
        # Not strictly necessary, but helps if B&B is slow.
        
    # Run B&B
    # To make B&B faster, let's sort candidates by distance from origin (prefer central points)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    candidates = candidates[sorted_idx]
    
    bnb_search(0, [], 0.0)
    
    # --- Phase 2: Refinement ---
    
    # If B&B found a solution, use it as start for L-BFGS-B
    # If not, use the init_centers
    if best_points is not None:
        start_centers = np.array(best_points)
    else:
        start_centers = init_centers.copy()
    
    # L-BFGS-B Objective: Minimize s, subject to constraints?
    # Or minimize penalty = s + M * (violations)
    
    def objective(x):
        centers = x.reshape(-1, 2)
        s = calc_min_s(centers)
        
        # Penalty for overlap
        penalty = 0.0
        all_verts = []
        for c in centers:
            all_verts.append(c + v_offsets)
        
        for i in range(N):
            for j in range(i+1, N):
                if not are_polygons_separated(all_verts[i], all_verts[j]):
                    # Calculate penetration depth? 
                    # For simplicity, large penalty if not separated
                    penalty += 100.0
        
        # Penalty for containment
        apothem = s * sqrt3 / 2
        for c in centers:
            verts = c + v_offsets
            projs = verts @ normals.T
            max_p = np.max(projs, axis=0)
            if np.any(max_p > apothem):
                penalty += np.sum(np.maximum(0, max_p - apothem)) * 10.0
                
        return s + penalty

    def gradient(x):
        # Finite difference gradient (expensive but robust)
        # Or analytic?
        # s is piecewise linear (max of linear functions). Gradient is the normal of the active face.
        # Penalty is complex.
        # Let's use numerical gradient via scipy if needed, or just let L-BFGS-B do finite diff.
        # scipy.optimize.minimize with method='L-BFGS-B' uses finite differences by default if jac is None.
        # This is slow but okay for N=24 variables.
        return None

    # Bounds: Keep centers within a reasonable range
    bounds = [(-5, 5)] * (2 * N)
    
    try:
        res = minimize(objective, start_centers.flatten(), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_s:
            final_centers = res.x.reshape(-1, 2)
            # Check validity
            final_s = calc_min_s(final_centers)
            if is_valid(final_centers, final_s):
                best_s = final_s
                best_points = final_centers
    except Exception as e:
        pass

    # Final check and return
    if best_points is None:
        best_points = init_centers
        best_s = calc_min_s(best_points)
    else:
        # Ensure it's valid
        if not is_valid(best_points, best_s):
             # Fallback to init
             best_points = init_centers
             best_s = calc_min_s(best_points)

    inner_ret = np.zeros((12, 3))
    for i in range(12):
        inner_ret[i, 0] = best_points[i, 0]
        inner_ret[i, 1] = best_points[i, 1]
        inner_ret[i, 2] = 0.0
        
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s

# EVOLVE-BLOCK-END
