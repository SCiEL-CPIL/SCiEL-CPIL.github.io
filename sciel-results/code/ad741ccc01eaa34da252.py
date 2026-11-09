"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    # Precompute vertex offsets for a unit flat-top hexagon
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    vertex_offsets = []
    for k in range(6):
        angle = math.radians(60 * k)
        vertex_offsets.append((math.cos(angle), math.sin(angle)))
    
    # Precompute normal vectors for the outer hexagon constraints
    # Outer hexagon is flat-top. Normals to edges are at 30, 90, 150, 210, 270, 330 degrees
    normal_vectors = []
    for j in range(6):
        angle = math.radians(60 * j + 30)
        normal_vectors.append((math.cos(angle), math.sin(angle)))
        
    inradius_factor = sqrt3 / 2.0 # inradius = L * sqrt(3)/2

    def required_L_fast(centers_subset):
        """Find minimum L such that all unit hexagons at given centers fit inside outer hex of side L."""
        min_L = 0.0
        for (cx, cy) in centers_subset:
            for vx_off, vy_off in vertex_offsets:
                vx = cx + vx_off
                vy = cy + vy_off
                # Find maximum projection onto any of the 6 normals
                max_proj = 0.0
                for nx, ny in normal_vectors:
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
                # L * inradius_factor >= max_proj  => L >= max_proj / inradius_factor
                L_needed = max_proj / inradius_factor
                if L_needed > min_L:
                    min_L = L_needed
        return min_L
    
    # Try all subsets of size 12 from 19 centers
    # C(19,12) = 50388
    best_L = float('inf')
    best_subset = None
    
    # To ensure disjointness (interiors do not overlap), grid placement is sufficient.
    # We do not apply any scale factor, assuming touching boundaries are allowed.
    # If strict gap is needed, we could add a tiny epsilon, but usually "disjoint"
    # in packing problems allows touching.
    
    for subset in combinations(range(19), 12):
        if time.time() > deadline:
            break
        
        # Extract centers
        subset_centers = [all_centers[i] for i in subset]
        
        L = required_L_fast(subset_centers)
        if L < best_L:
            best_L = L
            best_subset = subset
    
    if best_subset is None:
        # Fallback: use first 12
        best_subset = tuple(range(12))
        subset_centers = [all_centers[i] for i in best_subset]
        best_L = required_L_fast(subset_centers)
    
    # Construct inner hexagons
    # angle is 0.0 for flat-top orientation
    inner = []
    for i in best_subset:
        x, y = all_centers[i]
        inner.append([x, y, 0.0])
    
    # Outer hexagon center is (0,0), side length best_L, rotation 0 (flat-top)
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
