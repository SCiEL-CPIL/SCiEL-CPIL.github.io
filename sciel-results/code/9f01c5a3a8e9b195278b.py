"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import math

def _hex_vertices_array(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_vertices(positions, angles, side=1.0):
    """Get all vertices of all hexagons as (N*6, 2) array."""
    n = len(positions)
    verts = np.zeros((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices_array(positions[i][0], positions[i][1], side, angles[i])
    return verts

def _min_enclosing_hex_side(positions, angles, outer_center=(0.0, 0.0)):
    """Compute minimum side length of outer hexagon (at 0 deg rotation) 
    centered at outer_center to contain all inner hexagons."""
    verts = _all_vertices(positions, angles, 1.0)
    # Translate so outer center is origin
    verts = verts - np.array(outer_center)
    
    # For a hexagon with side S, rotation 0, centered at origin:
    # A point (x,y) is inside iff for all k=0..5:
    #   x*cos(theta_k) + y*sin(theta_k) <= S * sqrt(3)/2
    # where theta_k = (2k+1)*pi/6
    thetas = np.array([(2*k+1)*np.pi/6 for k in range(6)])
    
    # Compute max projection for each direction
    max_proj = 0.0
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    
    return max_proj / (np.sqrt(3) / 2)

def _check_overlap_pair(p1, a1, p2, a2, side=1.0):
    """Check if two unit hexagons overlap using SAT (Separating Axis Theorem).
    p1, p2 are (x,y) tuples, a1, a2 are angles in degrees."""
    v1 = _hex_vertices_array(p1[0], p1[1], side, a1)
    v2 = _hex_vertices_array(p2[0], p2[1], side, a2)
    
    # Get edges of both hexagons
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    # Normal directions (perpendicular to edges)
    # For a convex polygon, the separating axes are the normals to all edges
    axes = []
    for e in list(edges1) + list(edges2):
        # Perpendicular to edge
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            axes.append(n / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # Check for gap
        if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
            return False
    
    return True

def _check_all_disjoint(positions, angles):
    """Check all pairs are disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _check_overlap_pair(positions[i], angles[i], positions[j], angles[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    sqrt3 = math.sqrt(3)
    
    # Dense hex grid arrangement for N=11
    # Use axial coordinates: (q, r) -> x = 1.5*q, y = sqrt(3)*(r + q/2)
    # This gives touching hexagons with center distance sqrt(3)
    
    # Try multiple compact arrangements and pick the best
    # Arrangement 1: Hexagonal cluster (center + ring1 + part of ring2)
    # Center: (0,0)
    # Ring 1 (6 hexagons): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (12 hexagons): pick 4 more
    
    def axial_to_xy(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    # Generate candidate arrangements
    arrangements = []
    
    # Arrangement A: Center + ring1 + 4 from ring2
    # Ring 2 axial coords: (2,0),(1,1),(0,2),(-1,2),(-2,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1),(2,-2)
    # Wait, let me be more careful. Ring 2: max(|q|,|r|,|q+r|) == 2
    # (2,0): max(2,0,2)=2 ✓
    # (1,1): max(1,1,2)=2 ✓
    # (0,2): max(0,2,2)=2 ✓
    # (-1,2): max(1,2,1)=2 ✓
    # (-2,2): max(2,2,0)=2 ✓
    # (-2,1): max(2,1,1)=2 ✓
    # (-2,0): max(2,0,2)=2 ✓
    # (-1,-1): max(1,1,2)=2 ✓
    # (0,-2): max(0,2,2)=2 ✓
    # (1,-2): max(1,2,1)=2 ✓
    # (2,-1): max(2,1,1)=2 ✓
    # (2,-2): max(2,2,0)=2 ✓
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # Try different subsets of ring2 to get 4 more hexagons
    # For symmetry, try groups of 4 that are roughly balanced
    
    # Option 1: 4 consecutive from ring2
    for start in range(12):
        subset = [ring2[(start + k) % 12] for k in range(4)]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Option 2: Symmetric selections (opposite pairs)
    # Pick 2 opposite pairs from ring2
    for i in range(6):
        for j in range(i+1, 6):
            # ring2[i] and ring2[i+6] are opposite
            # ring2[j] and ring2[j+6] are opposite
            subset = [ring2[i], ring2[i+6], ring2[j], ring2[j+6]]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [axial_to_xy(q, r) for q, r in axial_coords]
            arrangements.append(positions)
    
    # Option 3: 3-4-3-1 row arrangement (more rectangular)
    # Row 1: 3 hexagons
    # Row 2: 4 hexagons  
    # Row 3: 3 hexagons
    # Row 4: 1 hexagon
    # In hex grid, rows are offset
    
    # Let's also try a more rectangular packing
    # 5+3+3 or 4+4+3 etc.
    
    # Try 4-4-3 arrangement
    # Row 0 (y=0): 4 hexagons at x = -1.5, 0, 1.5, 3.0 -> center at 0.75
    # Actually let me just compute several and evaluate
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    
    for arr in arrangements:
        if time.time() > deadline - 5.0:
            break
        angles = [0.0] * N
        # Verify disjointness
        if not _check_all_disjoint(arr, angles):
            continue
        side = _min_enclosing_hex_side(arr, angles)
        if side < best_side:
            best_side = side
            best_positions = arr[:]
            best_angles = angles[:]
    
    # Also try with the outer hexagon rotated by 30 degrees (pointy-top vs flat-top)
    # Actually the outer hexagon rotation is part of the output, so we can try both
    # But _min_enclosing_hex_side assumes 0 deg rotation for outer
    # Let's also compute with outer rotated by 30 deg
    
    best_side_rot30 = float('inf')
    best_positions_rot30 = None
    
    for arr in arrangements:
        if time.time() > deadline - 5.0:
            break
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            continue
        # For outer hexagon rotated by 30 deg, the constraint directions are different
        # theta_k = 30 + (2k+1)*30 = (2k+2)*30 = (k+1)*60 degrees
        # Actually let me just compute by rotating all points by -30 deg and using the 0-deg formula
        # Then the outer side is the same but rotation is 30
        
        # Simpler: compute min enclosing for outer at 30 deg
        # The 6 directions for outer at rotation phi: phi + (2k+1)*pi/6
        # For phi=pi/6: directions are pi/6 + (2k+1)*pi/6 = (2k+2)*pi/6 = (k+1)*pi/3
        thetas_rot = np.array([(k+1)*np.pi/3 for k in range(6)])
        
        verts = _all_vertices(arr, angles, 1.0)
        max_proj = 0.0
        for theta in thetas_rot:
            proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
            mp = np.max(proj)
            if mp > max_proj:
                max_proj = mp
        side_rot = max_proj / (np.sqrt(3) / 2)
        
        if side_rot < best_side_rot30:
            best_side_rot30 = side_rot
            best_positions_rot30 = arr[:]
    
    # Pick the best overall
    if best_side_rot30 < best_side:
        final_positions = best_positions_rot30
        final_angles = [0.0] * N
        final_side = best_side_rot30
        final_outer_rotation = 30.0
    else:
        final_positions = best_positions
        final_angles = best_angles
        final_side = best_side
        final_outer_rotation = 0.0
    
    if final_positions is None:
        # Fallback: linear arrangement
        final_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        final_angles = [0.0] * N
        final_side = _min_enclosing_hex_side(final_positions, final_angles)
        final_outer_rotation = 0.0
    
    # Add tiny epsilon for strict containment
    final_side *= (1.0 + 1e-9)
    
    inner = [[final_positions[i][0], final_positions[i][1], final_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, final_outer_rotation
# EVOLVE-BLOCK-END
