# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a deterministic 3x4 hexagonal grid packing and finds the minimal enclosing regular hexagon.
    """
    # 1. Generate 12 unit hexagon centers in a compact 3x4 grid
    # Basis vectors for edge-sharing hexagons (side 1)
    # Distance between centers is sqrt(3)
    # Directions: 30, 90, 150 degrees (and opposites)
    d = np.sqrt(3.0)
    u = d * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    v = d * np.array([np.cos(np.radians(90)), np.sin(np.radians(90))])
    
    centers = []
    for i in range(3):
        for j in range(4):
            centers.append(i * u + j * v)
    centers = np.array(centers)
    
    # 2. Find optimal outer hexagon parameters
    # We want to minimize s.
    # s is determined by the max support function of the cluster of hexagons
    # against the 6 normals of the outer hexagon.
    
    # Pre-compute vertices for all 12 inner hexagons (angle 0)
    inner_verts_list = []
    for c in centers:
        inner_verts_list.append(_hex_vertices(c[0], c[1], 1.0, 0.0))
    all_inner_verts = np.vstack(inner_verts_list)
    
    # Shift centers to origin to find optimal orientation/center
    # The optimal outer center is the centroid of the cluster? 
    # Not necessarily, but for a symmetric cluster it is.
    # Let's assume outer center is the centroid of the inner centers for now, 
    # and optimize the angle.
    centroid = np.mean(centers, axis=0)
    shifted_verts = all_inner_verts - centroid
    
    best_s = np.inf
    best_angle = 0.0
    
    # Try a range of angles for the outer hexagon
    # Due to symmetry, we only need to check 0 to 30 degrees
    for angle_deg in np.linspace(0, 30, 61):
        a = np.radians(angle_deg)
        # Normals of the outer hexagon edges
        # Vertices are at a + 60k. Edges are between them.
        # Normal to edge k is at angle a + 60k + 30? 
        # Let's derive:
        # Vertices v_k = (cos(a+60k), sin(a+60k)).
        # Edge vector e_k = v_{k+1} - v_k.
        # Normal n_k is perpendicular to e_k, pointing out.
        # For a regular polygon, normals are at angles a + 60k + 30? 
        # Let's check k=0. v0=(cos a, sin a), v1=(cos(a+60), sin(a+60)).
        # Midpoint of edge is at angle a+30.
        # So normal is at angle a+30.
        normal_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Support function: max dot product of shifted vertices with each normal
        supports = np.max(shifted_verts @ normals.T, axis=0)
        
        # The apothem of the outer hexagon must be at least the max support
        apothem = np.max(supports)
        
        # Side length s = 2 * apothem / sqrt(3)
        s = 2.0 * apothem / np.sqrt(3.0)
        
        if s < best_s:
            best_s = s
            best_angle = angle_deg
            
    # 3. Construct final result
    # Inner data: (x, y, angle)
    inner_data = np.column_stack([
        centers[:, 0],
        centers[:, 1],
        np.zeros(12)
    ])
    
    # Outer data: (x, y, angle)
    outer_data = np.array([centroid[0], centroid[1], best_angle])
    
    # Ensure s is at least sqrt(12) (though it will be much larger)
    if best_s < np.sqrt(12.0):
        best_s = np.sqrt(12.0)
        
    return inner_data, outer_data, best_s

# EVOLVE-BLOCK-END
