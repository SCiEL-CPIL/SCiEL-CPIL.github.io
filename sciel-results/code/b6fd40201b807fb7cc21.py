import numpy as np
import time
from scipy.optimize import linprog

def hexagon_packing_12():
    # Start time for safety
    start_time = time.time()
    
    # Lattice vectors for hexagons of side 1
    # Distance between centers of touching hexes is sqrt(3)
    # To ensure separation > 1e-6, we scale the lattice spacing slightly
    gap = 1e-6
    scale = 1.0 + gap / np.sqrt(3)
    d = np.sqrt(3) * scale
    
    e1 = np.array([d, 0])
    e2 = np.array([d/2, d * np.sqrt(3)/2])
    
    # Define 12 lattice coordinates (q, r)
    # Configuration: 
    # r=0: q=-2, -1, 0, 1, 2 (5)
    # r=1: q=-1, 0, 1 (3)
    # r=-1: q=-1, 0, 1 (3)
    # r=2: q=0 (1)
    # Total 12
    coords = [
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1),
        (0, 2)
    ]
    
    centers = np.array([q * e1 + r * e2 for q, r in coords])
    
    # Center the cluster at origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Inner hexagon angle
    # Knowledge says 30 degrees (flat top or pointy top? 
    # Vertices at 0, 60, ... implies pointy top if 0 is right.
    # Standard hexagon usually has flat top if angle is 0?
    # The problem statement: vertices (cx + d cos(a + 60i), ...).
    # If a=0, vertex 0 is at (cx+d, cy). This is "pointy right".
    # Let's stick with 0 or 30. It doesn't matter for the lattice as long as it's consistent.
    # The previous code used 30.
    angle_deg = 30.0
    inner_angles = np.full(12, angle_deg)
    
    # Get all vertices of all inner hexagons
    def get_vertices(cx, cy, ang_deg):
        ang = np.radians(ang_deg)
        verts = []
        for k in range(6):
            theta = ang + k * np.pi / 3
            verts.append([cx + np.cos(theta), cy + np.sin(theta)])
        return np.array(verts)
    
    all_verts = np.vstack([get_vertices(cx, cy, angle_deg) for cx, cy in centers])
    
    # Find optimal outer hexagon
    best_s = np.inf
    best_phi = 0
    best_center = np.zeros(2)
    
    # Scan angles
    # 60 degrees is the period. 200 steps is enough for resolution.
    for phi_deg in np.linspace(0, 60, 201):
        if time.time() - start_time > 550:
            break
            
        phi = np.radians(phi_deg)
        # Normals for the 3 independent directions
        # The hexagon edges are perpendicular to normals.
        # Normals are at angles phi, phi+60, phi+120.
        normals = np.array([
            [np.cos(phi), np.sin(phi)],
            [np.cos(phi + np.pi/3), np.sin(phi + np.pi/3)],
            [np.cos(phi + 2*np.pi/3), np.sin(phi + 2*np.pi/3)]
        ])
        
        # Project all vertices onto these 3 normals
        projs = all_verts @ normals.T
        min_projs = np.min(projs, axis=0)
        max_projs = np.max(projs, axis=0)
        spans = max_projs - min_projs
        
        # Required side length s such that apothem a = s*sqrt(3)/2 >= span/2
        s_candidate = np.max(spans) / np.sqrt(3)
        
        if s_candidate >= best_s:
            continue
            
        # We need to find a center c = (cx, cy) such that:
        # For each k in 0,1,2:
        # min_k + s*sqrt(3)/2 >= c . n_k >= max_k - s*sqrt(3)/2
        # Let a_k = s_candidate * np.sqrt(3) / 2
        # Inequalities:
        # -c . n_k <= min_k + a_k  =>  n_k . (-c) <= min_k + a_k
        # c . n_k <= max_k - a_k   =>  -n_k . (-c) <= max_k - a_k  (Wait, signs)
        
        # Let x = -c.
        # Constraint 1: n_k . x <= min_k + a_k
        # Constraint 2: -n_k . x <= max_k - a_k  => n_k . x >= a_k - max_k
        
        # So: a_k - max_k <= n_k . x <= min_k + a_k
        
        A_ub = np.vstack([normals, -normals])
        b_ub = np.concatenate([min_projs + s_candidate * np.sqrt(3)/2, 
                               s_candidate * np.sqrt(3)/2 - max_projs])
        
        # Minimize 0
        res = linprog(c=np.zeros(2), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)], method='highs')
        
        if res.status == 0:
            c = -res.x
            # Verify containment to be absolutely sure
            # Check all 6 normals
            normals_full = np.array([
                [np.cos(phi + k * np.pi/3), np.sin(phi + k * np.pi/3)] for k in range(6)
            ])
            # Distance from center to edge is s*sqrt(3)/2
            # Point v is inside if n . (v - c) <= s*sqrt(3)/2
            diffs = (all_verts - c) @ normals_full.T
            if np.max(diffs) <= s_candidate * np.sqrt(3)/2 + 1e-9:
                best_s = s_candidate
                best_phi = phi
                best_center = c
    
    # Construct output
    inner = np.column_stack([centers, inner_angles])
    outer = np.array([best_center[0], best_center[1], np.degrees(best_phi)])
    s = best_s
    
    return inner, outer, s
