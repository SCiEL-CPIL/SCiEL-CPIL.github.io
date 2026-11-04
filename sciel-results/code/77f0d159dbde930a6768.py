import math
import time
import numpy as np

N = 11

# Precompute hexagon vertex offsets for unit hexagon at angle 0
# Vertices at angles 0, 60, 120, 180, 240, 300
_HEX_VERTS_0 = np.array([
    [math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] 
    for i in range(6)
])

def _get_cluster_centers():
    """
    Generate the centers for the 11 hexagons in a hexagonal lattice.
    Returns a list of candidate sets of 11 centers.
    """
    d = math.sqrt(3)
    
    # Center
    centers = [[0.0, 0.0]]
    
    # Ring 1: 6 hexagons
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
    
    # Ring 2: 12 positions
    # Angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # Distances:
    # 0, 180: 2*d
    # 90, 270: 2*d
    # 30, 150, 210, 330: 3? No.
    # Let's calculate exactly.
    ring2 = []
    for k in range(12):
        angle = k * math.pi / 6.0
        if k % 2 == 0:
            # 0, 60, 120...
            # These correspond to axial (2,0), (1,1) etc?
            # (2,0) -> (2d, 0)
            # (1,1) -> (1.5d, 1.5) -> dist 3
            # Wait, my previous analysis said:
            # 0 deg: (2d, 0) dist 2d
            # 30 deg: (1.5d, 1.5) dist 3
            # 60 deg: (d, 3) dist 2d? 
            # Let's re-verify (0,2) axial.
            # x = d * (0 + 1) = d. y = 1.5 * 2 = 3.
            # dist = sqrt(d^2 + 9) = sqrt(3+9)=sqrt(12)=2d.
            # So 0, 60, 120, 180, 240, 300 are at distance 2d.
            # 30, 90, 150, 210, 270, 330 are at distance 3.
            dist = 2 * d
        else:
            dist = 3.0
        ring2.append([dist * math.cos(angle), dist * math.sin(angle)])
    
    candidates = []
    
    # Candidate 1: "Cross" - pick 0, 90, 180, 270
    # Indices in ring2 list (0..11) corresponding to angles 0, 30, 60...
    # 0 deg is index 0.
    # 90 deg is index 3.
    # 180 deg is index 6.
    # 270 deg is index 9.
    idx_cross = [0, 3, 6, 9]
    extra1 = [ring2[i] for i in idx_cross]
    candidates.append(centers + extra1)
    
    # Candidate 2: "Diagonal" - pick 0, 60, 180, 240
    # Indices: 0, 2, 6, 8
    idx_diag = [0, 2, 6, 8]
    extra2 = [ring2[i] for i in idx_diag]
    candidates.append(centers + extra2)
    
    # Candidate 3: "Mixed" - pick 0, 90, 180, 330? (Asymmetric)
    # Maybe 0, 150, 180, 330?
    # Let's try 0, 120, 180, 300? (Indices 0, 4, 6, 10)
    idx_mixed = [0, 4, 6, 10]
    extra3 = [ring2[i] for i in idx_mixed]
    candidates.append(centers + extra3)
    
    return candidates

def _compute_min_side_length(centers, angle_rad):
    """
    Compute the minimum side length of the outer hexagon (fixed orientation)
    that contains the rotated cluster.
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate centers
    rotated_centers = np.array(centers) @ R.T
    
    # Outer normals (fixed)
    outer_normals = np.array([
        [math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)] 
        for k in range(6)
    ])
    
    max_extent = 0.0
    for i in range(len(centers)):
        # Vertices of the i-th hexagon
        verts = _HEX_VERTS_0 @ R.T + rotated_centers[i]
        projections = verts @ outer_normals.T
        abs_proj = np.abs(projections)
        max_extent = max(max_extent, np.max(abs_proj))
    
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def construct():
    start_time = time.time()
    
    candidates = _get_cluster_centers()
    
    best_S = float('inf')
    best_centers = None
    best_angle = 0.0
    
    for centers in candidates:
        # Grid search for angle
        # Search 0 to 180 degrees (pi)
        # Step 0.1 degrees
        angles = np.linspace(0, math.pi, 1801)
        S_vals = []
        for a in angles:
            S = _compute_min_side_length(centers, a)
            S_vals.append(S)
            if time.time() - start_time > 100:
                break
        
        S_vals = np.array(S_vals)
        min_idx = np.argmin(S_vals)
        best_local_S = S_vals[min_idx]
        best_local_angle = angles[min_idx]
        
        # Local refinement
        if best_local_S < best_S:
            # Golden section search around best_local_angle
            # Range: +/- 0.1 deg
            a0 = best_local_angle - 0.005
            a1 = best_local_angle + 0.005
            gr = (math.sqrt(5) - 1) / 2
            a2 = a1 - gr * (a1 - a0)
            a3 = a0 + gr * (a1 - a0)
            f2 = _compute_min_side_length(centers, a2)
            f3 = _compute_min_side_length(centers, a3)
            
            for _ in range(50):
                if f2 < f3:
                    a1 = a3
                    a3 = a2
                    f3 = f2
                    a2 = a1 - gr * (a1 - a0)
                    f2 = _compute_min_side_length(centers, a2)
                else:
                    a0 = a2
                    a2 = a3
                    f2 = f3
                    a3 = a0 + gr * (a1 - a0)
                    f3 = _compute_min_side_length(centers, a3)
                
                if abs(a1 - a0) < 1e-6:
                    break
            
            best_local_S = min(f2, f3)
            best_local_angle = (a0 + a1) / 2 if f2 < f3 else (a0 + a1) / 2 # Approx
            
            if best_local_S < best_S:
                best_S = best_local_S
                best_centers = centers
                best_angle = best_local_angle
        
        if time.time() - start_time > 120:
            break
    
    # Fallback if something went wrong
    if best_centers is None:
        best_centers = candidates[0]
        best_angle = 0.0
        best_S = _compute_min_side_length(best_centers, best_angle)
    
    # Format output
    inner = []
    angle_deg = math.degrees(best_angle)
    for c in best_centers:
        inner.append([c[0], c[1], angle_deg])
    
    return inner, (0.0, 0.0), best_S, 0.0

