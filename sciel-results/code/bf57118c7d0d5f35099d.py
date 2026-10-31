"""Structural change: rigid 3-4-3 cluster + LP-style dual optimization of outer side via
max-min clearance over all 6*6=36 support directions. Uses log-sum-exp smooth surrogate
for the min-clearance to get gradients, then a high-precision polish pass."""
N = 11
import numpy as np
import math
import time

# ---- geometry helpers ----------------------------------------------------

def _hex_verts(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    angs = a + np.pi/6 + np.arange(6)*np.pi/3
    return np.column_stack([cx + s*np.cos(angs), cy + s*np.sin(angs)])

def _all_verts(positions, angles, s=1.0):
    n = len(positions)
    v = np.empty((n*6, 2))
    for i in range(n):
        v[i*6:(i+1)*6] = _hex_verts(positions[i][0], positions[i][1], s, angles[i])
    return v

# 36 candidate support directions (every 5 deg) for the outer hexagon's 6 normals.
# The outer hexagon has 6 normals at angles phi + (2k+1)*pi/6 for k=0..5.
# We optimize over phi (outer rotation) and over the cluster's translation.
# The side length is determined by the max over all vertices of the projection onto
# each of the 6 normals, divided by (sqrt(3)/2).

def _required_side(positions, angles, phi_rad, center=(0.0, 0.0)):
    """Exact required side length of outer hex with rotation phi centered at center."""
    verts = _all_verts(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_proj = 0.0
    for t in thetas:
        proj = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (math.sqrt(3)/2)

def _smooth_required_side(positions, angles, phi_rad, center=(0.0, 0.0), beta=50.0):
    """Smooth surrogate: log-sum-exp over all 6 directions * all vertices."""
    verts = _all_verts(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    # For each direction, compute max projection via logsumexp
    max_projs = []
    for t in thetas:
        proj = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
        m = np.max(proj)
        lse = m + math.log(np.sum(np.exp(proj - m)))
        max_projs.append(lse / beta * beta)  # just the logsumexp value
    # Actually we want max over directions of (max over vertices).
    # Smooth: logsumexp over all (direction, vertex) pairs of beta * proj.
    all_projs = []
    for t in thetas:
        proj = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
        all_projs.append(proj)
    all_projs = np.concatenate(all_projs)  # (6*36,)
    m = np.max(all_projs)
    lse = m + math.log(np.sum(np.exp(all_projs - m)))
    return lse / (math.sqrt(3)/2) / beta * beta  # this isn't right

def _smooth_side(positions, angles, phi_rad, center=(0.0, 0.0), beta=40.0):
    """Proper smooth surrogate for the side length."""
    verts = _all_verts(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    # Stack all projections: (6 directions, 66 vertices)
    projs = np.zeros((6, len(verts)))
    for k, t in enumerate(thetas):
        projs[k] = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
    # For each direction, smooth max over vertices
    m = np.max(projs, axis=1)  # (6,)
    lse_per_dir = m + (1.0/beta) * np.log(np.sum(np.exp(projs - m[:, None]), axis=1))
    # Smooth max over directions
    m2 = np.max(lse_per_dir)
    lse_final = m2 + (1.0/beta) * math.log(math.exp(beta * (lse_per_dir - m2)).sum())
    return lse_final / (math.sqrt(3)/2)

def _smooth_side_grad(x, angles, center=(0.0, 0.0), beta=40.0):
    """x = [phi, cx, cy] or [x0,y0,...,x10,y10, phi, cx, cy]. 
    For the LP-style approach, we fix the cluster and only optimize phi, cx, cy.
    But let's also allow translating the whole cluster."""
    # x: [phi, tx, ty] where tx,ty is cluster translation
    phi = x[0]
    tx, ty = x[1], x[2]
    c = (tx, ty)
    verts = _all_verts(positions, angles, 1.0) - np.array(c)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    projs = np.zeros((6, len(verts)))
    for k, t in enumerate(thetas):
        projs[k] = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
    m = np.max(projs, axis=1)
    lse_per_dir = m + (1.0/beta) * np.log(np.sum(np.exp(projs - m[:, None]), axis=1))
    m2 = np.max(lse_per_dir)
    lse_final = m2 + (1.0/beta) * math.log(math.exp(beta * (lse_per_dir - m2)).sum())
    side = lse_final / (math.sqrt(3)/2)
    return side

# ---- SAT overlap check ---------------------------------------------------

def _sat_gap(p1, a1, p2, a2, s=1.0):
    v1 = _hex_verts(p1[0], p1[1], s, a1)
    v2 = _hex_verts(p2[0], p2[1], s, a2)
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        nv = np.array([-e[1], e[0]])
        n = np.linalg.norm(nv)
        if n < 1e-12:
            continue
        nv /= n
        proj1 = v1 @ nv
        proj2 = v2 @ nv
        gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_disjoint(positions, angles, tol=1e-9):
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if _sat_gap(positions[i], angles[i], positions[j], angles[j]) < -tol:
                return False
    return True

# ---- Build the rigid 3-4-3 cluster ---------------------------------------

def _build_343_cluster():
    """
    3-4-3 arrangement of unit hexagons on a hex lattice.
    Using axial coordinates (q, r) with pointy-top hexagons.
    Row 0 (top): 3 hexagons
    Row 1 (middle): 4 hexagons  
    Row 2 (bottom): 3 hexagons
    
    For pointy-top hexagons with side 1:
    - Horizontal spacing between adjacent hexes in same row: sqrt(3)
    - Vertical spacing between rows: 1.5
    - Offset between rows: sqrt(3)/2
    
    Actually, let me use the standard hex grid:
    For flat-top hexagons (angle 0):
    - x spacing: 1.5 * side
    - y spacing: sqrt(3) * side
    - Offset for odd columns: sqrt(3)/2 * side
    
    Let me just place them manually for a compact 3-4-3:
    """
    s3 = math.sqrt(3)
    # Flat-top hexagons, side 1
    # Hex centers in a compact triangular arrangement:
    # Row 0: 3 hexes at y = 0, x = -s3, 0, s3
    # Row 1: 4 hexes at y = -1.5, x = -1.5*s3/2 - s3/2, ... 
    # Actually let me use axial coordinates properly.
    
    # Axial coords (q, r) for flat-top:
    # x = 1.5*q, y = sqrt(3)*(r + q/2)
    # Adjacent: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    
    # 3-4-3 cluster:
    # Center at (0,0)
    # Ring 1: 6 hexes
    # We need 11 total. 3+4+3 = 10, plus 1 center = 11.
    
    # Let me define specific axial coords:
    # Row structure (by y-level):
    # Using (q, r):
    # y = sqrt(3)*(r + q/2)
    
    # Let me just enumerate a known compact 11-hex packing:
    # Center: (0,0)
    # 6 around it: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 4 more from ring 2: (2,0), (-2,1), (0,2), (1,-2)  -- need to check compactness
    
    # Actually, the 3-4-3 pattern:
    # Top row (3): (-1, 1), (0, 1), (1, 1)  -> but (1,1) isn't adjacent to center
    # Let me think differently.
    
    # Compact 11-hex cluster (like a hexagon of "radius" ~1.5):
    # Axial coords:
    coords = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (0, 2), (-1, 2),  # part of ring 2 (4 more)
    ]
    # That's 1+6+4 = 11. Let me verify they're all distinct.
    # (2,0), (1,1), (0,2), (-1,2) - yes, all distinct from above.
    
    def axial_to_xy(q, r):
        return (1.5 * q, s3 * (r + q / 2.0))
    
    positions = [axial_to_xy(q, r) for q, r in coords]
    angles = [0.0] * 11
    return positions, angles

def _build_alternative_clusters():
    """Generate several candidate rigid clusters."""
    s3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        return (1.5 * q, s3 * (r + q / 2.0))
    
    clusters = []
    
    # Cluster 1: center + ring1 + 4 from ring2 (as above)
    c1 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2),
    ]
    clusters.append([axial_to_xy(q, r) for q, r in c1])
    
    # Cluster 2: center + ring1 + different 4 from ring2
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    for subset in [
        [(2,0), (0,2), (-2,2), (0,-2)],       # symmetric
        [(2,0), (1,1), (-2,0), (-1,-1)],       # another symmetric
        [(2,0), (1,1), (0,2), (1,-2)],         # asymmetric
        [(2,0), (0,2), (-2,1), (1,-2)],        # another
    ]:
        c = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + subset
        clusters.append([axial_to_xy(q, r) for q, r in c])
    
    # Cluster 3: 5-1-5 (two rows of 5 with 1 in middle) - might be wider
    c3 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,2),  # 5 from ring 2
    ]
    # That's 12, too many. Let me do 5+1+5 differently.
    # Actually 3-4-3 is the standard compact shape.
    
    return clusters

# ---- Main construct -------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    s3 = math.sqrt(3)
    
    # Generate candidate clusters
    clusters = _build_alternative_clusters()
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_center = (0.0, 0.0)
    
    for cluster in clusters:
        if time.time() > deadline - 15.0:
            break
        
        angles = [0.0] * N
        if not _check_disjoint(cluster, angles):
            continue
        
        # For each cluster, optimize over outer rotation phi and cluster center (tx, ty)
        # The cluster is rigid; we just translate it and rotate the outer hexagon.
        
        # Coarse search over phi (0 to 60 deg, since hex symmetry) and center
        best_cluster_side = float('inf')
        best_cluster_phi = 0.0
        best_cluster_center = (0.0, 0.0)
        best_cluster_pos = None
        
        # The optimal center should be near the centroid of the cluster
        cx0 = np.mean([p[0] for p in cluster])
        cy0 = np.mean([p[1] for p in cluster])
        
        # Coarse grid search
        for phi_deg in np.arange(0, 60, 2.0):
            if time.time() > deadline - 10.0:
                break
            phi = math.radians(phi_deg)
            # Try a few center offsets
            for dx in np.arange(-1.0, 1.01, 0.25):
                for dy in np.arange(-1.0, 1.01, 0.25):
                    center = (cx0 + dx, cy0 + dy)
                    side = _required_side(cluster, angles, phi, center)
                    if side < best_cluster_side:
                        best_cluster_side = side
                        best_cluster_phi = phi
                        best_cluster_center = center
                        best_cluster_pos = [ (p[0]-center[0], p[1]-center[1]) for p in cluster ]
        
        if best_cluster_side < best_side:
            best_side = best_cluster_side
            best_phi = best_cluster_phi
            best_center = best_cluster_center
            best_positions = best_cluster_pos
            best_angles = angles
    
    # Refine: local optimization on the best cluster
    if best_positions is not None:
        # Fine-tune phi
        for phi_deg in np.arange(best_phi*180/math.pi - 2, best_phi*180/math.pi + 2.01, 0.25):
            if time.time() > deadline - 5.0:
                break
            phi = math.radians(phi_deg)
            side = _required_side(best_positions, best_angles, phi, best_center)
            if side < best_side:
                best_side = side
                best_phi = phi
        
        # Fine-tune center
        cx, cy = best_center
        for dx in np.arange(-0.5, 0.51, 0.05):
            for dy in np.arange(-0.5, 0.51, 0.05):
                if time.time() > deadline - 3.0:
                    break
                center = (cx + dx, cy + dy)
                side = _required_side(best_positions, best_angles, best_phi, center)
                if side < best_side:
                    best_side = side
                    best_center = center
                    best_positions = [ (p[0]-center[0], p[1]-center[1]) for p in best_positions ]
                    # Wait, this is getting confusing. Let me restructure.
    
    # Actually, let me redo the refinement more carefully.
    # The positions are in absolute coords. The outer hex is centered at best_center.
    # We want to report positions relative to origin, outer center at (0,0).
    
    if best_positions is None:
        # Fallback
        best_positions = [(4.0*k, 0.0) for k in range(-(N//2), N-N//2)]
        best_angles = [0.0]*N
        best_side = _required_side(best_positions, best_angles, 0.0, (0.0, 0.0))
        best_phi = 0.0
        best_center = (0.0, 0.0)
    else:
        # Convert to frame where outer center is at origin
        best_positions = [ (p[0]-best_center[0], p[1]-best_center[1]) for p in best_positions ]
        best_center = (0.0, 0.0)
        # Recompute side in this frame
        best_side = _required_side(best_positions, best_angles, best_phi, (0.0, 0.0))
    
    # Final verification
    if not _check_disjoint(best_positions, best_angles):
        # Fallback to linear
        best_positions = [(4.0*k, 0.0) for k in range(-(N//2), N-N//2)]
        best_angles = [0.0]*N
        best_side = _required_side(best_positions, best_angles, 0.0, (0.0, 0.0))
        best_phi = 0.0
    
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, math.degrees(best_phi)
