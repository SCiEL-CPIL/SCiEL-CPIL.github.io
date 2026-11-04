"""
Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.
Structural change: treat the 12 hexagons as a rigid honeycomb cluster.
Optimize the cluster's global orientation and position via a fine grid search
over rotation angles and translations to minimize the bounding hexagon's side
length via direct geometric projection.
"""
import time
import math
import numpy as np

N = 12
DEADLINE = 115.0  # safety margin before the 120s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices_local(theta):
    """Vertices of unit regular hexagon (flat-top) at origin, rotated by theta."""
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _norms(theta):
    """6 outward edge-normal directions of a regular hexagon oriented by theta."""
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _bounding_side(pts, theta, center):
    """Side length of smallest regular hexagon (orientation theta, center) containing pts."""
    nrm = _norms(theta)
    P = pts @ nrm.T
    c = center @ nrm.T
    req = (P - c).max(axis=0)
    return 2.0 * float(np.max(req)) / math.sqrt(3.0)

# ---------------------------------------------------------------------------
# Rigid honeycomb cluster construction
# ---------------------------------------------------------------------------
def _build_honeycomb_cluster():
    """
    Build a dense honeycomb cluster of 12 unit hexagons.
    Returns:
      - centers: (N, 2) array of local center positions
      - angles: (N,) array of local rotation angles (radians), all 0 for aligned hexes
    """
    # Honeycomb lattice: hexagons tile with centers separated by 2*sqrt(3) in axial directions
    # For flat-top hexagons, the lattice vectors are:
    #   a1 = (3, sqrt(3))  -> no, let's use the standard honeycomb center spacing
    # For unit side hexagons, the distance between centers of adjacent hexagons is 2*sqrt(3)/2 * 2 = 2*sqrt(3)?
    # Actually, for regular hexagons of side s, the distance between centers of adjacent hexagons
    # in a honeycomb tiling is 2 * apothem = 2 * (s * sqrt(3)/2) = s * sqrt(3).
    # Wait: apothem = s * sqrt(3)/2. The distance between parallel edges of adjacent hexagons is 2*apothem = s*sqrt(3).
    # So center-to-center distance for adjacent hexagons is s * sqrt(3) = sqrt(3) for s=1.
    
    # Let's use axial coordinates for a honeycomb.
    # Hexagon centers in a honeycomb can be indexed by (q, r) in axial coordinates.
    # The position is: x = sqrt(3) * (q + r/2), y = 3/2 * r  (for pointy-top)
    # Or for flat-top: x = 3/2 * q, y = sqrt(3) * (r + q/2)
    
    # Let's use a simple approach: place hexagons in a compact cluster.
    # A 12-hexagon cluster can be:
    #   - A central hexagon surrounded by 6, plus 5 more on the outside
    #   - Or a 4x3 rectangular-ish arrangement
    
    # Let's use the "flower" pattern: 1 center + 6 ring + 5 more
    # Actually, for a rigid cluster, let's use a hexagonal arrangement:
    # Ring 0: 1 hex at (0,0)
    # Ring 1: 6 hexes at distance d = sqrt(3) from center
    # Ring 2: 12 hexes at distance 2*sqrt(3) from center, but we only need 5 more
    
    d = math.sqrt(3.0)  # center-to-center distance for adjacent hexagons
    
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = k * math.pi / 3.0
        centers.append([d * math.cos(ang), d * math.sin(ang)])
        angles.append(0.0)
    
    # Ring 2: 12 positions, take 5
    for k in range(12):
        ang = k * math.pi / 6.0 + math.pi / 6.0  # offset by 30 degrees
        centers.append([2 * d * math.cos(ang), 2 * d * math.sin(ang)])
        angles.append(0.0)
        if len(centers) == N:
            break
    
    return np.array(centers), np.array(angles)

# ---------------------------------------------------------------------------
# Rigid cluster evaluation
# ---------------------------------------------------------------------------
def _evaluate_rigid_cluster(centers_local, angles_local, cluster_angle, tx, ty, outer_angle, outer_center):
    """
    Evaluate the bounding side for a rigid cluster rotated by cluster_angle,
    translated by (tx, ty), and enclosed in an outer hexagon with given orientation.
    
    Returns the outer side length.
    """
    # Transform cluster
    ca = math.cos(cluster_angle)
    sa = math.sin(cluster_angle)
    
    # Rotate local centers
    rotated_centers = np.zeros_like(centers_local)
    rotated_centers[:, 0] = ca * centers_local[:, 0] - sa * centers_local[:, 1] + tx
    rotated_centers[:, 1] = sa * centers_local[:, 0] + ca * centers_local[:, 1] + ty
    
    # Compute all vertices
    all_pts = []
    for k in range(N):
        # Local vertices rotated by cluster_angle + local angle
        local_verts = _hex_vertices_local(cluster_angle + angles_local[k])
        all_pts.append(local_verts + rotated_centers[k])
    
    all_pts = np.vstack(all_pts)
    
    return _bounding_side(all_pts, outer_angle, np.array(outer_center))

# ---------------------------------------------------------------------------
# Grid search over cluster orientation and position
# ---------------------------------------------------------------------------
def _grid_search_rigid(centers_local, angles_local, time_limit):
    """
    Perform a grid search over cluster rotation, translation, and outer hexagon orientation.
    """
    start = time.time()
    
    best_side = float('inf')
    best_params = None
    
    # Precompute: for each cluster angle, the cluster is rotated.
    # The outer hexagon angle can be independently optimized.
    
    # Coarse grid search
    cluster_angles = np.linspace(0.0, math.pi / 3.0, 121)  # 60 degrees in 1 degree steps
    
    for ca_angle in cluster_angles:
        if time.time() > time_limit:
            break
        
        ca = math.cos(ca_angle)
        sa = math.sin(ca_angle)
        
        # Rotate local centers
        rotated_centers = np.zeros_like(centers_local)
        rotated_centers[:, 0] = ca * centers_local[:, 0] - sa * centers_local[:, 1]
        rotated_centers[:, 1] = sa * centers_local[:, 0] + ca * centers_local[:, 1]
        
        # Compute all vertices for this cluster orientation
        all_pts = []
        for k in range(N):
            local_verts = _hex_vertices_local(ca_angle + angles_local[k])
            all_pts.append(local_verts + rotated_centers[k])
        all_pts = np.vstack(all_pts)
        
        # Center the cluster
        centroid = all_pts.mean(axis=0)
        centered_pts = all_pts - centroid
        
        # Now search over outer hexagon angle and translation
        # For a given outer hexagon angle, the optimal translation is to center the
        # bounding hexagon on the cluster's centroid in the hexagon's coordinate system.
        
        outer_angles = np.linspace(0.0, math.pi / 3.0, 121)
        
        for oa_angle in outer_angles:
            if time.time() > time_limit:
                break
            
            nrm = _norms(oa_angle)
            P = centered_pts @ nrm.T
            c = np.zeros(2) @ nrm.T  # outer center at origin for now
            req = (P - c).max(axis=0)
            side = 2.0 * float(np.max(req)) / math.sqrt(3.0)
            
            if side < best_side:
                best_side = side
                best_params = (ca_angle, 0.0, 0.0, oa_angle)
    
    return best_side, best_params

# ---------------------------------------------------------------------------
# Fine refinement around best parameters
# ---------------------------------------------------------------------------
def _refine_rigid(centers_local, angles_local, best_side, best_params, time_limit):
    """
    Fine-grained refinement around the best parameters found by grid search.
    """
    start = time.time()
    ca_angle, tx, ty, oa_angle = best_params
    
    best = best_side
    best_p = best_params
    
    # Fine grid around best cluster angle
    for dc in np.linspace(-2.0, 2.0, 41):  # +/- 2 degrees in 0.1 degree steps
        if time.time() > time_limit:
            break
        ca_new = ca_angle + math.radians(dc)
        
        ca = math.cos(ca_new)
        sa = math.sin(ca_new)
        
        rotated_centers = np.zeros_like(centers_local)
        rotated_centers[:, 0] = ca * centers_local[:, 0] - sa * centers_local[:, 1]
        rotated_centers[:, 1] = sa * centers_local[:, 0] + ca * centers_local[:, 1]
        
        all_pts = []
        for k in range(N):
            local_verts = _hex_vertices_local(ca_new + angles_local[k])
            all_pts.append(local_verts + rotated_centers[k])
        all_pts = np.vstack(all_pts)
        
        centroid = all_pts.mean(axis=0)
        centered_pts = all_pts - centroid
        
        # Fine grid around best outer angle
        for do in np.linspace(-2.0, 2.0, 41):
            if time.time() > time_limit:
                break
            oa_new = oa_angle + math.radians(do)
            
            nrm = _norms(oa_new)
            P = centered_pts @ nrm.T
            c = np.zeros(2) @ nrm.T
            req = (P - c).max(axis=0)
            side = 2.0 * float(np.max(req)) / math.sqrt(3.0)
            
            if side < best:
                best = side
                best_p = (ca_new, tx, ty, oa_new)
    
    return best, best_p

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    
    # Build the rigid honeycomb cluster
    centers_local, angles_local = _build_honeycomb_cluster()
    
    # Coarse grid search
    best_side, best_params = _grid_search_rigid(centers_local, angles_local, DEADLINE - 5.0)
    
    # Fine refinement
    if time.time() < DEADLINE - 1.0:
        best_side, best_params = _refine_rigid(centers_local, angles_local, best_side, best_params, DEADLINE - 0.5)
    
    # Extract best parameters
    ca_angle, tx, ty, oa_angle = best_params
    
    # Compute final configuration
    ca = math.cos(ca_angle)
    sa = math.sin(ca_angle)
    
    rotated_centers = np.zeros_like(centers_local)
    rotated_centers[:, 0] = ca * centers_local[:, 0] - sa * centers_local[:, 1] + tx
    rotated_centers[:, 1] = sa * centers_local[:, 0] + ca * centers_local[:, 1] + ty
    
    # Build inner hexagon list
    inner = []
    for k in range(N):
        inner.append([float(rotated_centers[k][0]), float(rotated_centers[k][1]), float(math.degrees(ca_angle + angles_local[k]))])
    
    # Outer hexagon center and parameters
    outer_center = (float(tx), float(ty))
    
    # Verify the side length
    all_pts = []
    for k in range(N):
        local_verts = _hex_vertices_local(ca_angle + angles_local[k])
        all_pts.append(local_verts + rotated_centers[k])
    all_pts = np.vstack(all_pts)
    
    final_side = _bounding_side(all_pts, oa_angle, np.array(outer_center))
    
    return inner, outer_center, final_side, math.degrees(oa_angle)
