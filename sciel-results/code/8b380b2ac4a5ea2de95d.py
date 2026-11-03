import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
START = time.time()
DEADLINE = 150.0

def _get_inner_hex_vertices(cx, cy, s, rot_deg):
    """
    Get vertices of a regular hexagon.
    cx, cy: center
    s: side length
    rot_deg: rotation in degrees
    """
    a0 = math.radians(rot_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _compute_min_side(cluster_positions, cluster_rot_deg, outer_rot_deg, outer_center=(0.0, 0.0)):
    """
    Compute the minimum side length of the outer hexagon.
    cluster_positions: list of (dx, dy, local_rot_deg) relative to cluster center.
    cluster_rot_deg: global rotation of the cluster.
    outer_rot_deg: rotation of the outer hexagon.
    outer_center: center of the outer hexagon.
    """
    # Precompute normals for the outer hexagon
    theta_rad = math.radians(outer_rot_deg)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_center
    max_proj = 0.0
    
    for (dx, dy, lrot) in cluster_positions:
        # Global rotation of this specific hexagon
        grot = cluster_rot_deg + lrot
        # Center of this hexagon in global coords
        # Note: The local position (dx, dy) is defined in the unrotated cluster frame.
        # We need to rotate (dx, dy) by cluster_rot_deg to get global offset.
        a_c = math.radians(cluster_rot_deg)
        gx = dx * math.cos(a_c) - dy * math.sin(a_c)
        gy = dx * math.sin(a_c) + dy * math.cos(a_c)
        
        # Get vertices of this hexagon
        verts = _get_inner_hex_vertices(gx, gy, 1.0, grot)
        
        for (px, py) in verts:
            for (nx, ny) in normals:
                proj = abs((px - ox) * nx + (py - oy) * ny)
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _objective(params):
    """
    Objective function for optimization.
    params: [cluster_rot_deg, outer_rot_deg, cx, cy]
    """
    crot, orot, cx, cy = params
    # We fix the cluster center at (cx, cy) and outer center at (0,0) for simplicity?
    # Actually, translating the cluster is equivalent to translating the outer hexagon in the other direction.
    # Since the outer hexagon is symmetric, we can fix the outer center at (0,0) and let the cluster center move.
    # Or fix cluster at (0,0) and move outer center.
    # Let's fix outer center at (0,0) and optimize cluster center (cx, cy).
    # Wait, the cluster positions are relative to the cluster center.
    # So global position of a part = cluster_center + R(cluster_rot) * local_pos.
    # This is already handled in _compute_min_side if we pass (0,0) as outer_center?
    # No, _compute_min_side takes cluster_positions relative to cluster center.
    # It computes global pos as R(cluster_rot) * local_pos.
    # It assumes cluster center is at (0,0) in the global frame?
    # Let's check _compute_min_side logic.
    # gx = dx * cos(a_c) ...
    # This assumes the cluster center is at (0,0).
    # So if we want to translate the cluster, we should add cx, cy to gx, gy.
    # Let's modify _compute_min_side to accept cluster_center.
    
    pass # Placeholder, will rewrite function below

# Rewrite _compute_min_side to include cluster center
def _compute_min_side_v2(cluster_positions, cluster_rot_deg, outer_rot_deg, cluster_center=(0.0, 0.0), outer_center=(0.0, 0.0)):
    theta_rad = math.radians(outer_rot_deg)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_center
    cx, cy = cluster_center
    max_proj = 0.0
    
    a_c = math.radians(cluster_rot_deg)
    cos_c = math.cos(a_c)
    sin_c = math.sin(a_c)
    
    for (dx, dy, lrot) in cluster_positions:
        # Rotate local position to global
        gx = dx * cos_c - dy * sin_c + cx
        gy = dx * sin_c + dy * cos_c + cy
        
        grot = cluster_rot_deg + lrot
        verts = _get_inner_hex_vertices(gx, gy, 1.0, grot)
        
        for (px, py) in verts:
            for (nx, ny) in normals:
                proj = abs((px - ox) * nx + (py - oy) * ny)
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _objective_v2(params, topology):
    crot, orot, cx, cy = params
    return _compute_min_side_v2(topology, crot, orot, (cx, cy), (0.0, 0.0))

def _generate_topology_flower():
    """
    1 center, 6 ring 1, 4 ring 2.
    Hexagon side 1. Distance between centers of touching hexagons is sqrt(3).
    """
    pos = []
    pos.append((0.0, 0.0, 0.0)) # Center
    
    d1 = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        # Local rotation 0 means flat-top? 
        # _get_inner_hex_vertices uses 0 deg as vertex at angle 0 (pointy-top).
        # Standard hex tiling usually aligns edges.
        # If vertex is at 0 deg, the hex is pointy-top.
        # Let's keep rotations consistent.
        pos.append((d1 * math.cos(a), d1 * math.sin(a), 0.0))
        
    d2 = 2 * d1
    # 4 hexagons in second ring. Which ones?
    # In a full ring of 6, the 2nd ring has 6 hexagons further out.
    # To pack 11, we take 4 of them.
    # Optimal is usually to cluster them or spread them.
    # Let's try all 4 consecutive? Or alternating?
    # Alternating (0, 2, 4, 6 -> indices 0,2,4,6 in 0..5? No, 0,2,4,6 is 4 items? 0,2,4,6 is 4 items if we skip 1,3,5.
    # Indices 0, 2, 4, 6? No, range(6) is 0..5.
    # Let's pick indices 0, 2, 4, 6? No, 6 is out of range.
    # Let's pick 0, 2, 4, 5?
    # Actually, for 11 hexagons, the "flower" with 4 in the second ring is common.
    # The 4 hexagons should be placed to minimize the bounding box.
    # Placing them at 0, 120, 240, 300? No.
    # Let's try 0, 2, 4, 6 (if we had 6). We have 6 slots.
    # Let's try placing them at angles 0, 120, 240, 300?
    # Let's just use a generic set and let the optimizer rotate the whole cluster.
    # Topology A: 0, 2, 4, 6 (mod 6) -> 0, 2, 4, 0? No.
    # Let's try 0, 2, 4, 5.
    for k in [0, 2, 4, 5]:
        a = math.radians(30 + k * 60)
        pos.append((d2 * math.cos(a), d2 * math.sin(a), 0.0))
    return pos

def _generate_topology_compact():
    """
    1 center, 6 ring 1, 4 ring 2 clustered.
    """
    pos = []
    pos.append((0.0, 0.0, 0.0))
    d1 = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        pos.append((d1 * math.cos(a), d1 * math.sin(a), 0.0))
        
    d2 = 2 * d1
    # Cluster them in one quadrant?
    # Indices 0, 1, 5, 4?
    for k in [0, 1, 5, 4]:
        a = math.radians(30 + k * 60)
        pos.append((d2 * math.cos(a), d2 * math.sin(a), 0.0))
    return pos

def _generate_topology_spread():
    """
    1 center, 6 ring 1, 4 ring 2 spread.
    """
    pos = []
    pos.append((0.0, 0.0, 0.0))
    d1 = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        pos.append((d1 * math.cos(a), d1 * math.sin(a), 0.0))
        
    d2 = 2 * d1
    # Spread: 0, 2, 4, 6? No. 0, 2, 4, 5?
    # Let's try 0, 2, 4, 6 -> 0, 2, 4, 0 is bad.
    # 0, 2, 4, 6 is 4 items if we consider 0..5. 0, 2, 4, 6 is invalid.
    # 0, 2, 4, 5.
    for k in [0, 2, 4, 5]:
        a = math.radians(30 + k * 60)
        pos.append((d2 * math.cos(a), d2 * math.sin(a), 0.0))
    return pos

def construct():
    best_score = float('inf')
    best_params = None
    best_topology = None
    
    topologies = [
        _generate_topology_flower(),
        _generate_topology_compact(),
        _generate_topology_spread()
    ]
    
    # Bounds for: [cluster_rot, outer_rot, cx, cy]
    # Rotations are periodic 0-360 (or 0-60 due to symmetry? 60 for hex).
    # Let's use 0-60 for both rotations to exploit symmetry and reduce search space.
    # Translation: The cluster fits in a box of roughly size 4x4.
    # cx, cy should be within [-5, 5].
    bounds = [
        (0, 60),
        (0, 60),
        (-5, 5),
        (-5, 5)
    ]
    
    budget = START + DEADLINE - 10.0
    
    for topo in topologies:
        if time.time() > budget:
            break
            
        # Use Differential Evolution for global search
        try:
            res = differential_evolution(
                _objective_v2,
                bounds,
                args=(topo,),
                maxiter=1000,
                popsize=20,
                tol=1e-4,
                mutation=(0.5, 1.0),
                recombination=0.9,
                seed=42
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_topology = topo
        except Exception:
            pass
        
        # Refine with Nelder-Mead
        if time.time() < budget - 5.0 and best_params is not None:
            try:
                res_nm = minimize(
                    _objective_v2,
                    best_params,
                    args=(best_topology,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res_nm.fun < best_score:
                    best_score = res_nm.fun
                    best_params = res_nm.x
                    best_topology = topo
            except Exception:
                pass

    if best_params is None:
        # Fallback: just use the first topology with 0 rotation
        topo = _generate_topology_flower()
        best_score = _compute_min_side_v2(topo, 0, 0, (0,0), (0,0))
        best_params = [0, 0, 0, 0]
        best_topology = topo

    crot, orot, cx, cy = best_params
    
    # Construct the final inner hexagon list
    # We need to return: inner (list of [x, y, angle_deg]), center (x, y), side, angle
    # The evaluator expects:
    # inner: list of N tuples/arrays (x, y, angle_deg)
    # center: (x, y) of outer hex
    # side: side length
    # angle: rotation of outer hex
    
    inner = []
    a_c = math.radians(crot)
    cos_c = math.cos(a_c)
    sin_c = math.sin(a_c)
    
    for (dx, dy, lrot) in best_topology:
        gx = dx * cos_c - dy * sin_c + cx
        gy = dx * sin_c + dy * cos_c + cy
        grot = crot + lrot
        inner.append((gx, gy, grot))
        
    # The outer hexagon is centered at (0,0) in our optimization.
    # We can shift everything so the outer hexagon center is at (0,0) or keep it.
    # The return interface asks for outer center.
    # We optimized with outer center at (0,0).
    # So outer center is (0,0).
    # Side is best_score.
    # Angle is orot.
    
    return inner, (0.0, 0.0), best_score, orot
