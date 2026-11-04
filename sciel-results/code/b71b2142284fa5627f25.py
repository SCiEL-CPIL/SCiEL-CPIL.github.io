import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin for 120s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side_for_cluster(cluster_verts, outer_angle_deg):
    """
    Given a set of vertices (N, 2) representing the cluster in its local frame,
    and the outer hexagon's rotation angle, compute the minimum side length
    of a regular hexagon centered at the origin (of the cluster's centroid) 
    that contains all vertices, assuming the outer hexagon is rotated by outer_angle_deg.
    
    Actually, it's simpler to rotate the cluster vertices by -outer_angle_deg 
    and then find the minimal enclosing hexagon aligned with the axes.
    """
    a_outer = np.radians(outer_angle_deg)
    ca, sa = np.cos(-a_outer), np.sin(-a_outer)
    
    # Rotate cluster vertices
    rx = ca * cluster_verts[:, 0] - sa * cluster_verts[:, 1]
    ry = sa * cluster_verts[:, 0] + ca * cluster_verts[:, 1]
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin, aligned with x-axis
    # The hexagon is defined by |x| <= S, |x + y/sqrt(3)| <= S, |x - y/sqrt(3)| <= S
    # Where S is the side length? No.
    # Let's check the geometry.
    # Vertices of outer hexagon with side S, rotated 0 deg:
    # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # Equations of sides:
    # 1. x = S  -> x <= S
    # 2. x - sqrt(3)y = -S -> -x + sqrt(3)y <= S ? Let's check.
    #    Line through (S,0) and (S/2, S*sqrt(3)/2). Slope = (S*sqrt(3)/2 - 0) / (S/2 - S) = (S*sqrt(3)/2) / (-S/2) = -sqrt(3).
    #    y - 0 = -sqrt(3)(x - S) => y = -sqrt(3)x + S*sqrt(3) => sqrt(3)x + y = S*sqrt(3).
    #    So constraint: sqrt(3)x + y <= S*sqrt(3) => x + y/sqrt(3) <= S.
    # 3. Next side: through (S/2, S*sqrt(3)/2) and (-S/2, S*sqrt(3)/2). Horizontal line y = S*sqrt(3)/2.
    #    Constraint: y <= S*sqrt(3)/2.
    #    Note: y/sqrt(3) <= S/2. This is covered by the other inequalities?
    #    Let's use the 3 pairs of parallel sides.
    #    Axes: 1. (1, 0). 2. (1/2, sqrt(3)/2). 3. (0, 1) ?? No.
    #    The normals to the sides are at 0, 60, 120, 180, 240, 300 degrees?
    #    Side 1 (right): Normal (1, 0). Distance from origin to side is S. So x <= S.
    #    Side 2 (top-right): Normal (1/2, sqrt(3)/2). Distance is S. So (1/2)x + (sqrt(3)/2)y <= S.
    #    Side 3 (top-left): Normal (-1/2, sqrt(3)/2). Distance is S. So (-1/2)x + (sqrt(3)/2)y <= S.
    #    And their opposites.
    #    So we need S >= max( |x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2| ) for all points?
    #    Wait, the distance from origin to the side x=S is S.
    #    The distance from origin to the side (1/2)x + (sqrt(3)/2)y = S is S / sqrt((1/2)^2 + (sqrt(3)/2)^2) = S/1 = S.
    #    So yes, S must be at least the maximum projection of any vertex onto the 3 normal axes.
    
    c1 = np.abs(rx)
    c2 = np.abs(rx * 0.5 + ry * (np.sqrt(3)/2))
    c3 = np.abs(-rx * 0.5 + ry * (np.sqrt(3)/2))
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _generate_cluster_positions(topology_type):
    """
    Generate relative positions of 11 unit hexagons in a rigid cluster.
    topology_type: 'flower', 'line', 'compact'
    """
    sqrt3 = np.sqrt(3)
    # Hexagonal lattice vectors
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    # v3 = (-sqrt(3)/2, 3/2)
    
    if topology_type == 'flower':
        # Center + 6 neighbors + 4 more? 
        # Standard flower is 7. We need 11.
        # Add 4 more at distance 2*sqrt(3) in specific directions?
        # Or fill in the gaps of the second ring?
        # Second ring has 12 spots. We pick 4.
        pos = [(0,0)]
        for i in range(6):
            a = i * np.pi / 3
            pos.append((2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)))
        
        # Add 4 more. To make it more compact, maybe pick the ones that align with the first ring?
        # The second ring positions are at distance 2*sqrt(3) from center? No.
        # Distance from center to ring 1 is 2*sqrt(3) (since side 1, distance between centers of touching hexes is 2*apothem = sqrt(3)*2? No.
        # Apothem of unit hex is sqrt(3)/2. Distance between centers of touching hexes is 2 * apothem = sqrt(3).
        # Wait. Side length 1.
        # Distance from center to vertex is 1.
        # Distance from center to midpoint of side is sqrt(3)/2.
        # Two hexes touching side-to-side: distance between centers is 2 * sqrt(3)/2 = sqrt(3).
        
        # So my previous 'flower' code used 2*sqrt(3) which is wrong for touching.
        # Let's correct the lattice.
        # Center (0,0).
        # Neighbors at distance sqrt(3).
        # Angles: 0, 60, 120, 180, 240, 300.
        
        pos = [(0,0)]
        for i in range(6):
            a = i * np.pi / 3
            pos.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
            
        # Now we have 7. Need 4 more.
        # The "second shell" in a hexagonal packing consists of 12 hexagons.
        # Their centers are at distance 2*sqrt(3) from the origin? No.
        # Let's visualize.
        #       H
        #     H H H
        #   H H H H H
        #     H H H
        #       H
        # This is 1+6+12 = 19.
        # The 12 in the second shell are at positions:
        # 2*sqrt(3) * (cos(0), sin(0)) -> (2*sqrt(3), 0)
        # 2*sqrt(3) * (cos(30), sin(30)) -> (2*sqrt(3)*sqrt(3)/2, 2*sqrt(3)*1/2) = (3, sqrt(3))
        # etc.
        
        # Let's pick 4 from the 12 to minimize the bounding box.
        # Due to symmetry, picking 4 at 90 degree intervals might be good? 
        # The 12 are at angles 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330.
        # If we pick 0, 90, 180, 270, we get a cross shape.
        # If we pick 30, 150, 270, 30? No.
        # Let's try picking the ones that are "aligned" with the existing 6 to fill the corners.
        # The existing 6 are at 0, 60, 120...
        # The gaps are at 30, 90, 150...
        # If we put hexes at 30, 90, 150, 210, 270, 330, we fill the gaps.
        # We need 4. Let's pick 0, 120, 240 (aligned with existing) and maybe one more?
        # Or pick 30, 90, 150, 210?
        
        # Let's just generate the 12 and let the optimizer (if we allowed it) or a heuristic pick 4.
        # For a rigid cluster, we must fix the topology.
        # Let's try the "cross" topology: 0, 90, 180, 270.
        angles_4 = [0, np.pi/2, np.pi, 3*np.pi/2]
        for a in angles_4:
            pos.append((2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)))
            
    elif topology_type == 'compact':
        # Try to pack tighter.
        # Maybe shift the second ring?
        # Or use a different base.
        # Let's stick to the lattice but try different sets of 4 from the 12.
        # Set 1: 0, 90, 180, 270 (Cross)
        # Set 2: 30, 150, 270, 30? No. 30, 90, 150, 210?
        # Let's try 30, 150, 210, 330. (Diamond-ish)
        angles_4 = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
        
        pos = [(0,0)]
        for i in range(6):
            a = i * np.pi / 3
            pos.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
            
        for a in angles_4:
            pos.append((2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)))
            
    else:
        # Fallback to flower cross
        pos = [(0,0)]
        for i in range(6):
            a = i * np.pi / 3
            pos.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
        angles_4 = [0, np.pi/2, np.pi, 3*np.pi/2]
        for a in angles_4:
            pos.append((2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)))

    return np.array(pos)

def _get_cluster_vertices(cluster_pos, cluster_angle_deg):
    """
    Get all vertices of the cluster, rotated by cluster_angle_deg.
    cluster_pos: (N, 2) relative positions of centers.
    """
    a = np.radians(cluster_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    
    # Rotate centers
    cx = ca * cluster_pos[:, 0] - sa * cluster_pos[:, 1]
    cy = sa * cluster_pos[:, 0] + ca * cluster_pos[:, 1]
    
    # For each hexagon, generate vertices
    # Base angles for hex vertices: 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    
    # All vertices for all hexes
    # (N, 6) angles
    all_angles = a + base_angles[None, :]
    
    # Offsets
    ox = np.cos(all_angles)
    oy = np.sin(all_angles)
    
    # Add to centers
    # cx is (N,), ox is (N, 6)
    vx = cx[:, None] + ox
    vy = cy[:, None] + oy
    
    return vx.ravel(), vy.ravel()

def _objective_rigid(params):
    """
    params: [x, y, angle_deg]
    x, y: translation of cluster centroid
    angle_deg: rotation of cluster
    """
    x, y, ang = params
    
    # Get cluster vertices in global frame
    vx, vy = _get_cluster_vertices(cluster_pos, ang)
    
    # Translate
    vx += x
    vy += y
    
    # We want to minimize the outer hexagon side length.
    # The outer hexagon is centered at (0,0) and has fixed rotation 0 (we can absorb outer rotation into cluster rotation).
    # So we just need the minimal enclosing hexagon for these points, aligned with axes.
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(vx)
    c2 = np.abs(vx * 0.5 + vy * (np.sqrt(3)/2))
    c3 = np.abs(-vx * 0.5 + vy * (np.sqrt(3)/2))
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def construct():
    global START_TIME, DEADLINE
    
    # Try a few different rigid topologies
    topologies = ['flower', 'compact']
    best_side = float('inf')
    best_params = None
    best_topology = None
    
    for topo in topologies:
        if time.time() > DEADLINE:
            break
            
        cluster_pos = _generate_cluster_positions(topo)
        
        # Initial guess: centered at 0,0, angle 0
        x0 = [0.0, 0.0, 0.0]
        
        # Optimization bounds
        # The cluster size is roughly 2*sqrt(3) ~ 3.5 radius.
        # So translation should be small.
        bounds = [(-5, 5), (-5, 5), (0, 360)]
        
        # Use Differential Evolution for global optimization of the 3 params
        # This is robust and fast for 3D space.
        try:
            res = differential_evolution(_objective_rigid, bounds, seed=42, maxiter=100, tol=1e-6, polish=True)
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
                best_topology = topo
        except Exception:
            pass
            
        # Also try local refinement with Nelder-Mead from the DE result
        if best_topology == topo:
            try:
                res2 = minimize(_objective_rigid, best_params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
                if res2.fun < best_side:
                    best_side = res2.fun
                    best_params = res2.x
            except Exception:
                pass

    if best_params is None:
        # Fallback
        best_params = [0.0, 0.0, 0.0]
        best_topology = 'flower'
        best_side = _objective_rigid(best_params)
        
    # Extract best solution
    x, y, ang = best_params
    cluster_pos = _generate_cluster_positions(best_topology)
    
    # Construct the inner list
    # The inner list requires absolute coordinates and angles for each hexagon.
    # Our cluster_pos are relative to the cluster centroid.
    # We rotated the cluster by ang and translated by (x,y).
    
    a_rad = np.radians(ang)
    ca, sa = np.cos(a_rad), np.sin(a_rad)
    
    inner_list = []
    for i in range(N):
        cx_rel, cy_rel = cluster_pos[i, 0], cluster_pos[i, 1]
        # Rotate relative center
        cx_rot = ca * cx_rel - sa * cy_rel
        cy_rot = sa * cx_rel + ca * cy_rel
        # Translate
        cx_abs = cx_rot + x
        cy_abs = cy_rot + y
        # The angle of the hexagon itself is the cluster angle
        # Note: The hexagon vertices are generated with angle 0 in _get_cluster_vertices relative to the cluster frame?
        # In _get_cluster_vertices, we used `all_angles = a + base_angles`.
        # So the hexagons are rotated by `ang` relative to the global frame.
        # So the angle_deg for the interface is `ang`.
        
        inner_list.append([float(cx_abs), float(cy_abs), float(ang)])
        
    # Outer hexagon
    # Center (0,0), side length best_side, angle 0
    outer_center = (0.0, 0.0)
    outer_side = float(best_side)
    outer_angle = 0.0
    
    return inner_list, outer_center, outer_side, outer_angle

if __name__ == "__main__":
    construct()
