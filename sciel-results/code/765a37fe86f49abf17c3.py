import numpy as np
import time
from scipy.optimize import differential_evolution
from itertools import product

N = 12
TIME_LIMIT = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check disjointness via segment intersection
    def _seg_seg_dist(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(r)
        elif a < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            return np.linalg.norm(r + t * d2)
        elif e < 1e-12:
            s = -np.dot(d1, r) / a
            s = max(0.0, min(1.0, s))
            return np.linalg.norm(s * d1 - r)
        
        c = np.dot(d1, r)
        b = np.dot(d1, d2)
        denom = a * e - b * b
        
        if denom < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            s = max(0.0, min(1.0, (b * t - c) / a))
        else:
            s = (b * f - c * e) / denom
            if s < 0.0 or s > 1.0:
                s = 0.0
                t = f / e
                t = max(0.0, min(1.0, t))
            else:
                t = (b * s + f) / e
                t = max(0.0, min(1.0, t))
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))
        
        return np.linalg.norm(s * d1 - r - t * d2)

    def _polygons_overlap(v1, v2, eps=1e-9):
        n1, n2 = len(v1), len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i + 1) % n1]
            for j in range(n2):
                q1 = v2[j]
                q2 = v2[(j + 1) % n2]
                if _seg_seg_dist(p1, p2, q1, q2) < eps:
                    return True
        return False

    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=1e-9):
                return False, "overlap"
            
            # Check containment
            is_inside = True
            for p in inner_verts[i]:
                inside = True
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k+1)%6]
                    cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                    if cross_ab < -eps:
                        inside = False
                        break
                if not inside:
                    is_inside = False
                    break
            
            if not is_inside:
                return False, "outside"
                
    return True, "valid"

def _compute_required_side(centers, angles, outer_angle=0.0):
    """Compute the minimum outer side length required to contain all hexagons."""
    max_proj = 0.0
    # Outer hexagon normals
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def _generate_topologies():
    """
    Generate a set of plausible adjacency topologies for 12 hexagons.
    We use a set of predefined 'shapes' or 'clusters' that are known to pack well.
    Since we can't easily enumerate all graph isomorphisms for N=12 in real-time,
    we use a few hand-crafted dense clusters.
    """
    topologies = []
    
    # 1. Hexagonal cluster of 19, take 12
    # Center at (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes
    # We pick 1 from center, 6 from ring 1, 5 from ring 2
    
    # Helper to get hex lattice neighbors
    def hex_to_pixel(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y
    
    # Generate lattice points
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                x, y = hex_to_pixel(q, r)
                points.append((x, y))
                
    # Sort by distance from center to get densest cluster
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Topology 1: Densest 12 (Center + 6 neighbors + 5 next)
    top1 = points[:12]
    topologies.append(top1)
    
    # Topology 2: More elongated
    # Take a 2x6 rectangle in hex lattice
    pts2 = []
    for i in range(2):
        for j in range(6):
            q = i - 1
            r = j - 2
            x, y = hex_to_pixel(q, r)
            pts2.append((x, y))
    topologies.append(pts2)
    
    # Topology 3: 3x4 rectangle
    pts3 = []
    for i in range(3):
        for j in range(4):
            q = i - 1
            r = j - 1
            x, y = hex_to_pixel(q, r)
            pts3.append((x, y))
    topologies.append(pts3)
    
    return topologies

def _optimize_topology(topology, outer_angle, time_budget=10.0):
    """
    Optimize the orientation of the cluster defined by topology to minimize
    the required outer side length.
    
    The topology defines relative positions. We assume all hexagons have the
    same orientation relative to the cluster frame. We optimize the global
    rotation of the cluster.
    """
    # Convert topology to array
    topo_arr = np.array(topology)
    
    # The variables to optimize:
    # 1. Global rotation of the cluster (0 to 60 degrees, due to hex symmetry)
    # 2. We can also try slight individual rotations, but that increases dimensionality.
    #    Let's start with just global rotation.
    
    def objective(params):
        rot = params[0]
        
        # Rotate topology
        theta = np.deg2rad(rot)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        rotated_topo = (R @ topo_arr.T).T
        
        # Centers are rotated_topo
        # Angles are all 'rot' (since the hexagons are aligned with the lattice)
        # Actually, if the lattice is rotated, the hexagons are rotated too.
        angles = np.full(N, rot)
        
        # Compute required side
        return _compute_required_side(rotated_topo, angles, outer_angle)
    
    # Bounds for rotation: 0 to 60
    bounds = [(0, 60)]
    
    start_time = time.time()
    try:
        # Use differential evolution for global search
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            polish=True
        )
        
        return result.fun, result.x[0]
        
    except Exception as e:
        return float('inf'), 0.0

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Strategy:
    # 1. Generate a set of topologies (relative positions).
    # 2. For each topology, optimize the global orientation using DE.
    # 3. Keep the best valid configuration.
    
    topologies = _generate_topologies()
    
    # Try different outer angles to break symmetry
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for topo in topologies:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                # Allocate time based on remaining budget
                remaining = TIME_LIMIT - 5 - (time.time() - start_time)
                time_budget = min(10.0, remaining / 2.0) # Conservative
                
                if time_budget < 1.0:
                    continue
                    
                side, best_rot = _optimize_topology(topo, outer_angle, time_budget)
                
                if side < best_side:
                    # Construct the inner hexagons
                    theta = np.deg2rad(best_rot)
                    R = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                    ])
                    
                    topo_arr = np.array(topo)
                    rotated_topo = (R @ topo_arr.T).T
                    angles = np.full(N, best_rot)
                    
                    inner_hexes = [[rotated_topo[i][0], rotated_topo[i][1], angles[i]] for i in range(N)]
                    
                    # Verify validity
                    is_valid, reason = _check_validity(inner_hexes, side, (0.0, 0.0), outer_angle)
                    if is_valid:
                        best_side = side
                        best_inner = inner_hexes
                        best_outer_center = (0.0, 0.0)
                        best_outer_angle = outer_angle
                        
            except Exception as e:
                pass
                
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

