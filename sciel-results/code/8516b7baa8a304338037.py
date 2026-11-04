import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False  # Found separating axis -> no overlap
    return True  # No separating axis -> overlap

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using distance-based pre-filter + SAT fallback.
    cx_arr, cy_arr, angle_arr: numpy arrays of shape (N,)
    Returns True if all hexagons are pairwise disjoint.
    """
    n = len(cx_arr)
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    # Precompute axes for all hexagons
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
            
            # Quick distance check
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            
            # If distance is large enough, definitely disjoint
            # Max extent of a unit hexagon from center is 1 (circumradius)
            # So if dist > 2, they could still overlap (vertex-to-vertex at 30°)
            # If dist > 2*sqrt(3) ≈ 3.464, definitely disjoint
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            # If distance is very small, definitely overlapping
            # Min non-overlap distance is 2.0 (at 30° relative rotation)
            if dist_sq < 4.0 - 1e-9:
                return False
            
            # Intermediate case: do SAT check
            # Combine axes from both polygons
            for axis in ai:
                ax, ay = axis
                # Project both polygons
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                # No separating axis found in ai, check aj
                for axis in aj:
                    ax, ay = axis
                    minA = min(vx * ax + vy * ay for vx, vy in vi)
                    maxA = max(vx * ax + vy * ay for vx, vy in vi)
                    minB = min(vx * ax + vy * ay for vx, vy in vj)
                    maxB = max(vx * ax + vy * ay for vx, vy in vj)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    # No separating axis at all -> overlap
                    return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # 6 outward normals of the outer hexagon
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _get_rigid_topologies():
    """
    Generate a list of rigid topologies.
    Each topology is a list of (x, y, angle) relative to the cluster center.
    The cluster center is (0,0).
    """
    topologies = []
    
    # 1. 2x6 Honeycomb (Rectangular-ish)
    # Hex lattice vectors: (2, 0) and (1, sqrt(3))
    # Actually, for "flat top" hexagons (vertex at 0 deg), the neighbors are at:
    # (2, 0), (1, sqrt(3)), (-1, sqrt(3)), (-2, 0), (-1, -sqrt(3)), (1, -sqrt(3))
    # Let's use "pointy top" for easier lattice math? 
    # Standard hex lattice for pointy top (vertex up):
    # dx = 2, dy = 0 for horizontal? No, width is 2, height is sqrt(3).
    # Let's stick to standard unit hexagons with flat top (vertex at 30, 90, ...).
    # Wait, _hex_vertices uses theta = 60k. k=0 -> (1,0). So vertex at 0 deg.
    # This is "pointy top" relative to x-axis? No, vertex on x-axis.
    # Neighbors for this orientation:
    # Distance to neighbor center: 2 * sqrt(3) / 2 * 2? 
    # Apothem = sqrt(3)/2. Distance between centers of touching hexes = 2 * apothem = sqrt(3).
    # Directions: 0, 60, 120, 180, 240, 300.
    # So neighbors are at distance sqrt(3) in those directions.
    
    # Let's construct a 2x6 grid in the lattice.
    # Basis vectors: u = (sqrt(3), 0), v = (sqrt(3)/2, 3/2) ? 
    # Distance between centers is sqrt(3).
    # If we align one axis with x, neighbors are at (sqrt(3), 0) and (sqrt(3)/2, 3/2).
    
    # Topology 1: 2 rows of 6
    pts = []
    for r in range(2):
        for c in range(6):
            x = c * math.sqrt(3) + (r % 2) * math.sqrt(3)/2
            y = r * 1.5 # Vertical distance is 3/2? 
            # Let's verify.
            # Hex 1 at (0,0). Neighbor at (sqrt(3), 0).
            # Next row neighbor of Hex 1 is at (sqrt(3)/2, 1.5).
            # Check dist: sqrt(3/4 + 2.25) = sqrt(3). Correct.
            pts.append([x, y, 0.0])
    # Center the cluster
    cx_avg = sum(p[0] for p in pts) / len(pts)
    cy_avg = sum(p[1] for p in pts) / len(pts)
    for p in pts:
        p[0] -= cx_avg
        p[1] -= cy_avg
    topologies.append(pts)
    
    # Topology 2: 3 rows of 4
    pts = []
    for r in range(3):
        for c in range(4):
            x = c * math.sqrt(3) + (r % 2) * math.sqrt(3)/2
            y = r * 1.5
            pts.append([x, y, 0.0])
    cx_avg = sum(p[0] for p in pts) / len(pts)
    cy_avg = sum(p[1] for p in pts) / len(pts)
    for p in pts:
        p[0] -= cx_avg
        p[1] -= cy_avg
    topologies.append(pts)

    # Topology 3: "Flower" + 5 extensions (less dense but might fit better in hex?)
    # Center + 6 ring + 5 more.
    # This is hard to make rigid and valid without overlap checks.
    # Let's try a "Hexagon of hexagons" 3x3x3? No, 7 in a hexagon.
    # 12 is 7 + 5.
    # Let's try 1 center, 6 ring, and 5 in the "valleys" of the second ring?
    # Second ring has 12 spots. We need 5.
    # This might be too sparse.
    
    # Topology 4: 4 rows of 3
    pts = []
    for r in range(4):
        for c in range(3):
            x = c * math.sqrt(3) + (r % 2) * math.sqrt(3)/2
            y = r * 1.5
            pts.append([x, y, 0.0])
    cx_avg = sum(p[0] for p in pts) / len(pts)
    cy_avg = sum(p[1] for p in pts) / len(pts)
    for p in pts:
        p[0] -= cx_avg
        p[1] -= cy_avg
    topologies.append(pts)
        
    return topologies

def construct():
    t0 = time.time()
    random.seed(123)
    
    topologies = _get_rigid_topologies()
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_global_rot = 0.0
    
    apothem_unit = math.sqrt(3) / 2.0
    
    # Precompute vertices for a unit hexagon at origin, angle 0
    # We will rotate and translate these.
    
    for topo in topologies:
        if time.time() - t0 > TIME_LIMIT - 20:
            break
            
        # Precompute local vertices for this topology
        # topo[i] = [x, y, angle]
        local_verts = []
        for i in range(len(topo)):
            x, y, a = topo[i]
            # Generate vertices relative to (x,y) with rotation a
            verts = _hex_vertices(x, y, a)
            local_verts.append(np.array(verts))
            
        # Stack into a single array (N, 6, 2)
        all_verts = np.array(local_verts) # Shape (12, 6, 2)
        
        # Function to evaluate side length for given global_rot and outer_rot
        def eval_side(params):
            global_rot, outer_rot = params
            gr = math.radians(global_rot)
            orr = math.radians(outer_rot)
            
            # Rotate all vertices by global_rot
            # Rotation matrix
            cg, sg = math.cos(gr), math.sin(gr)
            # Apply rotation to all_verts
            # v' = R * v
            # v'x = cg*vx - sg*vy
            # v'y = sg*vx + cg*vy
            
            vx = all_verts[:, :, 0]
            vy = all_verts[:, :, 1]
            
            rx = cg * vx - sg * vy
            ry = sg * vx + cg * vy
            
            # Find bounding box in rotated frame? No, we need to project onto outer hex normals.
            # Outer hex normals depend on outer_rot.
            # Normals are at angles: outer_rot + 90, outer_rot + 150, ...
            
            max_proj = 0.0
            # 6 normals
            for k in range(6):
                na = orr + math.radians(90 + 60 * k)
                nx = math.cos(na)
                ny = math.sin(na)
                
                # Project all vertices onto this normal
                # proj = rx * nx + ry * ny
                proj = rx * nx + ry * ny
                # max over all vertices
                p_max = np.max(proj)
                p_min = np.min(proj)
                
                # The width in this direction is p_max - p_min
                # The side length S must satisfy: S * sqrt(3) / 2 * 2 >= width?
                # Distance between parallel sides of hexagon with side S is S * sqrt(3).
                # So S >= width / sqrt(3).
                # Wait, the "side length" in the task is the side of the outer hexagon.
                # The width of a regular hexagon with side S is 2 * apothem = S * sqrt(3).
                # So we need S * sqrt(3) >= max_width.
                # S >= max_width / sqrt(3).
                
                width = p_max - p_min
                required_s = width / math.sqrt(3)
                
                if required_s > max_proj:
                    max_proj = required_s
            
            return max_proj

        # Initial guess: 0, 0
        # Grid search over angles to find a good start
        best_local_side = float('inf')
        best_local_params = (0, 0)
        
        # Coarse grid
        for g_rot in range(0, 60, 5):
            for o_rot in range(0, 60, 5):
                side = eval_side((g_rot, o_rot))
                if side < best_local_side:
                    best_local_side = side
                    best_local_params = (g_rot, o_rot)
                    
        # Refine with Nelder-Mead
        try:
            res = minimize(eval_side, best_local_params, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_local_side:
                best_local_side = res.fun
                best_local_params = res.x
        except:
            pass
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_global_rot = best_local_params[0]
            best_angle = best_local_params[1]
            
            # Reconstruct final configuration
            gr = math.radians(best_global_rot)
            cg, sg = math.cos(gr), math.sin(gr)
            
            final_cfg = []
            for i in range(len(topo)):
                x, y, a = topo[i]
                # Rotate position
                fx = cg * x - sg * y
                fy = sg * x + cg * y
                # Rotate angle
                fa = a + best_global_rot
                final_cfg.append([fx, fy, fa])
                
            # The center of the outer hexagon should be the centroid of the cluster?
            # Actually, the optimal center minimizes the max projection distance.
            # For a hexagon, the center that minimizes the side length given the orientation
            # is the midpoint of the min/max projections for each pair of opposite sides?
            # Since we computed width, the center that centers the width is (min+max)/2.
            # We should calculate the optimal center for the found angles.
            
            orr = math.radians(best_angle)
            cx_sum = 0
            cy_sum = 0
            # Actually, we can just set center to (0,0) if we centered the topology.
            # But the "width" calculation assumed the cluster is centered at origin?
            # No, `eval_side` computed width (max-min).
            # The side length is determined by width.
            # The position of the outer hexagon center doesn't change the width,
            # only the max distance from center.
            # Wait, `_outer_hex_side` in the original code computes:
            # max( (nx*(vx-cx) + ny*(vy-cy)) / apothem )
            # This is effectively (max_proj - cx_proj) / apothem.
            # To minimize the max of these 6 values, we should choose cx, cy such that
            # the projections are centered.
            # i.e., cx_proj = (min_proj + max_proj) / 2 for each normal?
            # This is only possible if the hexagon is "balanced".
            # For a regular hexagon, the optimal center is the centroid of the bounding box?
            # Actually, if we define the outer hexagon by its center and side,
            # and we want to contain the cluster,
            # the minimum side S is max( width_k / sqrt(3) ).
            # The center of the outer hexagon should be the center of the cluster's bounding box?
            # If we place the outer hexagon center at the cluster's "center" (midpoint of min/max for all axes),
            # the required S is max_width / sqrt(3).
            
            # So we just need to return the cluster centered at (0,0) and outer center at (0,0).
            # Is the cluster centered at (0,0)?
            # We centered the topology around its centroid.
            # So yes, (0,0) is a good center.
            
            best_cfg = final_cfg
            best_center = (0.0, 0.0)

    # Fallback if no topology worked (shouldn't happen)
    if best_cfg is None:
        # Use a loose random config
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0])
        # Scale to be valid
        scale = 2.0
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = _outer_hex_side(np.array([p[0] for p in cfg]), np.array([p[1] for p in cfg]), np.array([p[2] for p in cfg]), best_center, best_angle)

    # Final verification and return
    cx_final = np.array([p[0] for p in best_cfg])
    cy_final = np.array([p[1] for p in best_cfg])
    angle_final = np.array([p[2] for p in best_cfg])
    
    # Ensure validity (should be valid for rigid topologies)
    # If not, fallback to scaled random
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        cfg = []
        for i in range(N):
            cfg.append([0, 0, 0])
        # Spread them out
        for i in range(N):
            angle = 2 * math.pi * i / N
            cfg[i][0] = 3 * math.cos(angle)
            cfg[i][1] = 3 * math.sin(angle)
        best_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
