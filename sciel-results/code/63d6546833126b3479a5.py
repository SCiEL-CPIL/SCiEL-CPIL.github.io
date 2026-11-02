import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 115.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial coordinates for N=11 hexagons
    # We generate several candidate configurations

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Candidate configurations
    candidates = []

    # Configuration 1: Central hexagon with 6 around, then 4 more
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_1)

    # Configuration 2: Linear-ish arrangement
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2)
    ]
    candidates.append(axial_2)

    # Configuration 3: Compact 3x3 with extras
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # Configuration 4: Star pattern
    axial_4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_4)

    # Configuration 5: Diagonal arrangement
    axial_5 = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
        (0, 1), (1, 2), (2, 3),
        (1, 0), (2, 1)
    ]
    candidates.append(axial_5)

    # Configuration 6: T-shape
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (1, -1)
    ]
    candidates.append(axial_6)

    # Configuration 7: Cross pattern
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 0), (-2, 0)
    ]
    candidates.append(axial_7)

    # Configuration 8: Compact cluster
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_8)

    # Configuration 9: Offset rows
    axial_9 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_9)

    # Configuration 10: Hexagonal spiral
    axial_10 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0)
    ]
    candidates.append(axial_10)

    def hexagon_vertices(cx, cy, side=1.0, angle_deg=0.0):
        """Generate vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for k in range(6):
            ang = angle_rad + k * math.pi / 3.0
            x = cx + side * math.cos(ang)
            y = cy + side * math.sin(ang)
            vertices.append((x, y))
        return np.array(vertices)

    def polygon_area(poly):
        """Shoelace formula for polygon area."""
        n = len(poly)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
        return abs(area) / 2.0

    def polygons_overlap(v1, v2, tol=1e-10):
        """Check if two convex polygons overlap using separation axis theorem."""
        # Get all edge normals from both polygons
        axes = []
        for poly in [v1, v2]:
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                edge = poly[j] - poly[i]
                # Normal to edge
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-15:
                    axes.append(normal / norm)
        
        for axis in axes:
            # Project both polygons onto axis
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 + tol < min2 or max2 + tol < min1:
                return False
        
        return True

    def compute_repulsion_forces(positions, angles, k_rep=5.0, k_attract=0.1, rest_length=math.sqrt(3) + 1.0):
        """Compute forces for force-directed layout."""
        n = len(positions)
        forces = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                pi = np.array(positions[i])
                pj = np.array(positions[j])
                diff = pi - pj
                dist = np.linalg.norm(diff)
                
                if dist < 1e-10:
                    # Random small perturbation
                    forces[i] += np.random.uniform(-0.1, 0.1, 2)
                    forces[j] -= np.random.uniform(-0.1, 0.1, 2)
                    continue
                
                dir_vec = diff / dist
                
                # Check for overlap
                vi = hexagon_vertices(pi[0], pi[1], 1.0, angles[i])
                vj = hexagon_vertices(pj[0], pj[1], 1.0, angles[j])
                
                if polygons_overlap(vi, vj):
                    # Strong repulsion
                    force_mag = k_rep * (1.0 / (dist + 0.1))
                    forces[i] += force_mag * dir_vec
                    forces[j] -= force_mag * dir_vec
                else:
                    # Weak attraction to form compact cluster
                    if dist > rest_length:
                        force_mag = k_attract * (dist - rest_length)
                        forces[i] -= force_mag * dir_vec
                        forces[j] += force_mag * dir_vec
        
        return forces

    def force_directed_layout(positions, angles, iterations=500, dt=0.01):
        """Run force-directed layout to find compact non-overlapping configuration."""
        n = len(positions)
        pos = np.array(positions, dtype=float)
        
        for it in range(iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            forces = compute_repulsion_forces(pos, angles)
            # Damping
            pos += dt * forces
            dt *= 0.999
        
        return pos

    def minimal_enclosing_hexagon(pts, beta):
        """Find minimal enclosing regular hexagon with rotation beta."""
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # For each normal, compute max projection
        max_projs = []
        for nx, ny in normals:
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_projs.append(max(np.max(projs), -np.min(projs)))
        
        # The hexagon must contain all points
        # S = (max_proj + APOTHEM) / APOTHEM
        max_proj = max(max_projs)
        S = (max_proj + APOTHEM) / APOTHEM
        return S, max_proj

    def optimize_for_candidate(pts_initial, angles):
        """Optimize a single candidate configuration."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        # First, run force-directed layout
        pos_fd = force_directed_layout(pts_initial, angles, iterations=300)
        
        # Center the configuration
        centroid = np.mean(pos_fd, axis=0)
        pos_fd = pos_fd - centroid
        
        # Now optimize the enclosing hexagon
        def objective(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            pts = pos_fd + np.array([cx, cy])
            S, _ = minimal_enclosing_hexagon(pts, beta)
            return S
        
        # Try multiple starting points
        best_S = float('inf')
        best_params = None
        
        start_betas = [i * math.pi / 36 for i in range(6)]
        for sb in start_betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            init = [0.0, 0.0, sb]
            try:
                res = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_params is None:
            return float('inf'), None, None
        
        cx, cy, beta = best_params
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
        final_pts = pos_fd + np.array([cx, cy])
        
        return best_S, final_pts, (cx, cy, beta)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_beta = 0.0

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts_initial = axial_to_cartesian(axial)
        angles = [0.0] * N
        
        S, pts, info = optimize_for_candidate(pts_initial, angles)
        if S < best_global_S and pts is not None:
            best_global_S = S
            best_global_pts = pts
            if info is not None:
                cx, cy, beta = info
                best_global_center = (cx, cy)
                best_global_beta = beta

    # Fallback if optimization failed
    if best_global_pts is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        centroid = np.mean(best_global_pts, axis=0)
        best_global_pts = best_global_pts - centroid
        best_global_center = (0.0, 0.0)
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale slightly to ensure no overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
