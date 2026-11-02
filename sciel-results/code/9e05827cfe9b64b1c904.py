import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hex_vertices(center, angle_deg, scale=1.0):
        """Return the 6 vertices of a regular hexagon with given center, rotation, and scale."""
        cx, cy = center
        theta = math.radians(angle_deg)
        verts = []
        for i in range(6):
            ang = theta + i * math.pi / 3.0
            x = cx + scale * math.cos(ang)
            y = cy + scale * math.sin(ang)
            verts.append((x, y))
        return np.array(verts)

    def get_all_vertices(positions, angles):
        """Return array of all vertices for all hexagons."""
        all_verts = []
        for i in range(len(positions)):
            cx, cy = positions[i]
            ang = angles[i]
            verts = hex_vertices((cx, cy), ang)
            all_verts.append(verts)
        return np.vstack(all_verts)

    def compute_outer_side_length(positions, angles, outer_center, outer_angle_deg):
        """
        Compute the minimal side length of the outer hexagon that encloses all inner hexagons.
        positions: Nx2 array of centers
        angles: N array of angles in degrees
        outer_center: (cx, cy)
        outer_angle_deg: rotation of outer hexagon
        """
        all_verts = get_all_vertices(positions, angles)
        ocx, ocy = outer_center
        otheta = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for k in range(6):
            ang = otheta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            # Project all vertices onto this normal
            projs = (all_verts[:, 0] - ocx) * nx + (all_verts[:, 1] - ocy) * ny
            max_abs = np.max(np.abs(projs))
            if max_abs > max_proj:
                max_proj = max_abs
        
        # Side length S such that apothem A_outer = max_proj + A_inner (since inner hex apothem is APOTHEM)
        # Wait, the inner hexagons have side length 1, so their apothem is APOTHEM.
        # The outer hexagon must contain them.
        # Distance from outer center to outer edge = A_outer = (S * sqrt(3)) / 2
        # So S = 2 * A_outer / sqrt(3) = 2 * (max_proj + APOTHEM) / sqrt(3)
        # Note: APOTHEM = sqrt(3)/2, so 2/sqrt(3) = 2 / (2*APOTHEM) = 1/APOTHEM? No.
        # A_outer = max_proj + APOTHEM
        # S = A_outer / APOTHEM * 1? No.
        # A = S * sqrt(3) / 2 => S = 2 * A / sqrt(3)
        # So S = 2 * (max_proj + APOTHEM) / sqrt(3)
        return 2.0 * (max_proj + APOTHEM) / math.sqrt(3.0)

    def overlap_penalty(positions, angles):
        """
        Calculate a penalty for overlapping inner hexagons.
        Approximate by checking center distances.
        For two regular hexagons of side 1, the minimum center distance to avoid overlap
        is sqrt(3) if they are parallel and touching flat sides.
        If rotated, it can be smaller, but for a simple penalty, we use a threshold.
        Actually, the minimum distance between centers of two non-overlapping unit hexagons
        is sqrt(3) (if aligned). If one is rotated 30 deg, the distance can be smaller?
        No, the circumradius is 1, inradius is sqrt(3)/2.
        Max distance from center to boundary is 1.
        So if center distance > 2, they definitely don't overlap.
        If center distance < sqrt(3), they might overlap if aligned.
        We use a smooth penalty: sum of (threshold - dist)^2 for dist < threshold.
        Threshold = sqrt(3) is the tightest for parallel.
        """
        penalty = 0.0
        n = len(positions)
        threshold = math.sqrt(3.0) # Minimum distance for parallel hexagons
        # A safer threshold considering rotation might be slightly larger or we rely on NLP to fix it.
        # Let's use a slightly larger threshold to be safe during relaxation, e.g., 1.8
        # But for the final objective, we need strict non-overlap.
        # We will use the NLP to minimize the outer size, and add a large penalty for overlap.
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < threshold:
                    # Penalty grows quadratically
                    penalty += (threshold - dist) ** 2
        return penalty

    def objective(params):
        """
        Objective function for NLP.
        params: [x1, y1, x2, y2, ..., xN, yN, ocx, ocy, oangle]
        """
        n = N
        pos_flat = params[:2*n]
        positions = pos_flat.reshape(n, 2)
        angles = np.zeros(n) # Assume inner hexagons are parallel for now to simplify?
        # The task allows rotation. But usually, parallel is optimal for hexagonal packing.
        # We can fix inner angles to 0 to reduce search space, as rotating individual hexagons
        # in a tight pack often increases the required outer size or complicates packing.
        # Let's fix inner angles to 0 for the initial NLP run.
        
        ocx = params[2*n]
        ocy = params[2*n+1]
        oangle = params[2*n+2]
        
        # Compute outer side length
        S = compute_outer_side_length(positions, angles, (ocx, ocy), oangle)
        
        # Compute overlap penalty
        pen = overlap_penalty(positions, angles)
        
        # Total cost
        return S + 100.0 * pen

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate initial configurations
    candidates = []
    
    # Compact hexagonal cluster (radius 2) - 19 hexes, we need 11.
    # We take a subset of the radius 2 cluster.
    # Radius 2 axial coords:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,1)
    # Total 19. We pick 11.
    # A good subset is the center + 6 neighbors + 4 from the next ring.
    
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_2)
    
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    best_S = float('inf')
    best_params = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        
        # Center the points
        centroid = np.mean(pts, axis=0)
        pts -= centroid
        
        # Initial angles: all 0
        angles = np.zeros(N)
        
        # Initial outer center: (0,0), angle: 0
        initial_params = np.concatenate([pts.flatten(), [0.0, 0.0, 0.0]])
        
        # Try a few local optimizations from this start
        try:
            res = minimize(objective, initial_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_S:
                # Check if penalty is small enough (valid solution)
                pos_flat = res.x[:2*N]
                positions = pos_flat.reshape(N, 2)
                angles = np.zeros(N)
                pen = overlap_penalty(positions, angles)
                if pen < 1e-4:
                    best_S = res.fun
                    best_params = res.x
        except Exception as e:
            pass

    # If NLP failed or didn't find a good solution, fallback to the best initial config
    if best_params is None:
        # Use the first candidate
        pts = axial_to_cartesian(candidates[0])
        centroid = np.mean(pts, axis=0)
        pts -= centroid
        angles = np.zeros(N)
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        S = compute_outer_side_length(pts, angles, outer_center, outer_angle)
        best_params = np.concatenate([pts.flatten(), [0.0, 0.0, 0.0]])
        best_S = S

    # Extract best solution
    pos_flat = best_params[:2*N]
    positions = pos_flat.reshape(N, 2)
    angles = np.zeros(N)
    ocx = best_params[2*N]
    ocy = best_params[2*N+1]
    oangle = best_params[2*N+2]
    
    # Final refinement of outer hexagon for the fixed inner positions
    # We can optimize just the outer center and angle to minimize S
    def outer_only_objective(outer_params):
        ocx, ocy, oangle = outer_params
        return compute_outer_side_length(positions, angles, (ocx, ocy), oangle)
    
    try:
        res_outer = minimize(outer_only_objective, [ocx, ocy, oangle], method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res_outer.fun < best_S:
            ocx, ocy, oangle = res_outer.x
            best_S = res_outer.fun
    except Exception:
        pass

    # Construct return values
    inner = []
    for i in range(N):
        x, y = positions[i]
        ang = angles[i]
        inner.append([float(x), float(y), float(ang)])
        
    center = [float(ocx), float(ocy)]
    side = float(best_S)
    rotation = float(oangle)
    
    return inner, center, side, rotation
