import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets in axial coordinates
    # (q, r) -> x = sqrt(3)*(q + r/2), y = 1.5*r
    
    candidates = []
    
    # 1. Ring (7) + 4 outer (diagonal-ish)
    # The 4 extra are placed at (1,1), (-1,2), (-2,1), (2,-1) relative to center?
    # Let's try standard "Ring + 4 corners"
    # Ring: (0,0) + 6 neighbors
    # Extra: (1,1), (-1,2), (-2,1), (2,-1) -- wait, these are not disjoint from ring?
    # Ring neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (1,1) is adjacent to (1,0) and (0,1). It fits in the gap.
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ])
    
    # 2. Ring (7) + 4 cardinal (further out)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])
    
    # 3. 3x3 Block + 2 extra
    # 3x3 block: q in 0..2, r in 0..2 (9 hexes)
    # Extra: (-1, 1), (3, 1) ? Or (0, -1) and (0, 3)?
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ])
    
    # 4. 3x3 Block + 2 top/bottom
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 5. 4x2 Block + 3
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 6. Staircase
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S(pts, cx, cy, beta):
        """
        Calculate the side length S of the outer hexagon required to contain
        the points 'pts' given center (cx, cy) and rotation beta.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        # Project points onto normals
        projs = pts_centered @ normals.T
        # Max absolute projection
        max_proj = np.max(np.abs(projs))
        
        # Side length S = (D + apothem) / apothem, where D is max distance from center to boundary
        # D = max_proj
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def objective(params, pts):
        cx, cy, beta = params
        # Normalize beta to [0, pi/6)
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
        return calc_S(pts, cx, cy, beta)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_beta = 0.0

    # Process candidates
    for i, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        
        # Initial guess: center of mass, rotation 0
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        beta0 = 0.0
        
        # Quick check with initial guess
        S_init = calc_S(pts, cx0, cy0, beta0)
        
        # Try a few rotations quickly
        best_local_S = S_init
        best_local_beta = beta0
        for b in [0.0, math.pi/12, math.pi/6, math.pi/24, 3*math.pi/24]:
            S_b = calc_S(pts, cx0, cy0, b)
            if S_b < best_local_S:
                best_local_S = S_b
                best_local_beta = b
        
        # Refine with Nelder-Mead
        # Parameters: cx, cy, beta
        init = [cx0, cy0, best_local_beta]
        
        try:
            res = minimize(objective, init, args=(pts,), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_local_S:
                cx_r, cy_r, beta_r = res.x
                beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                best_local_S = res.fun
                best_local_center = (cx_r, cy_r)
                best_local_beta = beta_r
            else:
                best_local_center = (cx0, cy0)
        except Exception:
            best_local_center = (cx0, cy0)
            best_local_beta = best_local_beta

        if best_local_S < best_global_S:
            best_global_S = best_local_S
            best_global_pts = pts
            best_global_center = best_local_center
            best_global_beta = best_local_beta

    # Fallback if no valid solution found
    if best_global_center is None or best_global_S == float('inf'):
        pts = axial_to_cartesian(candidates[0])
        best_global_pts = pts
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        best_global_S = calc_S(pts, best_global_center[0], best_global_center[1], best_global_beta)

    # Final Polish: High precision Nelder-Mead on the best candidate
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        init = [best_global_center[0], best_global_center[1], best_global_beta]
        try:
            res = minimize(objective, init, args=(best_global_pts,), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_global_S:
                cx_p, cy_p, beta_p = res.x
                beta_p = ((beta_p % (math.pi/6)) + math.pi/6) % (math.pi/6)
                best_global_S = res.fun
                best_global_center = (cx_p, cy_p)
                best_global_beta = beta_p
        except Exception:
            pass

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Construct inner hexagons
    # The points in best_global_pts are the centers of the unit hexagons.
    # All inner hexagons have the same rotation angle as the outer hexagon?
    # The problem states: "each inner hexagon may translate and rotate".
    # However, for optimal packing of regular hexagons, they usually align.
    # The interface requires: inner rows (x, y, angle_degrees).
    # We assume all inner hexagons have the same rotation as the outer one for simplicity,
    # which is valid for the standard symmetric packings.
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

# Test execution
if __name__ == "__main__":
    inner, center, side, rotation = construct()
    print(f"Side: {side}, Center: {center}, Rotation: {rotation}")
