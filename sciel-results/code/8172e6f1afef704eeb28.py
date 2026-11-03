import math
import numpy as np
import time
from scipy.optimize import minimize_scalar
from itertools import combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list, scale=1.0):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x * scale, y * scale))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """
        Calculate the minimum side length S of a regular hexagon with rotation beta
        that encloses all points in pts.
        Returns S, center (cx, cy).
        """
        # Normals of the outer hexagon
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Project points onto normals
        # projs[i, k] is projection of point i onto normal k
        projs = pts @ normals.T
        
        # For each normal k, find min and max projection
        mins = np.min(projs, axis=0)
        maxs = np.max(projs, axis=0)
        
        # The width in direction k is maxs[k] - mins[k]
        # The half-width is (maxs[k] - mins[k]) / 2
        # The required apothem R is the maximum of these half-widths
        half_widths = (maxs - mins) / 2.0
        R = float(np.max(half_widths))
        
        # The center of the hexagon must be such that the projections of the points
        # are centered within [-R, R] relative to the center.
        # For a specific normal k, the center's projection c_k must satisfy:
        # mins[k] - c_k >= -R  => c_k <= mins[k] + R
        # maxs[k] - c_k <= R   => c_k >= maxs[k] - R
        # So c_k must be in [maxs[k] - R, mins[k] + R].
        
        # We need to find a point (cx, cy) whose projections onto all normals fall into these intervals.
        # Since the hexagon is convex and symmetric, the center of the bounding box of the points
        # (in the sense of the Chebyshev distance in the hexagonal metric) is a good candidate.
        # A simple and effective heuristic for the center of the minimal enclosing hexagon
        # for a fixed orientation is the centroid of the points, or the midpoint of the overall min/max.
        # However, to be precise, we can solve for the center that minimizes R, but here R is fixed by the orientation.
        # Any center that satisfies the constraints works. The "center" of the feasible region for the center
        # is often the average of the midpoints of the min/max projections?
        
        # Let's just pick the center as the centroid of the points. 
        # Wait, if we pick the centroid, the R calculated above might not be sufficient if the centroid is not the optimal center for that R.
        # Actually, R = max half_width is the *minimum* possible apothem for that orientation.
        # The center must be chosen such that all points are within distance R (in the normal directions).
        # Let's find the center that minimizes the maximum deviation from the origin in these projections?
        # No, we just need *a* valid center.
        # The intersection of the strips defined by the normals is non-empty if R is large enough.
        # We can find the center by averaging the "ideal" centers for each direction?
        
        # A robust way: The center (cx, cy) should be the centroid of the points for simplicity, 
        # BUT we must verify if R is sufficient for that center.
        # If we use the centroid, the effective R might be larger.
        # To get the *true* minimal S for this beta, we should find the center that minimizes R.
        # But R is already the minimal possible half-width. The center just needs to be in the feasible region.
        # Let's compute the feasible region for cx, cy.
        # This is a small LP. But we want speed.
        
        # Alternative: Just use the centroid. If the centroid is not in the feasible region for R, 
        # it means R was calculated correctly as the max half-width, but the centroid might be off-center 
        # such that it requires a larger R.
        # Actually, the minimal enclosing hexagon for a fixed orientation is unique in terms of size R.
        # The center is not unique if the set of points doesn't "touch" all 6 sides.
        # But usually, we can just take the midpoint of the global min and max of the projections?
        
        # Let's refine: The optimal center is the one that minimizes the max projection.
        # For a fixed orientation, the problem reduces to finding the center of the smallest axis-aligned box 
        # in the transformed coordinate system (projections).
        # The projections are coupled (not independent axes).
        
        # Let's just use the centroid of the points as the center. 
        # And recalculate the required R for that specific center.
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        center = np.array([cx, cy])
        
        pts_centered = pts - center
        projs_centered = pts_centered @ normals.T
        R_actual = float(np.max(np.abs(projs_centered)))
        
        S = (R_actual + APOTHEM) / APOTHEM
        return S, (cx, cy)

    def optimize_rotation(pts):
        """
        Find the rotation beta that minimizes S.
        """
        def obj(beta):
            S, _ = calc_S_for_beta(pts, beta)
            return S
        
        # Beta is periodic with pi/3 (60 degrees).
        # We can search in [0, pi/3].
        res = minimize_scalar(obj, bounds=(0, math.pi / 3), method='bounded', options={'xatol': 1e-6})
        if res.success:
            return res.fun, res.x
        else:
            # Fallback to grid
            best_S = float('inf')
            best_beta = 0.0
            for i in range(30):
                beta = i * (math.pi / 3) / 30
                S, _ = calc_S_for_beta(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
            return best_S, best_beta

    # --- Generate candidate configurations ---
    candidates = []
    seen = set()
    
    # We want to ensure disjointness. 
    # Standard axial grid has touching hexagons.
    # We will scale the Cartesian coordinates of the centers by a factor > 1.
    # Let's use 1.02 to be safe.
    SCALE_FACTOR = 1.02

    def generate_greedy_candidates():
        import random
        rng = random.Random(42)
        all_axial = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                if abs(q + r) <= 3:
                    all_axial.append((q, r))
        
        all_axial.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        remaining = [p for p in all_axial if p != (0,0)]
        
        for _ in range(20): # Reduced iterations for speed
            if time.time() - START_TIME > TIME_LIMIT - 30.0:
                break
            current = [(0,0)]
            rem = list(remaining)
            rng.shuffle(rem)
            
            for _ in range(N-1):
                if not rem:
                    break
                best_cand = None
                best_score = float('inf')
                # Check closest few
                for cand in rem[:10]:
                    test_set = current + [cand]
                    # Quick score: bounding box area in cartesian
                    pts = axial_to_cartesian(test_set, scale=1.0) # Scale doesn't affect relative ordering
                    w = pts[:, 0].max() - pts[:, 0].min()
                    h = pts[:, 1].max() - pts[:, 1].min()
                    score = w * h
                    if score < best_score:
                        best_score = score
                        best_cand = cand
                
                if best_cand:
                    current.append(best_cand)
                    rem.remove(best_cand)
            
            if len(current) == N:
                key = tuple(sorted(current))
                if key not in seen:
                    seen.add(key)
                    candidates.append(current)

    generate_greedy_candidates()

    # Add a few known good patterns (compact clusters)
    hand_picked = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,-1),(1,1),(-1,2),(2,0)]
    ]
    for cfg in hand_picked:
        key = tuple(sorted(cfg))
        if key not in seen:
            seen.add(key)
            candidates.append(cfg)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by bounding box
    def bbox_score(cfg):
        pts = axial_to_cartesian(cfg, scale=1.0)
        return (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())

    candidates.sort(key=bbox_score)

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Apply scaling to ensure disjointness
        pts = axial_to_cartesian(axial, scale=SCALE_FACTOR)
        
        S, beta = optimize_rotation(pts)
        
        if S < best_global_S:
            # Get the center for this best beta
            _, center = calc_S_for_beta(pts, beta)
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0], scale=SCALE_FACTOR)
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        best_global_S = 10.0 # Large fallback

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
