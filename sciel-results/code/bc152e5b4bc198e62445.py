import numpy as np
import time
import itertools
from scipy.optimize import minimize
import cvxpy as cp

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations of indices
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    # Area = 0.5 * |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _min_area_and_indices(points):
    """Return (min_area, list of indices of points involved in min triangles)."""
    areas = _compute_areas(points)
    min_area = float(np.min(areas))
    # Find all triangles with area close to min
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    # Collect unique point indices involved
    involved = set()
    for idx in min_indices:
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        # u + v = 1 - y/H
        # v - u = 2*x - 1
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points, ensuring validity."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _random_valid_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.empty((N, 2))
    for m in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[m, 0] = u
                bary[m, 1] = v
                break
    return bary

def _local_nudge_search(points, max_iter=500, time_limit=10.0, rng=None):
    """
    Local search that only moves points involved in the smallest triangles.
    Uses targeted perturbations to break near-degeneracies.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    current_points = points.copy()
    current_min, _ = _min_area_and_indices(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    start_t = time.time()
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
            
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            min_area, involved = _min_area_and_indices(current_points)
            
            if min_area > best_min:
                best_min = min_area
                best_points = current_points.copy()
                
            if len(involved) == 0:
                break
                
            # Try nudging each involved point in random directions
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                    
                # Try several random perturbations for this point
                for _ in range(20):
                    du = rng.uniform(-step_size, step_size)
                    dv = rng.uniform(-step_size, step_size)
                    
                    bary = _cart_to_bary(current_points)
                    new_u = bary[idx, 0] + du
                    new_v = bary[idx, 1] + dv
                    
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                        
                    new_x = new_v * 1.0 + (1.0 - new_u - new_v) * 0.5
                    new_y = (1.0 - new_u - new_v) * H
                    
                    # Check if inside triangle
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                        
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        min_area = test_min
                        improved = True
                        break
                        
                if improved:
                    break
                    
            if not improved:
                break
                
    return best_points, best_min

def _solve_sdp():
    """
    Solve the SDP relaxation of the Heilbronn problem.
    Maximizes t such that area(i,j,k) >= t for all triangles.
    Area is 0.5 * |det|. We relax |det| >= 2t to det >= 2t OR det <= -2t.
    Since we can't disjunction in SDP, we maximize the minimum of the signed areas
    after aligning orientations? No, that's not right.
    
    Alternative: Use the fact that Area = 0.5 * || p_i x p_j + p_j x p_k + p_k x p_i ||? No.
    
    Let's use the Gram matrix approach.
    Let X be the (N+1) x (N+1) Gram matrix of points in 2D (plus a constant term for affine).
    Actually, standard SDP for TSP-like or packing problems uses:
    Maximize t
    Subject to:
      X >= 0
      X[i,i] = 1 (or fixed)
      Tr(X) = N
      Area constraints: 
        The area of triangle (i,j,k) is 0.5 * | (p_j - p_i) x (p_k - p_i) |.
        In terms of squared distances d_ij = ||p_i - p_j||^2:
        16 * Area^2 = 2(d_ij^2 + d_jk^2 + d_ki^2) - (d_ij^4 + d_jk^4 + d_ki^4)
        This is non-linear in d.
        
        However, we can linearize if we fix the "orientation". 
        If we assume the points are in general position, we can try to maximize the minimum signed area.
        But the problem is symmetric under reflection, so the optimal configuration might have mixed signs.
        
        A common relaxation is to maximize t such that:
        det(p_i, p_j, p_k) >= t  AND  det(p_i, p_j, p_k) <= -t is NOT possible simultaneously.
        
        Let's try a simpler heuristic SDP:
        Maximize t
        Subject to:
        (p_j - p_i) x (p_k - p_i) >= 2t  for all (i,j,k) in a specific orientation?
        This forces all triangles to have the same orientation, which might be restrictive but yields a valid lower bound.
        
        Variables: p_i (2D).
        Constraints:
        p_i inside triangle.
        (p_j - p_i) x (p_k - p_i) >= 2t for all i<j<k? 
        This implies all triangles are oriented CCW. This is a convex problem (linear constraints in p_i and t).
        Wait, (p_j - p_i) x (p_k - p_i) is linear in p_i, p_j, p_k.
        Yes! This is a Linear Program (LP) if we fix the orientation.
        But is the optimal Heilbronn configuration "all CCW"? No, because points can be anywhere.
        However, if we solve the LP with "all CCW" constraint, we get a feasible solution for the original problem (since |area| >= area if area > 0).
        Actually, if area is negative, |area| is positive. The constraint area >= 2t ensures area is positive and large.
        This might be too restrictive.
        
        Better: Use SDP with the squared area constraint?
        16 A^2 >= (2t)^2 = 4t^2.
        16 A^2 = 2(sum d^2) - sum d^4.
        This is not convex in d.
        
        Let's stick to the LP relaxation with fixed orientation. It's fast and provides a structured start.
        If the LP fails or gives a bad result, we fallback to random.
    """
    try:
        # Variables
        P = cp.Variable((N, 2))
        t = cp.Variable()
        
        # Objective
        objective = cp.Maximize(t)
        
        # Constraints
        constraints = [t >= 0]
        
        # Triangle containment constraints (Barycentric)
        # p = u*V0 + v*V1 + w*V2, u+v+w=1, u,v,w>=0
        # V0=(0,0), V1=(1,0), V2=(0.5, H)
        # x = v + 0.5w
        # y = Hw
        # w = y/H
        # v = x - 0.5w = x - y/(2H)
        # u = 1 - v - w = 1 - x + y/(2H) - y/H = 1 - x - y/(2H)
        
        for i in range(N):
            constraints += [
                P[i, 1] >= 0,
                P[i, 1] <= H,
                P[i, 0] >= 0,
                P[i, 0] <= 1,
                # y <= sqrt(3) x  => x - y/sqrt(3) >= 0
                P[i, 0] - P[i, 1]/SQRT3 >= -1e-9,
                # y <= sqrt(3)(1-x) => x + y/sqrt(3) <= 1
                P[i, 0] + P[i, 1]/SQRT3 <= 1 + 1e-9
            ]
            
        # Orientation constraints: (P[j]-P[i]) x (P[k]-P[i]) >= 2t
        # Cross product in 2D: (x1*y2 - y1*x2)
        # (P[j,0]-P[i,0])*(P[k,1]-P[i,1]) - (P[j,1]-P[i,1])*(P[k,0]-P[i,0]) >= 2t
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    cp_x = (P[j,0]-P[i,0])*(P[k,1]-P[i,1]) - (P[j,1]-P[i,1])*(P[k,0]-P[i,0])
                    constraints += [cp_x >= 2*t]
                    
        prob = cp.Problem(objective, constraints)
        
        # Solve with a fast solver
        prob.solve(solver=cp.SCS, verbose=False)
        
        if P.value is not None and t.value is not None and t.value > 0:
            pts = np.array(P.value)
            # Clean up numerical noise
            pts = np.clip(pts, 0, 1)
            # Re-clip to triangle
            for i in range(N):
                x, y = pts[i]
                if y < 0: y = 0
                if y > H: y = H
                if x < 0: x = 0
                if x > 1: x = 1
                if y > SQRT3 * x:
                    # Project to edge y = sqrt(3)x
                    # x' = (x + y/sqrt(3)) / (1 + 1/sqrt(3))? No, simple clip might be enough for start
                    pass
                if y > SQRT3 * (1 - x):
                    pass
            return pts, float(t.value)
    except Exception:
        pass
    return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: SDP Relaxation for Global Structure
    # The LP relaxation with fixed orientation gives a valid configuration where all triangles are CCW.
    # This is a strong structural prior.
    sdp_pts, sdp_t = _solve_sdp()
    if sdp_pts is not None:
        # Verify and score
        # Ensure strict validity
        valid = True
        for p in sdp_pts:
            x, y = p
            if y < -1e-6 or y > H + 1e-6 or x < -1e-6 or x > 1 + 1e-6:
                valid = False
            if y > SQRT3 * x + 1e-6 or y > SQRT3 * (1 - x) + 1e-6:
                valid = False
        if valid:
            score = _min_area(sdp_pts)
            if score > best_score:
                best_points = sdp_pts
                best_score = score
    
    # Fallback if SDP fails or is too slow
    if best_points is None or time.time() > start_time + 10:
        # Use a good structured guess
        positions = [
            (0.1, 0.1), (0.4, 0.1), (0.7, 0.1),
            (0.2, 0.3), (0.5, 0.3), (0.8, 0.3),
            (0.15, 0.5), (0.45, 0.5), (0.75, 0.5),
            (0.3, 0.7), (0.6, 0.7),
        ]
        bary = np.empty((N, 2))
        for m, (u, v) in enumerate(positions):
            if u + v > 1.0:
                s = 1.0 / (u + v)
                u *= s
                v *= s
            bary[m, 0] = u
            bary[m, 1] = v
        best_points = _bary_to_cart(bary)
        best_score = _min_area(best_points)

    # Phase 2: Local Refinement from SDP/Structured Start
    # The SDP solution is likely "stiff" due to the orientation constraints.
    # Local search will relax these implicit constraints by moving points.
    
    if time.time() < deadline - 20:
        nudged_pts, nudged_score = _local_nudge_search(best_points, max_iter=1000, time_limit=20.0, rng=rng)
        if nudged_score > best_score:
            best_points = nudged_pts
            best_score = nudged_score

    # Phase 3: Multiple Random Restarts + Local Search
    # If we have time, explore other basins.
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        random_bary = _random_valid_barycentric(rng)
        random_pts = _bary_to_cart(random_bary)
        
        # Quick local search on random start
        if time.time() < deadline - 15:
            nudged_pts, nudged_score = _local_nudge_search(random_pts, max_iter=200, time_limit=5.0, rng=rng)
            if nudged_score > best_score:
                best_points = nudged_pts
                best_score = nudged_score

    # Final validation
    if best_points is None:
        bary = _random_valid_barycentric(rng)
        best_points = _bary_to_cart(bary)
        
    # Strictly enforce containment
    final_bary = _cart_to_bary(best_points)
    final_points = _bary_to_cart(final_bary)
    
    return final_points
