import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def _smooth_objective(params, beta=10.0):
    """
    Smooth objective function for Differential Evolution.
    params: [x1, y1, ..., xN, yN, shared_angle_deg]
    Returns: Approximate outer side length + collision penalty.
    """
    centers = params[:2*N].reshape(N, 2)
    shared_angle_deg = params[2*N]
    
    # 1. Compute Enclosing Hexagon Size
    # The outer hexagon is centered at (0,0) with angle shared_angle_deg.
    # We need the minimum S such that all inner hexagons are inside.
    # A point P is inside outer hexagon (center C, side S, angle A) if:
    # dot(P-C, n_k) <= S * sqrt(3)/2 for all k=0..5
    # where n_k are unit outward normals at angles A + k*60.
    # So S >= max_{P, k} dot(P-C, n_k) * 2 / sqrt(3)
    
    # To make it smooth, we use a soft-max over the projections.
    # However, the max over vertices and normals is already a convex function of centers.
    # The non-smoothness comes from the 'max' operation.
    # We can approximate the max of a set of linear functions f_i(x) with the soft-max:
    # M = (1/beta) * log(sum(exp(beta * f_i)))
    
    # Let's gather all projections f_i
    # f_i = dot(v_{i,j}, n_k)
    # Note: The inner hexagons also rotate by shared_angle_deg? 
    # The task says "each inner hexagon may translate and rotate".
    # The strategy suggests "shared rotation angle". Let's assume all inner hexagons 
    # rotate by the same angle as the outer one for simplicity, or maybe 0 relative to outer?
    # If outer is at angle A, and inner is at angle A, they are aligned.
    # Let's assume inner angle = outer angle (aligned) for the DE search.
    # This reduces variables.
    
    inner_angle = shared_angle_deg
    
    max_proj_log_sum = 0.0
    count = 0
    
    # Precompute normals for outer hexagon
    theta_outer = np.deg2rad(shared_angle_deg)
    normals = []
    for k in range(6):
        ang = theta_outer + k * (np.pi / 3)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    
    # Precompute relative vertices for inner hexagon (side 1)
    theta_inner = np.deg2rad(inner_angle)
    rel_verts = []
    for k in range(6):
        ang = theta_inner + k * (np.pi / 3)
        rel_verts.append(np.array([np.cos(ang), np.sin(ang)]))
    
    for i in range(N):
        c = centers[i]
        for rv in rel_verts:
            v = c + rv
            for n in normals:
                proj = np.dot(v, n)
                # Soft max accumulation
                # We need to be careful with overflow.
                # Since proj can be negative, we can shift by max_val first.
                # But we are accumulating log-sum-exp.
                # Let's just collect all projs first to find max for stability.
                pass
    
    # Recalculate to find max for stability
    projs = []
    for i in range(N):
        c = centers[i]
        for rv in rel_verts:
            v = c + rv
            for n in normals:
                projs.append(np.dot(v, n))
                
    projs = np.array(projs)
    max_proj = np.max(projs)
    
    # Soft max
    exp_projs = np.exp(beta * (projs - max_proj))
    log_sum = np.log(np.sum(exp_projs))
    smooth_max_proj = max_proj + (1.0 / beta) * log_sum
    
    side = smooth_max_proj * 2.0 / np.sqrt(3.0)
    
    # 2. Collision Penalty
    # We want centers to be at least 2 * apothem = sqrt(3) apart if aligned?
    # Actually, for two aligned hexagons, the minimum distance between centers 
    # to be disjoint is the sum of their "radius" in the direction of the center line.
    # For aligned hexagons, the width is 2*side = 2? No, width across flats is sqrt(3)*side.
    # Wait, side length is 1.
    # Distance between parallel sides is sqrt(3).
    # If they are aligned, the center distance must be at least sqrt(3) + sqrt(3) = 2*sqrt(3)?
    # No. If they share a side, the distance between centers is the distance between parallel sides = sqrt(3).
    # So min distance is sqrt(3).
    # If they are rotated relative to each other, the distance might be different, but 
    # since we assume shared rotation, they are aligned.
    # So constraint: dist(c_i, c_j) >= sqrt(3).
    
    min_dist = np.sqrt(3.0)
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                # Smooth penalty: (min_dist - d)^2
                diff = min_dist - d
                penalty += diff * diff
                
    # Weight the penalty heavily to ensure disjointness
    total_penalty = 10.0 * penalty
    
    return side + total_penalty

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Bounds for DE
    # Centers: roughly in a box of size 10x10 centered at 0
    # Angle: 0 to 60 degrees (periodicity of hexagon)
    bounds = [(-5, 5)] * (2 * N) + [(0, 60)]
    
    # Strategy: Run DE with limited time
    # DE is stochastic, so we run it once with a tight time limit or maxiter.
    # We can also try a few different seeds if time permits.
    
    def _de_callback(xk, convergence=True):
        nonlocal best_side, best_inner, best_outer_center, best_outer_angle
        if time.time() - start_time > TIME_LIMIT - 5:
            raise RuntimeError("Time limit exceeded")
            
        # Evaluate current best in DE population?
        # DE minimizes the function. We can check the current xk.
        # But xk is the current individual, not necessarily the best.
        # We should evaluate the best found so far by DE.
        # However, DE doesn't easily expose the global best in callback unless we track it.
        # Let's just let DE run and evaluate the result at the end, or periodically.
        # For now, we'll just guard against timeout.
        pass

    try:
        # Use Differential Evolution
        # tol and maxiter control convergence.
        # We want to use as much time as possible.
        # DE is slow. Let's limit popsize and maxiter.
        
        # popsize=15 is default. 15 * 25 (2*N+1) = 375 evals per gen.
        # If we have 100 gens, that's 37500 evals.
        # Each eval is O(N^2 * 36) ~ 12^2 * 36 ~ 5000 ops. Very fast in numpy.
        # 37500 * 5000 ~ 1.8e8 ops. Should be < 1s.
        # So we can afford many iterations.
        
        res = differential_evolution(
            _smooth_objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            callback=_de_callback,
            polish=False  # Disable polish to save time and avoid local minima issues
        )
        
        # Evaluate the result
        if res.success or res.fun < best_side:
            params = res.x
            centers = params[:2*N].reshape(N, 2)
            shared_angle_deg = params[2*N]
            
            # Compute exact side length (hard max, not soft)
            theta_outer = np.deg2rad(shared_angle_deg)
            normals = [np.array([np.cos(theta_outer + k * np.pi/3), np.sin(theta_outer + k * np.pi/3)]) for k in range(6)]
            
            theta_inner = np.deg2rad(shared_angle_deg)
            rel_verts = [np.array([np.cos(theta_inner + k * np.pi/3), np.sin(theta_inner + k * np.pi/3)]) for k in range(6)]
            
            max_proj = -np.inf
            for i in range(N):
                c = centers[i]
                for rv in rel_verts:
                    v = c + rv
                    for n in normals:
                        proj = np.dot(v, n)
                        if proj > max_proj:
                            max_proj = proj
            
            side = max_proj * 2.0 / np.sqrt(3.0)
            
            # Check validity
            inner_hexes = [[centers[i][0], centers[i][1], shared_angle_deg] for i in range(N)]
            is_valid, reason = _check_validity(inner_hexes, side, (0.0, 0.0), shared_angle_deg)
            
            if is_valid:
                if side < best_side:
                    best_side = side
                    best_inner = inner_hexes
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = shared_angle_deg
            else:
                # If invalid, we might need to adjust. 
                # The smooth penalty might have allowed slight overlaps.
                # We can try to "fix" by increasing side slightly? No, that increases score.
                # We should try to move them apart.
                # For now, if invalid, we ignore this result.
                pass

    except Exception as e:
        pass
    
    # If DE didn't find a valid solution, try a fallback or a second run with different seed
    if best_inner is None:
        # Try a simpler initial guess optimization or just return a valid but large config
        # Let's try a quick local search from a good initial guess if time permits
        if time.time() - start_time < TIME_LIMIT - 10:
            # Re-run DE with different seed or params?
            # Or just use a known valid configuration.
            # Let's construct a valid configuration:
            # Place 12 hexagons in a 4x3 grid?
            # Spacing sqrt(3) ~ 1.732
            # Width: 3 * 1.732 = 5.196
            # Height: 2 * 1.732 = 3.464 (plus hex height?)
            # Hex height (flat to flat) is sqrt(3).
            # So 3 columns, 4 rows?
            # Let's just use a spiral or grid.
            
            pts = []
            # Grid layout
            # dx = sqrt(3), dy = 1.5? No, hex grid.
            # For aligned hexagons:
            # Neighbors at distance sqrt(3) in 6 directions.
            # Directions: 0, 60, 120, 180, 240, 300.
            # Vectors: (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
            # Wait, side=1. Apothem = sqrt(3)/2.
            # Distance between centers of touching hexagons = 2 * apothem = sqrt(3).
            
            # Let's place them in a triangular lattice.
            # (0,0)
            # (sqrt(3), 0)
            # (sqrt(3)/2, 1.5) -- wait, y-component is side * sin(60) * 2? 
            # Let's check: Vector to neighbor at 60 deg.
            # Length sqrt(3). Angle 60.
            # x = sqrt(3) * cos(60) = sqrt(3)/2
            # y = sqrt(3) * sin(60) = sqrt(3) * sqrt(3)/2 = 1.5
            # Correct.
            
            lattice_vectors = [
                (0, 0),
                (np.sqrt(3), 0),
                (np.sqrt(3)/2, 1.5),
                (-np.sqrt(3)/2, 1.5),
                (-np.sqrt(3), 0),
                (-np.sqrt(3)/2, -1.5),
                (np.sqrt(3)/2, -1.5),
                (2*np.sqrt(3), 0),
                (3*np.sqrt(3)/2, 1.5),
                (np.sqrt(3), 1.5),
                (np.sqrt(3)/2, 1.5) # duplicate?
            ]
            # Let's just generate a grid of indices
            # i in -2..2, j in -2..2
            # x = i * sqrt(3) + 0.5 * j * sqrt(3)
            # y = j * 1.5
            
            candidates = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    x = i * np.sqrt(3) + 0.5 * j * np.sqrt(3)
                    y = j * 1.5
                    candidates.append((x, y))
            
            # Pick 12 closest to origin
            candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
            pts = candidates[:12]
            
            centers = np.array(pts)
            angle = 0.0
            
            # Compute side
            theta_outer = 0
            normals = [np.array([np.cos(k * np.pi/3), np.sin(k * np.pi/3)]) for k in range(6)]
            rel_verts = [np.array([np.cos(k * np.pi/3), np.sin(k * np.pi/3)]) for k in range(6)]
            
            max_proj = -np.inf
            for i in range(N):
                c = centers[i]
                for rv in rel_verts:
                    v = c + rv
                    for n in normals:
                        proj = np.dot(v, n)
                        if proj > max_proj:
                            max_proj = proj
            side = max_proj * 2.0 / np.sqrt(3.0)
            
            inner_hexes = [[centers[i][0], centers[i][1], angle] for i in range(N)]
            is_valid, reason = _check_validity(inner_hexes, side, (0.0, 0.0), angle)
            
            if is_valid and side < best_side:
                best_side = side
                best_inner = inner_hexes
                best_outer_center = (0.0, 0.0)
                best_outer_angle = angle

    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
