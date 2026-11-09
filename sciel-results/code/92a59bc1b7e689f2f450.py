"""Hexagon packing: place 11 unit regular hexagons to minimize enclosing hexagon side."""
import math
import time
import numpy as np

N = 11
_START = time.time()
_DEADLINE = _START + 130.0   # safety margin before the 150 s kill

# Unit regular hexagon vertices (flat-top), circumradius = 1.
_BASE = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

# Three edge-normal directions (30, 90, 150 deg) for the enclosing-hexagon width.
_NORM = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in (30.0, 90.0, 150.0)])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _hex_vertices(cx, cy, angle):
    """Return 6 vertices of a unit hexagon centered at (cx, cy) rotated by angle radians."""
    R = _rot(angle)
    return np.array([[cx, cy]] * 6) + _BASE @ R.T


def _sat_collision(V, W):
    """
    Check if two convex hexagons overlap using SAT.
    Returns (penetration_depth, min_axis) or (0, None) if separated.
    penetration_depth > 0 means overlap; the min_axis is the direction of minimum separation.
    """
    min_pen = 1e18
    min_axis = None
    
    # Helper to compute projection range of polygon onto a direction
    def proj_range(verts, dir_vec):
        dots = verts @ dir_vec
        return dots.min(), dots.max()
    
    # Get all 12 edge normals (6 from each hexagon)
    for poly in (V, W):
        for i in range(6):
            e = poly[(i+1) % 6] - poly[i]
            # Normal perpendicular to edge
            n = np.array([-e[1], e[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            
            # Project both polygons onto this normal
            r1_min, r1_max = proj_range(V, n)
            r2_min, r2_max = proj_range(W, n)
            
            # Check for overlap: if intervals overlap, compute penetration
            overlap = min(r1_max, r2_max) - max(r1_min, r2_min)
            if overlap < 0:
                # Separated along this axis
                gap = -overlap
                if gap < min_pen:
                    min_pen = gap
                    min_axis = n
            else:
                # Overlapping along this axis; track minimum overlap
                if overlap < min_pen:
                    min_pen = overlap
                    min_axis = n
    
    # If min_pen > 0, polygons are separated
    if min_pen > 0:
        return 0.0, None
    
    # Overlap: return penetration depth and the axis of minimum separation
    return abs(min_pen), min_axis


def _width(v):
    """Width of a point set along the 3 enclosing-hexagon normal directions."""
    w = []
    for n in _NORM:
        p = np.dot(v, n)
        w.append(p.max() - p.min())
    return w


def _enclosing_side(v):
    return max(_width(v)) / math.sqrt(3.0)


def _eval(params):
    n = N
    verts = np.empty((n, 6, 2))
    centers = np.empty((n, 2))
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        angle = params[3 * i + 2]
        centers[i] = [cx, cy]
        verts[i] = _hex_vertices(cx, cy, angle)
    # enclosing side
    side = _enclosing_side(verts.reshape(-1, 2))
    # collision penalty: sum over pairs of penetration^2
    pen = 0.0
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            vj = verts[j]
            depth, axis = _sat_collision(vi, vj)
            if depth > 0:
                pen += depth ** 2
    return side, pen


def _initial_config(seed=7):
    """Generate an initial hexagonal cluster configuration."""
    rng = np.random.default_rng(seed)
    theta_g = float(rng.uniform(0.0, math.pi / 3.0))
    R = _rot(theta_g)
    centers = []
    # central hexagon
    centers.append(np.array([0.0, 0.0]))
    # 6 neighbours at distance 2 (edge-to-edge touching)
    for k in range(6):
        ang = math.radians(60.0 * k + 30.0)
        centers.append(2.0 * np.array([math.cos(ang), math.sin(ang)]))
    # 4 satellites at the 4 "corner" gaps (radius ~3.46, angles 0/90/180/270)
    for k in range(4):
        ang = math.radians(90.0 * k)
        centers.append(3.464 * np.array([math.cos(ang), math.sin(ang)]))
    centers = np.array(centers)
    centers = centers @ R.T  # apply global rotation
    # local rotations: small random
    thetas = rng.uniform(-0.05, 0.05, size=N)
    params = np.empty(3 * N + 1)
    for i in range(N):
        params[3 * i] = centers[i][0]
        params[3 * i + 1] = centers[i][1]
        params[3 * i + 2] = thetas[i]
    params[3 * N] = theta_g
    params += rng.normal(0.0, 0.02, size=params.shape)
    return params


def _penalty_gradient(params):
    """
    Compute gradient of the collision penalty w.r.t. parameters.
    For each colliding pair, push apart along the SAT minimum axis.
    """
    n = N
    grad = np.zeros_like(params)
    verts = np.empty((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(params[3*i], params[3*i+1], params[3*i+2])
    
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            vj = verts[j]
            depth, axis = _sat_collision(vi, vj)
            if depth > 0 and axis is not None:
                # Gradient: push i and j apart along axis
                # The axis points from i to j (or vice versa)
                # We need to determine direction: project centers
                ci = np.array([params[3*i], params[3*i+1]])
                cj = np.array([params[3*j], params[3*j+1]])
                d = cj - ci
                # If axis dot d < 0, flip axis so it points from i to j
                if np.dot(axis, d) < 0:
                    axis = -axis
                
                # Each hexagon moves half the penetration depth along axis
                push = depth * 0.5
                grad[3*i] -= push * axis[0]
                grad[3*i+1] -= push * axis[1]
                grad[3*j] += push * axis[0]
                grad[3*j+1] += push * axis[1]
    
    return grad


def _pack():
    """Optimize 11 hexagons using simulated annealing with SAT-based collision resolution."""
    best_side = 1e18
    best_pen = 1e18
    best_params = None
    rng = np.random.default_rng(7)
    
    # Multiple restarts with different initial configurations
    num_restarts = 5
    per_restart_time = (_DEADLINE - time.time()) / max(1, num_restarts)
    
    for restart in range(num_restarts):
        if time.time() > _DEADLINE:
            break
        
        # Generate initial configuration with different seed
        params = _initial_config(seed=7 + restart * 13)
        
        # Simulated annealing
        T = 0.5  # initial temperature
        T_min = 0.001
        cooling = 0.995
        step_base = 0.1
        
        cur_side, cur_pen = _eval(params)
        cur_cost = cur_side + 100.0 * cur_pen
        
        # Track best valid solution
        if cur_pen <= 1e-6 and cur_side < best_side:
            best_side = cur_side
            best_pen = cur_pen
            best_params = params.copy()
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < _DEADLINE and no_improve < max_no_improve:
            # Pick a random coordinate to nudge
            idx = int(rng.integers(0, params.shape[0]))
            old = params[idx]
            
            # Adaptive step size based on temperature
            step = step_base * (0.5 + T)
            delta = float(rng.normal(0.0, step))
            params[idx] = old + delta
            
            if idx == 3 * N:  # keep global rotation in [0, pi/3)
                params[idx] = params[idx] % (math.pi / 3.0)
            
            # Evaluate
            ns, npen = _eval(params)
            ncost = ns + 100.0 * npen
            
            # Metropolis acceptance
            if ncost <= cur_cost:
                accept = True
            else:
                delta_cost = ncost - cur_cost
                if T > T_min:
                    accept = (rng.random() < math.exp(-delta_cost / T))
                else:
                    accept = False
            
            if accept:
                cur_cost, cur_side, cur_pen = ncost, ns, npen
                no_improve = 0
                # Cool down
                T = max(T_min, T * cooling)
                
                # Accept into global best if valid (no collision)
                if npen <= 1e-6 and ns < best_side:
                    best_side = ns
                    best_pen = npen
                    best_params = params.copy()
            else:
                params[idx] = old
                no_improve += 1
                # Occasionally reheat to escape local minima
                if no_improve % 500 == 0 and T > T_min:
                    T = max(T_min, T * 2.0)
        
        # Gradient-based polishing on the best params from this restart
        if best_params is not None and time.time() < _DEADLINE:
            params = best_params.copy()
            step = 0.02
            cur_side, cur_pen = _eval(params)
            cur_cost = cur_side + 100.0 * cur_pen
            no_improve = 0
            
            # Use gradient to guide moves
            while time.time() < _DEADLINE and no_improve < 500:
                # Compute penalty gradient
                grad = _penalty_gradient(params)
                
                # Pick a coordinate with large gradient magnitude or random
                if rng.random() < 0.7 and np.any(np.abs(grad) > 1e-4):
                    # Pick from top gradient coordinates
                    top_idx = np.argsort(np.abs(grad))[-10:]
                    idx = int(top_idx[rng.integers(0, len(top_idx))])
                    # Move against gradient
                    delta = -np.sign(grad[idx]) * step * (1.0 + abs(grad[idx]))
                else:
                    idx = int(rng.integers(0, params.shape[0]))
                    delta = float(rng.normal(0.0, step))
                
                old = params[idx]
                params[idx] = old + delta
                
                if idx == 3 * N:
                    params[idx] = params[idx] % (math.pi / 3.0)
                
                ns, npen = _eval(params)
                ncost = ns + 100.0 * npen
                
                if ncost < cur_cost:
                    cur_cost, cur_side, cur_pen = ncost, ns, npen
                    no_improve = 0
                    step = min(0.1, step * 1.05)
                else:
                    params[idx] = old
                    no_improve += 1
                    step = max(0.005, step * 0.95)
                
                if npen <= 1e-6 and ns < best_side:
                    best_side = ns
                    best_pen = npen
                    best_params = params.copy()
    
    return best_params, best_side, best_pen


def construct():
    try:
        params, side, pen = _pack()
        if params is None:
            # fallback: the seed-like line (still valid, large side)
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
        # apply global rotation to centers and angles
        theta_g = float(params[3 * N]) % (math.pi / 3.0)
        inner = []
        for i in range(N):
            cx = float(params[3 * i])
            cy = float(params[3 * i + 1])
            ca = float(params[3 * i + 2])
            # local center already includes global rotation in x,y; add global to angle
            ang = (ca + theta_g) % 360.0
            inner.append([cx, cy, ang])
        return inner, (0.0, 0.0), float(side), float(theta_g)
    except Exception:
        # safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0


if __name__ == "__main__":
    res = construct()
    print("outer side:", res[2], "rotation:", res[3], "n inner:", len(res[0]))
    print("elapsed:", time.time() - _START)
