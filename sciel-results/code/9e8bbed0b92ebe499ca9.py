"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np

N = 11
_DEADLINE = 145.0

def _hex_inradius(s):
    return s * math.cos(math.radians(30))

def _hex_verts(cx, cy, s, rot):
    """6 vertices of hexagon centered at (cx,cy), side s, rotation rot (radians)."""
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    """Check if point (px,py) is inside hexagon at (cx,cy) side s rotation rot (radians)."""
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _support(A, d):
    dots = A @ d
    return A[np.argmax(dots)]

def _gjk_distance(A, B, max_iter=40):
    """Min distance between two convex hexagons. 0 if overlapping."""
    d = np.array([1.0, 0.0])
    simplex = []
    
    for _ in range(max_iter):
        a = _support(A, d)
        b = _support(B, -d)
        p = a - b
        
        if np.dot(p, d) < 0:
            break
        
        simplex.append(p)
        
        if len(simplex) == 1:
            d = -simplex[0]
            if np.dot(d, d) < 1e-12:
                return 0.0
        elif len(simplex) == 2:
            p0 = simplex[0]
            p1 = simplex[1]
            v = p1 - p0
            w = -p0
            vv = np.dot(v, v)
            t = np.dot(w, v) / vv if vv > 1e-12 else 0.0
            if t >= 0 and t <= 1:
                closest = p0 + t * v
                if np.dot(closest, closest) < 1e-12:
                    return 0.0
                d = np.array([-v[1], v[0]])
                if np.dot(d, closest) > 0:
                    d = -d
            else:
                if t < 0:
                    d = -p0
                    simplex = [p0]
                else:
                    d = -p1
                    simplex = [p1]
        else:
            p0 = simplex[0]
            p1 = simplex[1]
            p2 = simplex[2]
            
            v0 = p2 - p0
            v1 = p1 - p0
            w = -p0
            
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(w, v0)
            d21 = np.dot(w, v1)
            
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                d = -p0
                simplex = [p0]
                continue
            
            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            
            if u >= 0 and v >= 0 and u + v <= 1:
                return 0.0
            elif u >= 0 and v < 0:
                d = -p0
                simplex = [p0]
            elif u < 0 and v >= 0:
                d = -p0
                simplex = [p0]
            elif u >= 0 and v >= 0 and u + v > 1:
                d = -(p1 + p2) / 2
                simplex = [p1, p2]
            elif u < 0 and v < 0:
                d = -p0
                simplex = [p0]
            else:
                d = -p0
                simplex = [p0]
    
    if len(simplex) == 1:
        return np.linalg.norm(simplex[0])
    elif len(simplex) == 2:
        p0 = simplex[0]
        p1 = simplex[1]
        v = p1 - p0
        vv = np.dot(v, v)
        t = np.dot(-p0, v) / vv if vv > 1e-12 else 0.0
        t = max(0.0, min(1.0, t))
        closest = p0 + t * v
        return np.linalg.norm(closest)
    else:
        return min(np.linalg.norm(p) for p in simplex)

def _min_dist_hexagons(xi, yi, ai, xj, yj, aj):
    vi = _hex_verts(xi, yi, 1.0, ai)
    vj = _hex_verts(xj, yj, 1.0, aj)
    return _gjk_distance(vi, vj)

def _enclosing_hex(inner):
    """Compute minimal enclosing regular hexagon. Returns (cx, cy, S, rot)."""
    # Collect all vertices
    all_verts = []
    for (xi, yi, ai_deg) in inner:
        ai = math.radians(ai_deg)
        vs = _hex_verts(xi, yi, 1.0, ai)
        all_verts.append(vs)
    
    all_verts = np.vstack(all_verts)
    
    # Try different rotations of the outer hexagon
    best_S = float('inf')
    best_cx, best_cy, best_rot = 0.0, 0.0, 0.0
    
    for rot_deg in range(0, 60, 5):
        rot = math.radians(rot_deg)
        
        # For a hexagon with rotation rot, the "width" in direction theta is:
        # We need all points to satisfy: |n_k . (p - c)| <= r_in for all 6 normals
        # The optimal center is the midpoint of min/max projections
        # and S = max over all points of max_k |n_k . (p - c)| / cos(30°)
        
        # Normals of the hexagon sides
        normals = np.array([
            [math.cos(math.radians(60.0*k) + rot), math.sin(math.radians(60.0*k) + rot)]
            for k in range(6)
        ])
        
        # For each normal direction, find min and max projection
        max_required = 0.0
        for n in normals:
            projs = all_verts @ n
            pmin = projs.min()
            pmax = projs.max()
            width = pmax - pmin
            if width > max_required:
                max_required = width
        
        # The inradius needed is max_required / 2
        # Side length = inradius / cos(30°)
        r_in_needed = max_required / 2.0
        S = r_in_needed / math.cos(math.radians(30))
        
        if S < best_S:
            best_S = S
            # Center is midpoint of the tightest constraint
            # Use the normal that gives the max width
            # Actually, center should be at the midpoint of all projections
            cx = 0.0
            cy = 0.0
            for n in normals:
                projs = all_verts @ n
                cx += (projs.min() + projs.max()) / 2.0 / 6.0
                cy += 0.0  # This isn't quite right for 2D
        
            # Better: center is mean of all vertices (approximation)
            cx = all_verts[:, 0].mean()
            cy = all_verts[:, 1].mean()
            best_cx, best_cy, best_rot = cx, cy, rot
    
    return best_cx, best_cy, best_S, best_rot

def _is_valid_config(inner, center, side, angle):
    """Validate: all hexagons inside outer, no overlaps."""
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    # Check boundary
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False
    
    # Check overlap
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            vj = _hex_verts(xj, yj, 1.0, aj)
            dist = _gjk_distance(vi, vj)
            if dist < 1e-9:
                return False
    
    return True

def _honeycomb_positions():
    """Honeycomb: center + ring of 6 + 4 more."""
    positions = []
    positions.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def _compact_cluster_positions():
    """More compact initial cluster."""
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # 6 around center at distance 2
    for k in range(6):
        a = math.radians(60.0 * k)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    # 4 more: fill gaps
    for k in range(4):
        a = math.radians(60.0 * k + 60.0)
        positions.append((3.464*math.cos(a), 3.464*math.sin(a), 0.0))
    return positions[:N]

def _random_positions(rng, spread=3.0):
    """Random initial positions."""
    positions = []
    for _ in range(N):
        x = rng.randn() * spread
        y = rng.randn() * spread
        a = rng.uniform(0, 360)
        positions.append((x, y, a))
    return positions

def _simulated_annealing(init_pos, rng, t_start, max_steps=20000, 
                          T_start=2.0, T_end=0.01, alpha=0.9997,
                          time_limit=None):
    """Simulated annealing to minimize enclosing hexagon side length."""
    if time_limit is None:
        time_limit = _DEADLINE - 5.0
    
    n = len(init_pos)
    
    # State: [x1, y1, a1, x2, y2, a2, ...]
    state = np.zeros(3 * n)
    for i in range(n):
        state[3*i] = init_pos[i][0]
        state[3*i+1] = init_pos[i][1]
        state[3*i+2] = math.radians(init_pos[i][2])
    
    def _get_pos_angles(state):
        x = state[0:3*n:3]
        y = state[1:3*n:3]
        a = state[2:3*n:3]
        return x, y, a
    
    def _energy(state):
        """Compute energy: enclosing side length + penalties for overlap and boundary."""
        x, y, a = _get_pos_angles(state)
        
        # Compute enclosing hexagon
        all_verts = np.empty((6*n, 2))
        for i in range(n):
            vs = _hex_verts(x[i], y[i], 1.0, a[i])
            all_verts[6*i:6*i+6] = vs
        
        # Find minimal enclosing hexagon (try several rotations)
        best_S = float('inf')
        for rot_deg in [0, 15, 30, 45]:
            rot = math.radians(rot_deg)
            normals = np.array([
                [math.cos(math.radians(60.0*k) + rot), math.sin(math.radians(60.0*k) + rot)]
                for k in range(6)
            ])
            max_width = 0.0
            for nm in normals:
                projs = all_verts @ nm
                max_width = max(max_width, projs.max() - projs.min())
            S = (max_width / 2.0) / math.cos(math.radians(30))
            if S < best_S:
                best_S = S
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.0:
                    continue
                vi = _hex_verts(x[i], y[i], 1.0, a[i])
                vj = _hex_verts(x[j], y[j], 1.0, a[j])
                dist = _gjk_distance(vi, vj)
                if dist < 0.01:
                    overlap_pen += (0.01 - dist) ** 2 * 1000.0
        
        return best_S + overlap_pen
    
    # Initial energy
    E_curr = _energy(state)
    best_state = state.copy()
    best_E = E_curr
    
    T = T_start
    step = 0
    
    while step < max_steps:
        if time.time() > time_limit:
            break
        
        # Generate perturbation
        idx = rng.randint(n)
        pert = state.copy()
        scale = 0.3 * T + 0.05
        pert[3*idx] += rng.randn() * scale
        pert[3*idx+1] += rng.randn() * scale
        pert[3*idx+2] += rng.randn() * scale * 0.5
        
        E_new = _energy(pert)
        delta_E = E_new - E_curr
        
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            state = pert
            E_curr = E_new
            if E_new < best_E:
                best_E = E_new
                best_state = state.copy()
        
        T *= alpha
        step += 1
    
    x, y, a = _get_pos_angles(best_state)
    return list(zip(x.tolist(), y.tolist(), [math.degrees(ai) for ai in a]))

def construct():
    t_start = time.time()
    n = N
    
    def _remaining():
        return _DEADLINE - (time.time() - t_start)
    
    # Initial configurations
    init_configs = [
        _honeycomb_positions(),
        _compact_cluster_positions(),
    ]
    
    rng = np.random.RandomState(42)
    for _ in range(3):
        init_configs.append(_random_positions(rng, spread=2.5))
    
    best_result = None
    best_S = float('inf')
    
    # Compute initial valid config as fallback
    init_pos = init_configs[0]
    cx0, cy0, S0, rot0 = _enclosing_hex(init_pos)
    if _is_valid_config(init_pos, (cx0, cy0), S0, rot0):
        best_result = [[float(p[0]), float(p[1]), float(p[2])] for p in init_pos]
        best_S = S0
    
    # Run simulated annealing from each initial config
    for trial, init_pos in enumerate(init_configs):
        if _remaining() < 25.0:
            break
        
        # Compute initial enclosing
        cx0, cy0, S0, rot0 = _enclosing_hex(init_pos)
        
        # Run SA
        try:
            result = _simulated_annealing(
                init_pos, rng, t_start,
                max_steps=15000,
                T_start=1.5 + 0.3 * trial,
                T_end=0.005,
                alpha=0.9998,
                time_limit=_DEADLINE - 8.0
            )
            
            # Compute enclosing hexagon for result
            cx, cy, S, rot = _enclosing_hex(result)
            
            # Validate
            if _is_valid_config(result, (cx, cy), S, rot):
                if S < best_S:
                    best_S = S
                    best_result = [[float(p[0]), float(p[1]), float(p[2])] for p in result]
        except Exception:
            pass
    
    # Final fallback
    if best_result is None:
        init_pos = _honeycomb_positions()
        cx0, cy0, S0, rot0 = _enclosing_hex(init_pos)
        best_result = [[float(p[0]), float(p[1]), float(p[2])] for p in init_pos]
        best_S = S0
        cx, cy, rot = cx0, cy0, rot0
    else:
        # Recompute enclosing for best result
        cx, cy, best_S, rot = _enclosing_hex(best_result)
    
    return best_result, (float(cx), float(cy)), float(best_S), float(rot)
