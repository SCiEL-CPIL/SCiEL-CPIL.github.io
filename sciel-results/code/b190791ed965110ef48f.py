import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

T0 = time.time()
DEADLINE = 140.0  # Safety margin for 150s limit
SAFETY = 5.0
N = 11

def _hex_verts(cx, cy, a, theta):
    # regular hexagon side a (= circumradius), rotation theta (radians)
    ang = theta + np.arange(6) * (np.pi / 3.0)
    r = a
    return np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])

def _edge_axes(verts):
    # outward-ish normals (perpendicular to edges) for SAT
    axes = []
    m = len(verts)
    for i in range(m):
        p1 = verts[i]
        p2 = verts[(i + 1) % m]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        n = n / (np.linalg.norm(n) + 1e-18)
        axes.append(n)
    return np.array(axes)

def _sat_gap(A, B):
    # returns separating gap (positive if separated, negative if overlap depth approx)
    axes = np.vstack([_edge_axes(A), _edge_axes(B)])
    best = -np.inf
    for n in axes:
        a_min, a_max = A @ n, A @ n
        a_min, a_max = a_min.min(), a_max.max()
        b_min, b_max = B @ n, B @ n
        b_min, b_max = b_min.min(), b_max.max()
        gap = min(a_max, b_max) - max(a_min, b_min)
        best = max(best, gap)
    return best  # >0 separated, <0 overlapping

def _containment_violation(inner_verts, outer_verts):
    # all inner verts must be inside outer (convex) hexagon
    ov = outer_verts
    m = len(ov)
    viol = 0.0
    for i in range(m):
        p1 = ov[i]
        p2 = ov[(i + 1) % m]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        # ensure n points inward: test against center
        c = ov.mean(axis=0)
        if np.dot(n, c - p1) < 0:
            n = -n
        n = n / (np.linalg.norm(n) + 1e-18)
        d = (inner_verts - p1) @ n
        v = -d.min()  # negative d means outside
        if v > 0:
            viol += v * v
    return viol

def _decode(params):
    # params: for each inner: x,y,theta (3 each) = 3N; outer: cx,cy,s,phi (4)
    inner = []
    for k in range(N):
        x, y, th = params[3*k:3*k+3]
        inner.append((x, y, th))
    ocx, ocy, s, phi = params[3*N:3*N+4]
    return inner, (ocx, ocy), s, phi

def _objective(params):
    inner, (ocx, ocy), s, phi = _decode(params)
    if s < 0.5:
        s = 0.5
    outer = _hex_verts(ocx, ocy, s, phi)
    inners = [_hex_verts(x, y, 1.0, th) for (x, y, th) in inner]
    # containment
    pen = 0.0
    for iv in inners:
        pen += 1000.0 * _containment_violation(iv, outer)
    # pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            g = _sat_gap(inners[i], inners[j])
            if g < 0:
                pen += 1000.0 * g * g
    return s + pen

def _is_valid(inner, center, s, phi):
    ocx, ocy = center
    outer = _hex_verts(ocx, ocy, s, phi)
    inners = [_hex_verts(x, y, 1.0, th) for (x, y, th) in inner]
    for iv in inners:
        if _containment_violation(iv, outer) > 1e-6:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _sat_gap(inners[i], inners[j]) < -1e-6:
                return False
    return True

def construct():
    global T0
    rng = np.random.default_rng(0)

    best = None
    best_val = np.inf

    def consider(params):
        nonlocal best, best_val
        inner, center, s, phi = _decode(params)
        if s < 0.5:
            return
        if _is_valid(inner, center, s, phi):
            val = s
            if val < best_val:
                best_val = val
                best = (inner, center, s, phi)

    def time_ok():
        return (time.time() - T0) < (DEADLINE - SAFETY)

    # Generate multiple diverse initial guesses
    def initial_guess(seed_offset=0):
        inner = []
        # Center
        inner.append((0.0, 0.0, 0.0))
        # Ring of 6
        d = np.sqrt(3.0) * 1.0
        for k in range(6):
            ang = k * np.pi / 3.0
            inner.append((d*np.cos(ang), d*np.sin(ang), 0.0))
        # 4 more placed around
        d2 = d * 1.9
        for k in range(4):
            ang = np.pi/6 + k * np.pi/2
            inner.append((d2*np.cos(ang), d2*np.sin(ang), 0.0))
        
        # Add some randomness to break symmetry
        for k in range(len(inner)):
            inner[k] = (inner[k][0] + rng.normal(0, 0.1), inner[k][1] + rng.normal(0, 0.1), rng.uniform(-0.1, 0.1))
            
        ocx, ocy = 0.0, 0.0
        s = 4.5 # Start slightly larger to ensure validity
        phi = 0.0
        params = []
        for (x,y,th) in inner:
            params += [x,y,th]
        params += [ocx, ocy, s, phi]
        return np.array(params)

    # Try a few random restarts with Nelder-Mead
    # Nelder-Mead is faster than DE for this dimensionality if started near a solution
    for i in range(5):
        if not time_ok():
            break
            
        p0 = initial_guess(i)
        consider(p0)
        
        try:
            # Use Nelder-Mead with strict time/iteration limits
            r = minimize(_objective, p0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            consider(r.x)
        except Exception:
            pass

    # If we haven't found a valid solution, try a simpler heuristic fallback
    if best is None:
        # Fallback: Linear arrangement is easy to validate
        inner = []
        for k in range(N):
            # Place them in a row with spacing 2*sqrt(3) for flat-to-flat touching
            x = k * 2 * np.sqrt(3)
            inner.append([x, 0.0, 0.0])
        
        # Calculate required side length
        # Total width is (N-1)*2*sqrt(3) + 2*sqrt(3) (for end hexagons)
        # Width of outer hexagon with side S is 2*S*sqrt(3)
        # So 2*S*sqrt(3) >= N * 2 * sqrt(3) => S >= N
        # Add a small margin
        s = N + 0.5
        center = (0.0, 0.0)
        phi = 0.0
        
        # Center the inner row
        offset = -(N-1)*np.sqrt(3)
        inner = [[x + offset, 0.0, 0.0] for x in [k * 2 * np.sqrt(3) for k in range(N)]]
        
        if _is_valid(inner, center, s, phi):
             best = (inner, center, s, phi)
             best_val = s

    if best is None:
        # Absolute fallback: Huge box
        inner = [[0.0, 0.0, 0.0] for _ in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0

    inner, center, s, phi = best
    inner_out = [[float(x), float(y), float(np.degrees(th)) % 360.0] for (x, y, th) in inner]
    return inner_out, (float(center[0]), float(center[1])), float(s), float(np.degrees(phi) % 360.0)
