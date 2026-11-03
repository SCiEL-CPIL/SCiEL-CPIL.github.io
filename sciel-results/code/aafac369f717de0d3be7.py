"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def _hex_verts(x, y, s, ang_deg):
    a = np.deg2rad(ang_deg)
    angs = a + np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([x + s * np.cos(angs), y + s * np.sin(angs)])

def _enclosing_hex(pts, theta_deg):
    """Fast: for given orientation, find min side and center."""
    theta = np.deg2rad(theta_deg)
    n_angs = theta + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack([np.cos(n_angs), np.sin(n_angs)])
    A = pts @ normals.T
    A_max = A.max(axis=0)
    
    # Minimize f(c) = max_i (A_max[i] - n_i . c)
    # This is convex. Use a few iterations of subgradient or just try centroid + a few candidates.
    # For speed, use a simple approach: the optimum c is where the max is balanced.
    # Try: c = centroid, and also c from pairs of active constraints.
    candidates = [pts.mean(axis=0)]
    # Try intersection of each pair of "lines" A_max[i] - n_i.c = A_max[j] - n_j.c
    for i in range(6):
        for j in range(i+1, 6):
            d = A_max[i] - A_max[j]
            # (n_i - n_j) . c = d
            diff = normals[i] - normals[j]
            if np.linalg.norm(diff) < 1e-10:
                continue
            # One equation, 2 unknowns. Add constraint that c is in direction of diff from centroid.
            c0 = pts.mean(axis=0)
            # c = c0 + t * diff
            # (n_i - n_j) . (c0 + t*diff) = d
            # t = (d - diff . c0) / (diff . diff)
            t = (d - diff @ c0) / (diff @ diff)
            c = c0 + t * diff
            candidates.append(c)
    
    best_t = 1e9
    best_c = candidates[0]
    for c in candidates:
        t = (A_max - normals @ c).max()
        if t < best_t:
            best_t = t
            best_c = c
    
    side = (2.0 / np.sqrt(3.0)) * best_t
    return side, best_c[0], best_c[1]

def _min_enclosing(pts):
    best = (1e9, 0, 0, 0)
    for td in np.arange(0, 60, 5.0):
        s, cx, cy = _enclosing_hex(pts, td)
        if s < best[0]:
            best = (s, cx, cy, td)
    # Refine
    for td in np.arange(best[3]-5, best[3]+5, 1.0):
        s, cx, cy = _enclosing_hex(pts, td)
        if s < best[0]:
            best = (s, cx, cy, td)
    for td in np.arange(best[3]-1, best[3]+1, 0.2):
        s, cx, cy = _enclosing_hex(pts, td)
        if s < best[0]:
            best = (s, cx, cy, td)
    return best

def _any_overlap(inner):
    verts = [_hex_verts(x, y, 1.0, a) for (x, y, a) in inner]
    def pt_in_hex(px, py, v):
        inside = False
        for k in range(6):
            x1, y1 = v[k]
            x2, y2 = v[(k+1) % 6]
            if ((y1 > py) != (y2 > py)) and (px < (x2-x1)*(py-y1)/(y2-y1)+x1):
                inside = not inside
        return inside
    for i in range(N):
        for j in range(i+1, N):
            ci = np.array([inner[i][0], inner[i][1]])
            cj = np.array([inner[j][0], inner[j][1]])
            if np.linalg.norm(ci - cj) > 3.5:
                continue
            for (px, py) in verts[i]:
                if pt_in_hex(px, py, verts[j]):
                    return True
            for (px, py) in verts[j]:
                if pt_in_hex(px, py, verts[i]):
                    return True
    return False

def construct():
    np.random.seed(42)
    
    # Honeycomb cluster
    e1 = np.sqrt(3)
    axial = [(-1,-1),(0,-1),(1,-1), (-1,0),(0,0),(1,0),(2,0), (-1,1),(0,1),(1,1), (0,2),(1,2)]
    inner = []
    for (q, r) in axial:
        x = e1 * (q + r/2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    all_v = np.vstack([_hex_verts(x, y, 1.0, a) for (x, y, a) in inner])
    side, ox, oy, oang = _min_enclosing(all_v)
    
    best_side = side
    best_inner = [list(x) for x in inner]
    best_oc = (ox, oy)
    best_oa = oang
    
    T = 0.3
    T_min = 0.003
    cool = 0.998
    max_iter = 3000
    
    for it in range(max_iter):
        idx = np.random.randint(N)
        new_inner = [list(x) for x in inner]
        new_inner[idx][0] += np.random.normal(0, 0.2)
        new_inner[idx][1] += np.random.normal(0, 0.2)
        new_inner[idx][2] += np.random.normal(0, 10.0)
        
        all_v = np.vstack([_hex_verts(x, y, 1.0, a) for (x, y, a) in new_inner])
        ns, nox, noy, noang = _min_enclosing(all_v)
        
        pen = 50.0 if _any_overlap(new_inner) else 0.0
        cost_new = ns + pen
        cost_old = best_side + (50.0 if _any_overlap(inner) else 0.0)
        
        delta = cost_new - cost_old
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            inner = new_inner
            side = ns
            ox, oy, oang = nox, noy, noang
            if pen == 0 and ns < best_side:
                best_side = ns
                best_inner = [list(x) for x in new_inner]
                best_oc = (nox, noy)
                best_oa = noang
        
        T *= cool
        if T < T_min:
            T = T_min
    
    fin = best_inner
    all_v = np.vstack([_hex_verts(x, y, 1.0, a) for (x, y, a) in fin])
    fs, fox, foy, foang = _min_enclosing(all_v)
    
    # Expand if overlap or containment violation
    for _ in range(30):
        if _any_overlap(fin):
            fs *= 1.02
            continue
        # Check containment
        theta = np.deg2rad(foang)
        n_angs = theta + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
        normals = np.column_stack([np.cos(n_angs), np.sin(n_angs)])
        apothem = fs * np.sqrt(3) / 2
        c = np.array([fox, foy])
        viol = 0.0
        for (x, y, a) in fin:
            v = _hex_verts(x, y, 1.0, a)
            for k in range(6):
                n = normals[k]
                dists = (v - c) @ n
                viol += np.maximum(0.0, dists - apothem).sum()
        if viol < 1e-6:
            break
        fs *= 1.01
    
    return fin, (fox, foy), fs, foang
# EVOLVE-BLOCK-END
