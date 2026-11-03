"""Fresh design: spectral graph embedding for 12 unit hexagons in a hexagon."""
import time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

N = 12

def hexagon_support(angle_deg, direction):
    """Support of unit flat-top hexagon rotated by angle_deg in unit direction."""
    a = np.radians(angle_deg)
    c = np.cos(a); s = np.sin(a)
    base = np.array([[np.cos(np.radians(d)), np.sin(np.radians(d))] for d in range(0, 360, 60)])
    R = np.array([[c, -s], [s, c]])
    v = base @ R.T
    return float(np.max(v @ direction))

def hexagon_min_support(angle_deg, direction):
    a = np.radians(angle_deg)
    c = np.cos(a); s = np.sin(a)
    base = np.array([[np.cos(np.radians(d)), np.sin(np.radians(d))] for d in range(0, 360, 60)])
    R = np.array([[c, -s], [s, c]])
    v = base @ R.T
    return float(np.min(v @ direction))

def get_check_directions(inner_angles_deg, theta_deg):
    dirs = []
    for d in range(0, 360, 30):
        ang = np.radians(d + theta_deg)
        dirs.append([np.cos(ang), np.sin(ang)])
    for ang_deg in inner_angles_deg:
        for d in range(0, 360, 30):
            ang = np.radians(d + ang_deg)
            dirs.append([np.cos(ang), np.sin(ang)])
    arr = np.array(dirs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    uniq = []
    for row in arr:
        found = False
        for u in uniq:
            if np.dot(row, u) > 0.9999:
                found = True
                break
        if not found:
            uniq.append(row)
    return np.array(uniq)

def min_outer_side(positions, inner_angles_deg, theta_deg, cx, cy):
    local = positions - np.array([cx, cy])
    dirs = get_check_directions(inner_angles_deg, theta_deg)
    L_needed = 0.0
    a = np.radians(theta_deg)
    c = np.cos(a); s = np.sin(a)
    base = np.array([[np.cos(np.radians(d)), np.sin(np.radians(d))] for d in range(0, 360, 60)])
    R = np.array([[c, -s], [s, c]])
    v_unit = base @ R.T
    sup_unit = float(np.max(v_unit @ dirs.T, axis=1))
    minsup_unit = float(np.min(v_unit @ dirs.T, axis=1))
    # For each direction, compute max_proj and min_proj over inner hexagons
    for i in range(len(dirs)):
        d = dirs[i]
        max_proj = -1e18
        min_proj = 1e18
        for j in range(N):
            sup_j = hexagon_support(inner_angles_deg[j], d) + float(local[j] @ d)
            minsup_j = hexagon_min_support(inner_angles_deg[j], d) + float(local[j] @ d)
            if sup_j > max_proj: max_proj = sup_j
            if minsup_j < min_proj: min_proj = minsup_j
        su = sup_unit[i]
        mu = minsup_unit[i]
        if su > 1e-9:
            L1 = max_proj / su
        else:
            L1 = 0.0
        if mu < -1e-9:
            L2 = min_proj / mu
        else:
            L2 = 0.0
        L_needed = max(L_needed, L1, L2)
    return L_needed

def build_base_graph():
    edges = []
    for i in range(1, 6):
        edges.append((0, i))
    for i in range(1, 6):
        edges.append((i, i % 5 + 1))
    edges.append((6, 1)); edges.append((7, 2)); edges.append((8, 3))
    edges.append((9, 4)); edges.append((10, 5)); edges.append((11, 0))
    edges.append((6, 7)); edges.append((7, 8)); edges.append((8, 9))
    edges.append((9, 10)); edges.append((10, 6)); edges.append((11, 6)); edges.append((11, 10))
    return edges

def spectral_embedding(weights, edges):
    n = N
    W = np.zeros((n, n))
    for (u, v), w in zip(edges, weights):
        W[u, v] = w; W[v, u] = w
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    vals, vecs = eigh(L)
    order = np.argsort(vals)
    vecs = vecs[:, order]
    pos = vecs[:, 1:3]
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    edges = build_base_graph()
    n_edges = len(edges)
    n_params = n_edges + 4 + N + 1

    def objective(params):
        if time.time() > deadline:
            return 1e10
        log_w = params[:n_edges]
        weights = np.exp(np.clip(log_w, -5, 5))
        theta_deg = params[n_edges]
        cx = params[n_edges + 1]
        cy = params[n_edges + 2]
        inner_angles = params[n_edges + 3 : n_edges + 3 + N]
        scale = params[n_edges + 3 + N]
        pos = spectral_embedding(weights, edges)
        pos = pos - pos.mean(axis=0)
        maxnorm = np.max(np.linalg.norm(pos, axis=1))
        if maxnorm > 1e-9:
            pos = pos / maxnorm
        real_pos = scale * pos
        return min_outer_side(real_pos, inner_angles, theta_deg, cx, cy)

    x0 = np.zeros(n_params)
    x0[:n_edges] = 0.0
    x0[n_edges] = 0.0
    x0[n_edges + 1] = 0.0
    x0[n_edges + 2] = 0.0
    x0[n_edges + 3 : n_edges + 3 + N] = 0.0
    x0[n_edges + 3 + N] = 4.0

    best_result = None
    best_L = 1e10
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
        if res.fun < best_L:
            best_L = res.fun
            best_result = res.x
    except Exception:
        pass

    if time.time() < deadline - 25.0 and best_result is not None:
        try:
            res2 = minimize(objective, best_result + np.random.randn(n_params) * 0.1,
                            method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
            if res2.fun < best_L:
                best_L = res2.fun
                best_result = res2.x
        except Exception:
            pass

    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    log_w = best_result[:n_edges]
    weights = np.exp(np.clip(log_w, -5, 5))
    theta_deg = best_result[n_edges]
    cx = best_result[n_edges + 1]
    cy = best_result[n_edges + 2]
    inner_angles = best_result[n_edges + 3 : n_edges + 3 + N]
    scale = best_result[n_edges + 3 + N]
    pos = spectral_embedding(weights, edges)
    pos = pos - pos.mean(axis=0)
    maxnorm = np.max(np.linalg.norm(pos, axis=1))
    if maxnorm > 1e-9:
        pos = pos / maxnorm
    real_pos = scale * pos
    L = min_outer_side(real_pos, inner_angles, theta_deg, cx, cy)
    inner = [[float(real_pos[i, 0]), float(real_pos[i, 1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(cx), float(cy)), float(L), float(theta_deg)
