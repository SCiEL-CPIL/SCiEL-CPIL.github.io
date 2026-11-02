import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
S3 = np.sqrt(3.0)

def _hex_verts(cx, cy, side, ang):
    a = ang + np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([cx + side * np.cos(a), cy + side * np.sin(a)])

def _sm(x, k=80.0):
    x = np.asarray(x)
    m = np.max(x)
    return m + np.log(np.sum(np.exp(k * (x - m)))) / k

def construct():
    t0 = time.time()
    DEADLINE = t0 + 135.0

    base = np.array([[1.0, 0.0], [0.5, S3/2], [-0.5, S3/2],
                     [-1.0, 0.0], [-0.5, -S3/2], [0.5, -S3/2]])

    def get_verts(pos, alpha, s):
        c = np.cos(alpha); sn = np.sin(alpha)
        R = np.array([[c, -sn], [sn, c]])
        return pos[:, None, :] + (base @ R.T)[None, :, :]

    def _axes(V):
        e = np.roll(V, -1, 0) - V
        n = np.column_stack([e[:, 1], -e[:, 0]])
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-15)
        return n

    def obj(p, K=60.0):
        if time.time() > DEADLINE:
            return 1e8
        pos = p[:2 * N].reshape(N, 2)
        s = p[2 * N]
        alpha = p[2 * N + 1]
        theta = p[2 * N + 2]
        if s < 2.0:
            return 1e6
        verts = get_verts(pos, alpha, s)
        outer = _hex_verts(0.0, 0.0, s, theta)
        on = _axes(outer)
        p_in = np.einsum('nij,jk->nik', verts, on.T)
        p_out = (outer @ on.T).max(0)
        contain = _sm(np.maximum(0.0, p_in.max(1) - p_out[None, :]), K).sum()
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                A = verts[i]; B = verts[j]
                nn = np.vstack([_axes(A), _axes(B)])
                pA = A @ nn.T; pB = B @ nn.T
                ov = np.minimum(pA.max(0), pB.max(0)) - np.maximum(pA.min(0), pB.min(0))
                pen += _sm(np.maximum(0.0, ov), K)
        return s + 50.0 * contain + 50.0 * pen

    def is_valid(pos, s, alpha, theta, tol=1e-5):
        verts = get_verts(pos, alpha, s)
        outer = _hex_verts(0.0, 0.0, s, theta)
        on = _axes(outer)
        p_in = np.einsum('nij,jk->nik', verts, on.T)
        p_out = (outer @ on.T).max(0)
        if np.max(p_in.max(1) - p_out[None, :]) > tol:
            return False
        for i in range(N):
            for j in range(i + 1, N):
                A = verts[i]; B = verts[j]
                nn = np.vstack([_axes(A), _axes(B)])
                pA = A @ nn.T; pB = B @ nn.T
                ov = np.minimum(pA.max(0), pB.max(0)) - np.maximum(pA.min(0), pB.min(0))
                if ov.min() > tol:
                    return False
        return True

    def s_of(p):
        return p[2 * N]

    def polish(x0, budget):
        best_x = np.array(x0, float)
        best_s = s_of(best_x)
        for _ in range(6):
            if time.time() > DEADLINE - 12 or time.time() > budget:
                break
            try:
                res = minimize(obj, best_x, method='Nelder-Mead',
                               options={'maxiter': 1200, 'xatol': 1e-8,
                                        'fatol': 1e-9, 'adaptive': True})
                if res.x is not None:
                    if s_of(res.x) < best_s:
                        best_x = res.x.copy(); best_s = s_of(best_x)
            except Exception:
                break
        return best_x, best_s

    def build_init():
        d1 = S3
        d2 = 2 * S3
        pos = [[0.0, 0.0]]
        for k in range(6):
            a = np.deg2rad(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(5):
            a = np.deg2rad(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        return np.array(pos)

    init_pos = build_init()
    init = np.concatenate([init_pos.ravel(), [3.7, 0.0, 0.0]])
    best = init.copy()
    best_s = s_of(init)

    if is_valid(init_pos, 3.7, 0.0, 0.0):
        pass
    else:
        # ensure a valid fallback
        for s_try in [3.7, 3.9, 4.0, 4.2]:
            if is_valid(init_pos, s_try, 0.0, 0.0):
                best = np.concatenate([init_pos.ravel(), [s_try, 0.0, 0.0]])
                best_s = s_try
                break

    # ---- Global search: differential evolution over full 36-d space ----
    bounds = [(-5.0, 5.0)] * (2 * N) + [(2.0, 5.0), (-0.785, 0.785), (-0.785, 0.785)]
    try:
        def de_obj(p):
            if time.time() > DEADLINE - 20:
                return 1e8
            return obj(p)
        de_res = differential_evolution(de_obj, bounds, seed=7, popsize=12,
                                        maxiter=25, tol=1e-5, polish=False,
                                        init='sobol', mutation=(0.5, 1.2),
                                        recombination=0.9,
                                        updating='deferred')
        if de_res.x is not None:
            if s_of(de_res.x) < best_s:
                best = de_res.x.copy(); best_s = s_of(best)
    except Exception:
        pass

    # ---- Local polish from best (and from initial) ----
    for seed_x in [best, init]:
        if time.time() > DEADLINE - 25:
            break
        try:
            px, ps = polish(seed_x, DEADLINE - 25)
            if ps < best_s:
                best = px.copy(); best_s = ps
        except Exception:
            pass

    # ---- Validate & final fallback ----
    bp = best[:2 * N].reshape(N, 2)
    bs = s_of(best); ba = best[2 * N + 1]; bt = best[2 * N + 2]
    if not is_valid(bp, bs, ba, bt):
        # try to repair by enlarging slightly
        for ds in [0.0, 0.05, 0.1, 0.2, 0.3]:
            if is_valid(bp, bs + ds, ba, bt):
                best = best.copy(); best[2 * N] = bs + ds; best_s = bs + ds
                break
        else:
            # full fallback to a known-valid layout
            for s_try in [4.0, 4.2, 4.5, 5.0]:
                if is_valid(init_pos, s_try, 0.0, 0.0):
                    best = np.concatenate([init_pos.ravel(), [s_try, 0.0, 0.0]])
                    best_s = s_try
                    break
            else:
                best = np.concatenate([init_pos.ravel(), [5.0, 0.0, 0.0]])
                best_s = 5.0

    bp = best[:2 * N].reshape(N, 2)
    bs = s_of(best); ba = best[2 * N + 1]; bt = best[2 * N + 2]
    inner = [[float(bp[i, 0]), float(bp[i, 1]), float(np.rad2deg(ba))] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = float(bs)
    outer_angle_deg = float(np.rad2deg(bt))
    return inner, outer_center, outer_side, outer_angle_deg
