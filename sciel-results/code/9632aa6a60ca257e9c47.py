"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Strategy:
- Use a smooth differentiable penalty (softplus for overlap, quadratic for boundary)
- Multi-stage L-BFGS-B with increasing softness precision
- Multiple aspect ratios
- Simulated annealing for escape from local minima
- Final high-precision polish
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _softplus_vec(x, beta=10.0):
    """Vectorized numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x > 1.0 / beta
    out[mask] = x[mask]
    xm = x[~mask]
    out[~mask] = np.log1p(np.exp(beta * xm)) / beta
    return out


def _sigmoid_vec(x, beta=10.0):
    """Vectorized sigmoid."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-beta * x))


def _overlap_penalty(xs, ys, rs, beta=10.0):
    """Smooth overlap penalty."""
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = rs[:, None] + rs[None, :]
    val = min_dist - dist
    iu = np.triu_indices(N, k=1)
    return np.sum(_softplus_vec(val[iu], beta))


def _overlap_grad(xs, ys, rs, beta=10.0):
    """Gradient of overlap penalty."""
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = rs[:, None] + rs[None, :]
    val = min_dist - dist
    sig = _sigmoid_vec(val, beta)
    
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    dx_ij = dx[i_idx, j_idx]
    dy_ij = dy[i_idx, j_idx]
    dist_ij = dist[i_idx, j_idx]
    sig_ij = sig[i_idx, j_idx]
    inv_dist = np.where(dist_ij > 1e-10, 1.0 / dist_ij, 0.0)
    
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    np.add.at(gx, i_idx, -sig_ij * dx_ij * inv_dist)
    np.add.at(gx, j_idx, sig_ij * dx_ij * inv_dist)
    np.add.at(gy, i_idx, -sig_ij * dy_ij * inv_dist)
    np.add.at(gy, j_idx, sig_ij * dy_ij * inv_dist)
    np.add.at(gr, i_idx, sig_ij)
    np.add.at(gr, j_idx, sig_ij)
    
    return gx, gy, gr


def _boundary_penalty(xs, ys, rs, w, h):
    """Boundary violation penalty."""
    pen = 0.0
    pen += np.sum(np.maximum(rs - xs, 0.0) ** 2)
    pen += np.sum(np.maximum(xs + rs - w, 0.0) ** 2)
    pen += np.sum(np.maximum(rs - ys, 0.0) ** 2)
    pen += np.sum(np.maximum(ys + rs - h, 0.0) ** 2)
    return pen


def _boundary_grad(xs, ys, rs, w, h):
    """Gradient of boundary penalty."""
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    viol_l = np.maximum(rs - xs, 0.0)
    gx[viol_l > 0] -= 2.0 * viol_l[viol_l > 0]
    gr[viol_l > 0] += 2.0 * viol_l[viol_l > 0]
    
    viol_r = np.maximum(xs + rs - w, 0.0)
    gx[viol_r > 0] += 2.0 * viol_r[viol_r > 0]
    gr[viol_r > 0] += 2.0 * viol_r[viol_r > 0]
    
    viol_b = np.maximum(rs - ys, 0.0)
    gy[viol_b > 0] -= 2.0 * viol_b[viol_b > 0]
    gr[viol_b > 0] += 2.0 * viol_b[viol_b > 0]
    
    viol_t = np.maximum(ys + rs - h, 0.0)
    gy[viol_t > 0] += 2.0 * viol_t[viol_t > 0]
    gr[viol_t > 0] += 2.0 * viol_t[viol_t > 0]
    
    return gx, gy, gr


def _obj(params, w, h, beta):
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    if np.any(rs <= 0):
        return 1e10
    return _overlap_penalty(xs, ys, rs, beta) + _boundary_penalty(xs, ys, rs, w, h) - np.sum(rs)


def _obj_grad(params, w, h, beta):
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    if np.any(rs <= 0):
        g = np.zeros_like(params)
        g[2::3] = 2.0 * np.minimum(rs, 0)
        return g
    
    gx_o, gy_o, gr_o = _overlap_grad(xs, ys, rs, beta)
    gx_b, gy_b, gr_b = _boundary_grad(xs, ys, rs, w, h)
    
    g = np.zeros_like(params)
    g[0::3] = gx_o + gx_b
    g[1::3] = gy_o + gy_b
    g[2::3] = gr_o + gr_b - 1.0
    return g


def _valid(params, w, h, tol=1e-6):
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    if np.any(rs < 1e-8):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_d = rs[:, None] + rs[None, :]
    if np.any(dist < min_d - tol):
        return False
    return True


def _update_best(params, w, h, state):
    if _valid(params, w, h):
        val = np.sum(params[2::3])
        if val > state['val']:
            state['val'] = val
            state['res'] = np.column_stack([params[0::3], params[1::3], params[2::3]])


def _scale(result):
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    mnx = np.min(xs - rs)
    mxx = np.max(xs + rs)
    mny = np.min(ys - rs)
    mxy = np.max(ys + rs)
    perim = 2.0 * ((mxx - mnx) + (mxy - mny))
    if perim > 4.0 + 1e-9:
        s = 4.0 / perim
        cx = 0.5 * (mnx + mxx)
        cy = 0.5 * (mny + mxy)
        result[:, 0] = cx + (result[:, 0] - cx) * s
        result[:, 1] = cy + (result[:, 1] - cy) * s
        result[:, 2] *= s
    return result


def _grid_init(w, h, seed=0):
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    x0 = np.zeros(3 * N)
    for i in range(N):
        col = i % a
        row = i // a
        x0[3*i] = (col + 0.5) * w / a
        x0[3*i+1] = (row + 0.5) * h / b
        x0[3*i+2] = 0.45 * min(w/a, h/b)
    if seed > 0:
        rng = np.random.default_rng(seed)
        for i in range(N):
            x0[3*i] += rng.normal(0, 0.02)
            x0[3*i+1] += rng.normal(0, 0.02)
            x0[3*i+2] *= (1.0 + rng.normal(0, 0.1))
            x0[3*i] = np.clip(x0[3*i], x0[3*i+2], w - x0[3*i+2])
            x0[3*i+1] = np.clip(x0[3*i+1], x0[3*i+2], h - x0[3*i+2])
            x0[3*i+2] = np.clip(x0[3*i+2], 1e-4, min(w,h)/2)
    return x0


def _sa(w, h, x0, iters=4000, seed=42, deadline=None):
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    rmax = min(w, h) / 2.0
    T = 0.06
    cool = 0.9997
    beta = 8.0
    rng = np.random.default_rng(seed)
    
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        idx = rng.integers(0, N)
        ox, oy, or_ = xs[idx], ys[idx], rs[idx]
        s = T * 0.35
        nx = np.clip(ox + rng.normal(0, s), or_, w - or_)
        ny = np.clip(oy + rng.normal(0, s), or_, h - or_)
        nr = np.clip(or_ + rng.normal(0, s*0.3), 1e-4, rmax)
        
        others = np.array([i for i in range(N) if i != idx])
        do = np.sqrt((ox - xs[others])**2 + (oy - ys[others])**2 + 1e-12)
        co = np.sum(_softplus_vec(or_ + rs[others] - do, beta))
        
        xs[idx], ys[idx], rs[idx] = nx, ny, nr
        dn = np.sqrt((nx - xs[others])**2 + (ny - ys[others])**2 + 1e-12)
        cn = np.sum(_softplus_vec(nr + rs[others] - dn, beta))
        
        delta = (cn - co) - (nr - or_)
        if delta < 0 or (T > 1e-7 and rng.random() < np.exp(-delta / T)):
            pass
        else:
            xs[idx], ys[idx], rs[idx] = ox, oy, or_
        
        T = max(T * cool, 0.002)
    
    xf = np.zeros(3*N)
    xf[0::3] = xs
    xf[1::3] = ys
    xf[2::3] = rs
    return xf


def construct():
    t0 = time.time()
    dl = t0 + 130.0
    state = {'val': -np.inf, 'res': None}
    
    widths = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
    best_warm = None
    
    for w in widths:
        if time.time() > dl - 20.0:
            break
        h = 2.0 - w
        rmax = min(w, h) / 2.0
        bounds = [(0, w), (0, h), (1e-4, rmax)] * N
        
        for sd in [0, 1, 2]:
            if time.time() > dl - 20.0:
                break
            x0 = _grid_init(w, h, seed=sd)
            
            try:
                r1 = minimize(lambda p: _obj(p, w, h, 3.0), x0,
                              jac=lambda p: _obj_grad(p, w, h, 3.0),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10})
                _update_best(r1.x, w, h, state)
                
                if time.time() < dl - 15.0:
                    r2 = minimize(lambda p: _obj(p, w, h, 15.0), r1.x,
                                  jac=lambda p: _obj_grad(p, w, h, 15.0),
                                  method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                    _update_best(r2.x, w, h, state)
                    if best_warm is None or np.sum(r2.x[2::3]) > np.sum(best_warm[2::3]):
                        best_warm = r2.x.copy()
            except Exception:
                pass
        
        # SA refinement
        if time.time() < dl - 25.0 and best_warm is not None:
            for s in [42, 123]:
                if time.time() > dl - 15.0:
                    break
                xw = best_warm.copy()
                if s != 42:
                    rng = np.random.default_rng(s)
                    for i in range(N):
                        xw[3*i] += rng.normal(0, 0.01)
                        xw[3*i+1] += rng.normal(0, 0.01)
                        xw[3*i+2] += rng.normal(0, 0.005)
                xw[0::3] = np.clip(xw[0::3], 0, w)
                xw[1::3] = np.clip(xw[1::3], 0, h)
                xw[2::3] = np.clip(xw[2::3], 1e-4, rmax)
                xsa = _sa(w, h, xw, iters=3000, seed=s, deadline=dl - 12.0)
                _update_best(xsa, w, h, state)
                
                if time.time() < dl - 10.0:
                    try:
                        rp = minimize(lambda p: _obj(p, w, h, 30.0), xsa,
                                      jac=lambda p: _obj_grad(p, w, h, 30.0),
                                      method='L-BFGS-B', bounds=bounds,
                                      options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-12})
                        _update_best(rp.x, w, h, state)
                    except Exception:
                        pass
    
    # Final polish
    if state['res'] is not None and time.time() < dl - 3.0:
        xs, ys, rs = state['res'][:, 0], state['res'][:, 1], state['res'][:, 2]
        mnx = np.min(xs - rs); mxx = np.max(xs + rs)
        mny = np.min(ys - rs); mxy = np.max(ys + rs)
        bw, bh = mxx - mnx, mxy - mny
        if bw > 0 and bh > 0:
            xp = np.zeros(3*N)
            xp[0::3] = xs - mnx
            xp[1::3] = ys - mny
            xp[2::3] = rs
            bp = [(0, bw), (0, bh), (1e-4, min(bw,bh)/2)] * N
            try:
                rp = minimize(lambda p: _obj(p, bw, bh, 80.0), xp,
                              jac=lambda p: _obj_grad(p, bw, bh, 80.0),
                              method='L-BFGS-B', bounds=bp,
                              options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-14})
                _update_best(rp.x, bw, bh, state)
                if time.time() < dl - 1.0:
                    rp2 = minimize(lambda p: _obj(p, bw, bh, 150.0), rp.x,
                                   jac=lambda p: _obj_grad(p, bw, bh, 150.0),
                                   method='L-BFGS-B', bounds=bp,
                                   options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15})
                    _update_best(rp2.x, bw, bh, state)
            except Exception:
                pass
    
    if state['res'] is None or state['val'] <= 0:
        # Fallback grid
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        res = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                res[idx, 0] = (i+0.5)*w/a
                res[idx, 1] = (j+0.5)*h/b
                res[idx, 2] = r
        return _scale(res)
    
    return _scale(state['res'].copy())
