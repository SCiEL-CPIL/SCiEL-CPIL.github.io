"""Optimize nonnegative step heights on 600 equal intervals of [-1/4, 1/4]
to minimize a normalized maximum of the autoconvolution (upper bound on C1)."""
import time
import numpy as np

N = 600
H = (1.0 / 4.0) * 2 / N  # interval width = (1/2)/600
START = time.time()
DEADLINE = 110.0  # safety margin below 120s

def _autoconv(f):
    # full linear autoconvolution of the step samples, scaled by H
    c = np.convolve(f, f, mode='full') * H
    return c

def _peak_to_area(c):
    # normalized maximum: peak / (integral of c)
    area = c.sum() * H
    if area <= 1e-12:
        return np.inf
    return c.max() / area

def _soft_max(c, k):
    m = c.max()
    if m <= 0:
        return 0.0
    z = c / m
    return m * np.log(np.sum(np.exp(k * (z - 1.0)))) / k  # approx; use stable below

def _soft_max_stable(c, k):
    m = c.max()
    if m <= 0:
        return 0.0
    a = k * (c - m)
    a = np.clip(a, -50, 50)
    return m + (1.0 / k) * np.log(np.sum(np.exp(a)))

def _objective(f, k):
    c = _autoconv(f)
    area = c.sum() * H
    if area <= 1e-12:
        return np.inf
    return _soft_max_stable(c, k) / area

def _grad_obj(f, k):
    # gradient of soft_max(c)/area w.r.t. f, via c = H * (f * f)
    c = _autoconv(f)
    area = c.sum() * H
    if area <= 1e-12:
        return np.zeros_like(f), np.inf
    m = c.max()
    a = k * (c - m)
    a = np.clip(a, -50, 50)
    e = np.exp(a)
    sm = m + (1.0 / k) * np.log(np.sum(e))
    # d soft_max / d c_i = e_i / sum(e)
    dc = e / np.sum(e)
    # d c / d f : c = H * convolve(f,f,full); d c_j / d f_i = H * (f_{j-i} + f_{j-i})... use gradient via correlation
    # c_j = H * sum_i f_i f_{j-i} (with full length). Use np.correlate for gradient.
    # grad_f = H * (2 * (f * c)) truncated to length N
    full_len = 2 * N - 1
    # c has length full_len. convolve(f,f,full)[j] = sum_i f[i] f[j-i]
    # d/dc_j df_i = f[j-i] if valid. So grad_f[i] = H * sum_j dc_j * f[j-i]
    # = H * (dc * f) correlation. Use np.convolve(dc, f, 'full') then take middle N.
    g = H * 2.0 * np.convolve(dc, f, mode='full')
    g = g[N - 1:N - 1 + N]
    # divide by area, and subtract (sm/area^2)*d area/d f ; area = H*sum(c) = H*(f*f) dot 1
    # d area / d f = H * 2 * (f * ones) truncated = 2H * (convolve(f, ones)) middle
    da = H * 2.0 * np.convolve(np.ones_like(dc), f, mode='full')
    da = da[N - 1:N - 1 + N]
    grad = (g * area - sm * da) / (area * area)
    return grad, sm / area

def _project_nonneg_unitL1(f):
    f = np.maximum(f, 0.0)
    s = f.sum()
    if s <= 1e-15:
        f = np.ones_like(f)
        s = f.sum()
    f = f / s
    return f

def _line_search(f, g, obj_cur, k):
    # backtracking
    step = 1.0
    for _ in range(30):
        f_new = _project_nonneg_unitL1(f - step * g)
        obj_new = _objective(f_new, k)
        if obj_new < obj_cur:
            return f_new, obj_new
        step *= 0.5
    return f, obj_cur

def construct():
    global DEADLINE
    rng = np.random.default_rng(12345)
    # start from a smooth bump (Gaussian-ish) on full support, normalized L1=1
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + H / 2
    f = np.exp(-((x / 0.10) ** 2))
    f = _project_nonneg_unitL1(f)
    best_f = f.copy()
    best_obj = _objective(best_f, 8.0)

    k = 8.0
    lr_base = 0.5
    it = 0
    while time.time() < DEADLINE:
        it += 1
        g, obj = _grad_obj(best_f, k)
        if not np.isfinite(obj):
            break
        # scale gradient
        g = g / (np.linalg.norm(g) + 1e-12)
        f_new, obj_new = _line_search(best_f, g, obj, k)
        if obj_new < best_obj:
            best_f = f_new
            best_obj = obj_new
        # anneal k upward for sharper max (more exact)
        if it % 200 == 0:
            k = min(k * 1.3, 60.0)
        # small random restarts to escape plateaus
        if it % 400 == 0:
            f_try = best_f + 0.05 * rng.standard_normal(N)
            f_try = _project_nonneg_unitL1(f_try)
            o_try = _objective(f_try, k)
            if o_try < best_obj:
                best_f = f_try
                best_obj = o_try
        if time.time() > DEADLINE - 20:
            # final high-precision polish with large k, small steps
            break

    # final polish: exact max (k large), fine steps
    k = 80.0
    for _ in range(2000):
        if time.time() > DEADLINE - 3:
            break
        g, obj = _grad_obj(best_f, k)
        if not np.isfinite(obj):
            break
        g = g / (np.linalg.norm(g) + 1e-12)
        f_new, obj_new = _line_search(best_f, g, obj, k)
        if obj_new < best_obj:
            best_f = f_new
            best_obj = obj_new
        else:
            break

    return [float(v) for v in best_f]
