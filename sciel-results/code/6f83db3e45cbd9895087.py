import time
import math
import numpy as np

_N = 24
_SQRN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 140.0  # safety margin under the 150 s wall-clock kill

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly(n=_N):
    """Rudin-Shapiro +/-1 sequence truncated to n coefficients."""
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

# Precomputed DFT matrix for a given sample count.
_ZCACHE = {}
def _get_Z(M):
    Z = _ZCACHE.get(M)
    if Z is None:
        j = np.arange(M)
        k = np.arange(_N)
        J, K = np.meshgrid(j, k, indexing='ij')
        Z = np.exp(1j * 2.0 * np.pi * J * K / M)
        _ZCACHE[M] = Z
    return Z

# ---------- Smooth surrogate: soft-max (log-sum-exp) of |P(z)| ----------
def _softmax(vals, beta):
    """Soft-max of |vals|: (1/beta) log sum exp(beta |vals|).
    As beta -> inf, approaches max|vals|; finite beta gives a smooth,
    differentiable surrogate that is sensitive to the whole peak region."""
    mags = np.abs(vals)
    b = beta * mags
    bmax = b.max()
    s = np.exp(b - bmax).sum()
    return (bmax + math.log(s)) / beta

# ---------- Fast local search with incremental DFT + smooth objective ----------
def _fast_local_search(coeffs, M=4096, beta=80.0, max_iters=4000):
    Z = _get_Z(M)
    Zc = [Z[:, k] for k in range(_N)]
    coeffs_arr = np.array(coeffs, dtype=float)
    vals = Z @ coeffs_arr

    def cur_obj():
        return _softmax(vals, beta)

    cur_objv = cur_obj()
    best_coeffs = coeffs[:]
    best_obj = cur_objv
    no_improve = 0

    for it in range(max_iters):
        if _time_left() < 8.0:
            break
        improved = False
        best_move = None
        best_new = cur_objv

        # single flips
        for k in range(_N):
            delta = -2.0 * coeffs_arr[k] * Zc[k]
            nv = vals + delta
            ov = _softmax(nv, beta)
            if ov < best_new:
                best_new = ov
                best_move = (k,)

        # double flips
        if best_move is None:
            for k1 in range(_N):
                d1 = -2.0 * coeffs_arr[k1] * Zc[k1]
                for k2 in range(k1 + 1, _N):
                    delta = d1 - 2.0 * coeffs_arr[k2] * Zc[k2]
                    nv = vals + delta
                    ov = _softmax(nv, beta)
                    if ov < best_new:
                        best_new = ov
                        best_move = (k1, k2)

        # triple flips (to escape 2-flip traps)
        if best_move is None and _time_left() > 12.0:
            for k1 in range(_N):
                d1 = -2.0 * coeffs_arr[k1] * Zc[k1]
                for k2 in range(k1 + 1, _N):
                    d2 = d1 - 2.0 * coeffs_arr[k2] * Zc[k2]
                    for k3 in range(k2 + 1, _N):
                        delta = d2 - 2.0 * coeffs_arr[k3] * Zc[k3]
                        nv = vals + delta
                        ov = _softmax(nv, beta)
                        if ov < best_new:
                            best_new = ov
                            best_move = (k1, k2, k3)

        if best_move is not None:
            for k in best_move:
                old = coeffs_arr[k]
                coeffs_arr[k] = -old
                vals += -2.0 * old * Zc[k]
            cur_objv = best_new
            best_coeffs = coeffs_arr.astype(int).tolist()
            best_obj = cur_objv
            improved = True
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 8:
                break
    return best_coeffs, best_obj

# ---------- Precise evaluation: dense FFT + golden-section refinement ----------
def _eval_precise(coeffs, M=32768):
    c = np.zeros(M, dtype=complex)
    c[:_N] = coeffs
    vals = M * np.fft.ifft(c)
    mags = np.abs(vals)
    top_idx = np.argsort(mags)[-12:]
    max_val = float(mags[top_idx[0]])

    k = np.arange(_N)
    c_arr = np.array(coeffs, dtype=complex)

    def P_mag(theta):
        zk = np.exp(1j * theta * k)
        return float(np.abs(np.dot(c_arr, zk)))

    for idx in top_idx:
        theta0 = 2.0 * np.pi * idx / M
        a = theta0 - math.pi / M
        b = theta0 + math.pi / M
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c_ = b - gr * (b - a)
        d_ = a + gr * (b - a)
        fc = P_mag(c_)
        fd = P_mag(d_)
        for _ in range(50):
            if fc < fd:
                b, d_, fd = d_, c_, fc
                c_ = b - gr * (b - a)
                fc = P_mag(c_)
            else:
                a, c_, fc = c_, d_, fd
                d_ = a + gr * (b - a)
                fd = P_mag(d_)
        pv = max(fc, fd)
        if pv > max_val:
            max_val = pv
    return max_val

def construct():
    t_start = time.time()
    np.random.seed(42)

    best_coeffs = _rs_poly()
    best_val = _eval_precise(best_coeffs)

    # 1. Local search from RS (smooth surrogate, moderate resolution)
    c1, _ = _fast_local_search(best_coeffs, M=4096, beta=80.0, max_iters=6000)
    v1 = _eval_precise(c1)
    if v1 < best_val:
        best_coeffs, best_val = c1, v1

    # 2. Random / perturbed restarts
    restart_count = 0
    while _time_left() > 25.0:
        if np.random.rand() < 0.4:
            c_rand = best_coeffs[:]
            nf = np.random.randint(1, 6)
            for _ in range(nf):
                c_rand[np.random.randint(_N)] *= -1
        else:
            c_rand = [1 if np.random.rand() > 0.5 else -1 for _ in range(_N)]

        c2, _ = _fast_local_search(c_rand, M=2048, beta=60.0, max_iters=4000)
        v2 = _eval_precise(c2)
        if v2 < best_val:
            best_coeffs, best_val = c2, v2
            # follow-up polish from the new best
            if _time_left() > 20.0:
                c2b, _ = _fast_local_search(best_coeffs, M=4096, beta=80.0, max_iters=3000)
                v2b = _eval_precise(c2b)
                if v2b < best_val:
                    best_coeffs, best_val = c2b, v2b
        restart_count += 1

    # 3. Final high-resolution polish
    if _time_left() > 20.0:
        c3, _ = _fast_local_search(best_coeffs, M=8192, beta=100.0, max_iters=6000)
        v3 = _eval_precise(c3)
        if v3 < best_val:
            best_coeffs, best_val = c3, v3

    # 4. High-precision final evaluation (not returned, but used to confirm)
    final_val = _eval_precise(best_coeffs, M=65536)

    # Ensure valid output: 24 ints in {+1,-1}
    return [1 if x > 0 else -1 for x in best_coeffs]
