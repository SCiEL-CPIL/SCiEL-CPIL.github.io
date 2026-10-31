"""Uncertainty inequality: tune H0/H4/H8 (H12 fixed by g(0)=0) to minimize
the upper bound on A(f)*A(Fourier(f)). Returns the three free coefficients."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 180.0  # hard wall-clock guard (evaluator kills at 210s)


def _hermite_coeffs():
    # Physicist Hermite polynomials (weight e^{-x^2}), monomial coefficient lists.
    return {
        0: [1.0],
        4: [12.0, 0.0, -48.0, 0.0, 8.0],
        8: [168.0, 0.0, -1680.0, 0.0, 3360.0, 0.0, -1344.0, 0.0, 128.0],
        12: [1330.0, 0.0, -12012.0, 0.0, 24024.0, 0.0, -15840.0, 0.0,
             3024.0, 0.0, -144.0, 0.0, 16.0],
    }


def _build_matrices():
    H = _hermite_coeffs()
    # a12 fixed so g(0)=0:  a0 + 12 a4 + 168 a8 + 1330 a12 = 0
    a12 = np.array([1.0 / 1330.0, -12.0 / 1330.0, -168.0 / 1330.0])

    def build_full(offset):
        # Q_j(x) = H_j(x) + a12_j * H12(x);  P(x) = sum_j a_j Q_j(x)
        Q = [np.array(H[j]) + a12[k] * np.array(H[12]) for k, j in enumerate([0, 4, 8])]
        M = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Qi, Qj = Q[i], Q[j]
                n = i + j + offset
                s = 0.0
                for u in range(len(Qi)):
                    v = n - u
                    if 0 <= v < len(Qj):
                        s += Qi[u] * Qj[v]
                M[i, j] = s
        return M

    const = np.sqrt(np.pi) / 2.0
    A = const * build_full(2)   # position:  int x^2 P_i P_j e^{-2x^2}
    B = const * build_full(0)   # Fourier:   int P_i P_j e^{-2x^2}
    return 0.5 * (A + A.T), 0.5 * (B + B.T)


_A, _B = _build_matrices()


def _product(v):
    v = np.asarray(v, dtype=float)
    num = float(v @ _A @ v)
    den = float(v @ _B @ v)
    if den <= 1e-300:
        return np.inf
    return num / den


def _objective(v):
    p = _product(v)
    return p if np.isfinite(p) else 1e30


def _secular_root(A, B, lo, hi):
    def s(lam):
        return np.linalg.det(A - lam * B)
    slo, shi = s(lo), s(hi)
    if slo == 0.0:
        return lo
    if shi == 0.0:
        return hi
    if slo * shi > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        smid = s(mid)
        if smid == 0.0:
            return mid
        if slo * smid <= 0:
            hi, shi = mid, smid
        else:
            lo, slo = mid, smid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def _eig_from_lambda(A, B, lam):
    M = A - lam * B
    w, V = np.linalg.eig(M)
    idx = int(np.argmin(np.abs(w)))
    v = V[:, idx].real
    bn = float(v @ B @ v)
    if abs(bn) < 1e-300:
        return None
    v = v / np.sqrt(abs(bn))
    k = int(np.argmax(np.abs(v)))
    if v[k] < 0:
        v = -v
    return v


def construct():
    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(best)

    def deadline_ok(margin=2.0):
        return (time.time() + margin) < _DEADLINE

    # Strategy 1: secular / generalized eigenvalue (exact 1D root find)
    try:
        w, _ = np.linalg.eig(np.linalg.solve(_B, _A))
        cand = sorted(set(round(x.real, 12) for x in w
                          if abs(x.imag) < 1e-9 and x.real > 0))
        for lam0 in cand:
            if not deadline_ok():
                break
            lo = max(lam0 * 0.5, lam0 - 1.0)
            hi = lam0 * 1.5 + 1.0
            lam = _secular_root(_A, _B, lo, hi)
            if lam is None:
                continue
            v = _eig_from_lambda(_A, _B, lam)
            if v is None:
                continue
            val = _objective(v)
            if val < best_val:
                best_val, best = val, v.copy()
    except Exception:
        pass

    # Strategy 2: robust multi-start Rayleigh-quotient minimization
    if deadline_ok(3.0):
        starts = [best.copy(),
                  np.array([-1.0, 0.0, 0.0]),
                  np.array([0.0, -1.0, 0.0]),
                  np.array([0.0, 0.0, -1.0]),
                  np.array([-1.0, -1.0, -1.0]),
                  np.array([1.0, 1.0, 1.0])]
        rng = np.random.default_rng(12345)
        for _ in range(40):
            starts.append(rng.normal(0.0, 1.0, 3))
        for s in starts:
            if not deadline_ok(3.0):
                break
            try:
                res = minimize(_objective, s, method="L-BFGS-B",
                               options={"maxiter": 400, "gtol": 1e-12})
                if res.fun < best_val:
                    best_val, best = float(res.fun), res.x.copy()
            except Exception:
                pass

    # Strategy 3: high-precision polish
    if deadline_ok(3.0):
        try:
            res = minimize(_objective, best, method="L-BFGS-B",
                           options={"maxiter": 2000, "gtol": 1e-15, "ftol": 1e-15})
            if res.fun < best_val:
                best_val, best = float(res.fun), res.x.copy()
        except Exception:
            pass

    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
