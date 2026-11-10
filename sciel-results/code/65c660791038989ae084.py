# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv

K = 10
N = 2 * K + 1  # 21
M = np.arange(-K, K + 1)

# ---------- Quadrature (matches evaluator: 70 GL in cos(theta), 130 phi) ----------
GL = 70
NP = 130
_xgl, _wgl = np.polynomial.legendre.leggauss(GL)
# leggauss nodes in (-1,1); cos(theta)=x, dOmega = dphi dx
_x = _xgl.copy()
_w = _wgl.copy()
_phi = (np.arange(NP) / NP) * (2.0 * np.pi)
_sinphi = np.sin(_phi)
_cosphi = np.cos(_phi)
_cosm = np.cos(np.outer(M, _phi))          # (21, NP)
_sinm = np.sin(np.outer(M, _phi))          # (21, NP)
_wfull = np.repeat(_w, NP)                 # (70*130,)

# ---------- Spherical harmonics ----------
def _norms(m):
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    return np.sqrt(21.0 / (4.0 * np.pi) * np.exp(
        np.lgamma(10 - np.abs(m) + 1) - np.lgamma(10 + np.abs(m) + 1)))

def _build_Y():
    # Y[m, (i,j)] with x = cos(theta), phi
    Y = np.zeros((N, GL * NP))
    for i, m in enumerate(M):
        am = abs(m)
        pm = lpmv(am, 10, _x) * _norms(am)          # (GL,) Condon-Shortley phase
        if m > 0:
            ang = np.sqrt(2.0) * _cosm[i]            # (NP,)
        elif m < 0:
            ang = np.sqrt(2.0) * _sinm[i]            # (NP,)
        else:
            ang = np.ones(NP)
        Y[i] = np.repeat(pm, NP) * ang               # (GL*NP,)
    return Y

Y = _build_Y()
Y2 = Y * Y                                          # (21, 70*130)

# ---------- Precompute per-quadrature-point basis values ----------
# Yq[c, q]
Yq = Y  # alias

def _eval_f(c):
    return Yq @ c                                    # (70*130,)

def _objective_and_derivs(c):
    f = Yq @ c
    f2 = f * f
    f3 = f2 * f
    a = np.sum(f2 * _wfull)                          # ||f||_2^2
    b = np.sum(f3 * f * _wfull)                      # ||f||_4^4
    if a <= 0.0:
        return None
    # integrals
    I2 = Yq @ (f2 * _wfull)                          # (21,) int f^2 Y_m
    I1 = Yq @ (f * _wfull)                           # (21,) int f Y_m
    I3 = Yq @ (f3 * _wfull)                          # (21,) int f^3 Y_m
    val = b / a
    grad = (4.0 / a) * I3 - (2.0 * b / (a * a)) * I1
    # Hessian
    H = (12.0 / a) * (Y2 * I2) @ _wfull              # (21,21) int f^2 Y_i Y_j
    H -= (4.0 / (a * a)) * np.outer(I3, I1)
    H -= (4.0 / (a * a)) * np.outer(I1, I3)
    H += (2.0 * b / (a ** 3)) * np.outer(I1, I1)
    H = 0.5 * (H + H.T)
    return val, grad, H

# ---------- Trust region on the unit sphere ----------
def _tangent_step(x, d, alpha):
    # step along geodesic: x + alpha*d, then reproject to sphere
    y = x + alpha * d
    nrm = np.linalg.norm(y)
    if nrm < 1e-14:
        y = x
    else:
        y = y / nrm
    return y

def _trust_region(x, max_iter, time_budget, start_delta):
    val, grad, H = _objective_and_derivs(x)
    if val is None:
        return x, 1e30
    best = x
    best_val = val
    delta = start_delta
    # initial trust-region subproblem
    for it in range(max_iter):
        # solve min (g^T d + 0.5 d^T H d) s.t. ||d|| <= delta, d in tangent space
        # project H, g to tangent space
        # tangent basis via QR of identity - x x^T
        P = np.eye(N) - np.outer(x, x)
        g_t = P @ grad
        H_t = P @ H @ P
        # find tangent direction
        # Solve trust region subproblem using eigen-decomp in tangent space
        # Build orthonormal tangent basis
        # Use SVD of P
        U, s, Vt = np.linalg.svd(P)
        # tangent space = span of Vt rows where s > tol
        tol = 1e-10
        idx = s > tol
        Vt_t = Vt[idx]                 # (N-1, N)
        g_T = Vt_t @ g_t               # (N-1,)
        H_T = Vt_t @ H_t @ Vt_t.T      # (N-1, N-1)
        H_T = 0.5 * (H_T + H_T.T)
        # solve: min g_T^T z + 0.5 z^T H_T z s.t. ||z|| <= delta
        # eigen
        w, V = np.linalg.eigh(H_T)
        # find lambda so that z = (H_T + lambda I)^{-1} (-g_T), ||z|| = delta
        gV = V.T @ g_T
        # find lambda
        def znorm(lam):
            A = w + lam
            z = -gV / A
            return np.linalg.norm(z)
        # bracket
        lo = -w.min() + 1e-12
        hi = 1e-6
        if znorm(hi) > delta:
            # increase hi
            hi = 1e-6
            for _ in range(100):
                if znorm(hi) <= delta:
                    break
                hi *= 2.0
        else:
            # decrease lo until znorm(lo) < delta
            lo = -w.min() + 1e-12
            for _ in range(100):
                if znorm(lo) < delta:
                    break
                lo = lo - (hi - lo) * 0.5
        # bisection
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if znorm(mid) > delta:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        A = w + lam
        z = -gV / A
        # back to ambient
        d = Vt_t.T @ z
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-14:
            # steepest descent direction
            d = -g_t
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-14:
                break
            d = d / d_norm * delta
        else:
            d = d / d_norm * min(1.0, delta / d_norm)
        # backtracking line search
        f_new = None
        alpha = 1.0
        for ls in range(40):
            x_new = _tangent_step(x, d, alpha)
            v, g, h = _objective_and_derivs(x_new)
            if v is None:
                alpha *= 0.5
                continue
            pred = val + alpha * (grad @ d)
            if v <= pred - 1e-12 * max(1.0, abs(val)):
                f_new = (v, g, h)
                break
            alpha *= 0.5
        if f_new is None:
            # accept small step anyway to make progress
            x_new = _tangent_step(x, d, alpha)
            v, g, h = _objective_and_derivs(x_new)
            if v is None:
                break
            f_new = (v, g, h)
        v, g, h = f_new
        # update trust region
        actual = val - v
        if actual > 1e-4 * delta:
            delta = max(1e-4, 0.25 * delta)
        elif actual > 0.5 * delta:
            delta = min(10.0, 2.0 * delta)
        x = x_new
        val, grad, H = v, g, h
        if val < best_val - 1e-13:
            best_val = val
            best = x.copy()
        if it % 5 == 0 and (time.time() - _t0) > time_budget:
            break
    return best, best_val

# ---------- Main ----------
def rudin_problem10() -> np.ndarray:
    global _t0
    _t0 = time.time()
    deadline = 630.0 - 30.0  # safety margin
    best = None
    best_val = np.inf

    # multi-start
    rng = np.random.default_rng(12345)
    n_starts = 12
    starts = []
    # all-ones (baseline)
    starts.append(np.ones(N))
    # symmetric patterns
    for _ in range(n_starts - 1):
        v = rng.standard_normal(N)
        starts.append(v / np.linalg.norm(v))

    for s in starts:
        if (time.time() - _t0) > deadline:
            break
        x0 = s / np.linalg.norm(s)
        x, v = _trust_region(x0, max_iter=60, time_budget=deadline - (time.time() - _t0), start_delta=0.5)
        if v < best_val:
            best_val = v
            best = x.copy()

    # final polish: more iterations from best
    if best is not None:
        x, v = _trust_region(best, max_iter=80, time_budget=deadline - (time.time() - _t0), start_delta=0.1)
        if v < best_val:
            best_val = v
            best = x.copy()

    if best is None:
        best = np.ones(N)
    # normalize (scale-invariant, keep moderate magnitude)
    best = best / np.linalg.norm(best)
    # sanity: enforce validity bounds
    mx = np.max(np.abs(best))
    if mx < 1e-3:
        best = best * (1.0 / mx)
    if mx > 1e5:
        best = best * (1e5 / mx)
    if not np.all(np.isfinite(best)):
        best = np.ones(N)
    return best.astype(np.float64)
# EVOLVE-BLOCK-END
