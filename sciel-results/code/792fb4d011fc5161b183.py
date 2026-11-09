"""Circle packing in unit square: maximize sum of 26 disjoint radii.

Strategy: smooth soft-min surrogate of the per-circle maximal radius, jointly
optimized over all positions and radii with L-BFGS-B (dense analytic gradient).
Warm start via coordinate-descent radius maximization; multi-restart with
annealed temperature; final exact radius polish. Deadline-guarded.
"""
import time
import math
import numpy as np

N = 26
DEADLINE = 150.0
MARGIN = 12.0          # seconds to stop new work before the kill
EPS = 1e-6


def _featurize():
    idx = np.arange(N)
    jx = np.tile(idx, (N, 1))
    jy = np.tile(idx, (1, N))
    mask = (jx != jy)
    pj, qj = np.nonzero(mask)
    return jx, jy, pj, qj


_JX, _JY, _PJ, _QJ = _featurize()


def _objective_grad(v, T):
    """Return (objective, grad) for the smooth soft-min surrogate.

    v = [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
    objective = sum_i softmin_i, softmin_i = -T*logsumexp(-(d_k - r_i)/T)
    """
    X = v[:N]
    Y = v[N:2 * N]
    R = v[2 * N:]
    grad = np.zeros(3 * N)

    # --- per-circle constraint distances and weights -----------------------
    # wall: d = [x, 1-x, y, 1-y];  d(r_i)/dr_i = 0
    wwall = np.exp((np.stack([X, 1.0 - X, Y, 1.0 - Y], axis=1) - R[:, None]) / T)
    # pair: d = dist_ij - r_j ;  d(r_i)/dr_i = -1
    dx = X[_PJ] - X[_QJ]
    dy = Y[_PJ] - Y[_QJ]
    dist = np.sqrt(dx * dx + dy * dy)
    dp = dist - R[_QJ]
    wp = np.exp(dp / T)
    wp[_PJ, _QJ] = 0.0

    lse_w = np.log(np.sum(wwall, axis=1) + 1e-300)
    lse_p = np.log(np.sum(wp, axis=1) + 1e-300)
    lse = np.logaddexp(lse_w, lse_p)
    S = -T * lse

    sw = np.exp(lse_w - lse)          # softmax over wall constraints
    sp = np.exp(lse_p - lse)          # softmax over pair constraints

    # --- dS/dx_i (position) -----------------------------------------------
    # wall: 0.5*[1,-1,0,0]
    gdx = 0.5 * (sw[:, 0] - sw[:, 1])
    gdy = 0.5 * (sw[:, 2] - sw[:, 3])
    # pair: sum_j sp_ij * (X_i - X_j)/dist
    gdx += np.zeros(N)
    gdy += np.zeros(N)
    np.add.at(gdx, _PJ, sp * dx / dist)
    np.add.at(gdy, _PJ, sp * dy / dist)
    grad[:N] = gdx
    grad[N:2 * N] = gdy

    # --- dS/dr_i -----------------------------------------------------------
    # wall: 0 ; pair: sum_j sp_ij * (-1)  = -sum_j sp_ij
    gdr = -np.sum(sp, axis=1)
    grad[2 * N:] = gdr

    return float(np.sum(S)), grad


def _objective_only(v, T):
    S, _ = _objective_grad(v, T)
    return S


def _polish(X, Y, R, iters=200):
    """Coordinate-descent radius maximization (feasible, near-optimal)."""
    R = R.copy()
    for _ in range(iters):
        for i in range(N):
            rmin = min(X[i], 1.0 - X[i], Y[i], 1.0 - Y[i])
            for j in range(N):
                if j != i:
                    rmin = min(rmin, math.hypot(X[i] - X[j], Y[i] - Y[j]) - R[j])
            R[i] = max(rmin, 0.0)
    return R


def _score(X, Y, R):
    return float(np.sum(R))


def _make_initial(seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.12, 0.88, N)
    Y = rng.uniform(0.12, 0.88, N)
    R = np.full(N, 0.03)
    return X, Y, R


def _lbfgs(X, Y, R, T, time_left):
    from scipy.optimize import minimize
    v0 = np.concatenate([X, Y, R])

    def fun(v):
        return -_objective_only(v, T)

    def gfun(v):
        _, g = _objective_grad(v, T)
        return -g

    res = minimize(fun, v0, jac=gfun, method='L-BFGS-B',
                   options={'maxiter': 120, 'maxls': 16, 'gtol': 1e-4,
                            'ftol': 1e-9, 'disp': False})
    Xn = res.x[:N].copy()
    Yn = res.x[N:2 * N].copy()
    Rn = res.x[2 * N:].copy()
    Rn = _polish(Xn, Yn, Rn, 60)
    return Xn, Yn, Rn


def construct():
    t0 = time.time()

    def left():
        return DEADLINE - MARGIN - (time.time() - t0)

    # ---- warm start: coordinate-descent radius maximization --------------
    X, Y, R = _make_initial(0)
    R = _polish(X, Y, R, 300)
    best = (X.copy(), Y.copy(), R.copy())
    best_score = _score(X, Y, R)

    # ---- multi-restart smooth-surrogate joint optimization ---------------
    temps = [1.0, 0.3, 0.1, 0.03, 0.01, 0.005]
    attempts = 0
    seed = 1
    for T in temps:
        while left() > 25.0:
            attempts += 1
            Xs, Ys, Rs = _make_initial(seed)
            Rs = _polish(Xs, Ys, Rs, 120)
            Xs, Ys, Rs = _lbfgs(Xs, Ys, Rs, T, left())
            sc = _score(Xs, Ys, Rs)
            if sc > best_score:
                best_score = sc
                best = (Xs.copy(), Ys.copy(), Rs.copy())
            # local refinement from the current best
            Xr, Yr, Rr = _lbfgs(best[0].copy(), best[1].copy(), best[2].copy(), T, left())
            scr = _score(Xr, Yr, Rr)
            if scr > best_score:
                best_score = scr
                best = (Xr.copy(), Yr.copy(), Rr.copy())
            seed += 1
            if left() < 25.0:
                break

    # ---- final exact polish on the best ----------------------------------
    if left() > 5.0:
        Rf = _polish(best[0].copy(), best[1].copy(), best[2].copy(), 500)
        best = (best[0].copy(), best[1].copy(), Rf)
        best_score = _score(best[0], best[1], best[2])

    # ---- validate & clamp -------------------------------------------------
    X, Y, R = best
    R = np.clip(R, 0.0, None)
    X = np.clip(X, R, 1.0 - R)
    Y = np.clip(Y, R, 1.0 - R)
    R = np.minimum(R, np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y)))
    R = np.maximum(R, 0.0)

    out = np.empty((N, 3), dtype=float)
    out[:, 0] = X
    out[:, 1] = Y
    out[:, 2] = R
    return out
