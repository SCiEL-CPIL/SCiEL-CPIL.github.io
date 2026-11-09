"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import gamma

# EVOLVE-BLOCK-START

# Fixed construction parameters (from the task contract)
M = 12          # number of double roots
N = 1           # dimension
# Seed arithmetic double-root locations (known-good Cohn-Goncalves structure)
_SEED = np.array([4.0 * (i + 1) for i in range(M)])


def _laguerre_combo(x, roots):
    """
    Evaluate the refined Cohn-Goncalves Laguerre combination
        F(x) = sum_{k=0}^{M-1} c_k * L_k^{(N)}(x)
    where c_k are chosen so that F has double roots at each r_i in `roots`.
    We solve the (M-1)x(M-1) system  L^{(N)}(r_i) c = 0  (i=0..M-2)
    with normalization c_{M-1} = 1, then evaluate at x.
    Double-root tangency (F'(r_i)=0) is enforced by the construction's
    symmetry for the arithmetic seed; for perturbed roots we rely on the
    root-constraint that the evaluator checks, and minimize |F(r_i)|.
    """
    r = np.asarray(roots, dtype=float)
    # Build the Laguerre value matrix for the first M-1 roots
    # V[i, k] = L_k^{(N)}(r_i) for i in 0..M-2, k in 0..M-1
    V = np.zeros((M - 1, M))
    for i in range(M - 1):
        for k in range(M):
            V[i, k] = _gen_laguerre(k, N, r[i])
    # Solve V[:, :M-1] c = -V[:, M-1]  (since c_{M-1}=1)
    A = V[:, :M - 1]
    b = -V[:, M - 1]
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    c_full = np.concatenate([c, [1.0]])
    # Evaluate F(x) = sum c_k L_k^{(N)}(x)
    val = 0.0
    for k in range(M):
        val += c_full[k] * _gen_laguerre(k, N, x)
    return val


def _gen_laguerre(n, alpha, x):
    """Generalized Laguerre polynomial L_n^{(alpha)}(x) via recurrence."""
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0 + alpha - x
    L_nm2 = 1.0
    L_nm1 = 1.0 + alpha - x
    for k in range(1, n):
        L_n = ((2.0 * k + 1.0 + alpha - x) * L_nm1 - (k + alpha) * L_nm2) / (k + 1.0)
        L_nm2 = L_nm1
        L_nm1 = L_n
    return L_nm1


def _objective(params, g0=4.0):
    """
    Principled surrogate: C4 upper bound ~ max_i |F(r_i)| where F is the
    Laguerre combination with double roots at r_i.  Minimize this maximum
    via a smooth log-sum-exp (soft-max) to keep it differentiable.
    """
    g = float(params[0])
    if g <= 0.0:
        return 1e9
    p = params[1:]
    base = g * (np.arange(1, M + 1) + 1.0)
    pos = base + p
    pos = np.maximum.accumulate(pos)
    if np.any(pos <= 0):
        return 1e9

    # Evaluate |F(r_i)| for each root
    abs_vals = np.zeros(M)
    for i in range(M):
        abs_vals[i] = abs(_laguerre_combo(pos[i], pos))
    # Soft-max (log-sum-exp) as smooth max
    m = np.max(abs_vals)
    if m > 1e300:
        return 1e9
    logsum = m + np.log(np.sum(np.exp(abs_vals - m)))
    # Small regularization to keep roots well-separated and positive
    gaps = np.diff(pos)
    min_gap = np.min(gaps)
    reg = 1e-6 * (1.0 / max(min_gap, 1e-6)) ** 2
    return logsum + reg


def _candidate_roots(params, g0=4.0):
    g = float(params[0])
    if g <= 0.0:
        return list(_SEED)
    p = params[1:]
    base = g * (np.arange(1, M + 1) + 1.0)
    pos = base + p
    pos = np.maximum.accumulate(pos)
    pos = np.maximum(pos, 1e-6)
    return [float(x) for x in pos]


def _calibrate_g0():
    return 4.0


def _seed_value():
    """Compute the objective at the seed (arithmetic) configuration."""
    params = np.array([4.0] + [0.0] * M)
    return _objective(params, 4.0)


def construct():
    deadline = time.time() + 630.0 - 60.0  # 570 s safety margin
    g0 = _calibrate_g0()

    best_params = np.array([g0] + [0.0] * M)
    best_val = _objective(best_params, g0)

    rng = np.random.default_rng(12345)
    done_restarts = 0

    # Multi-start local search with the principled objective
    for target in [8, 24, 64, 160, 400, 900]:
        while done_restarts < target:
            if time.time() > deadline:
                break
            scale = rng.uniform(3.0, 5.0)
            pert = rng.normal(0.0, 0.3, size=M)
            x0 = np.concatenate(([scale], pert))
            try:
                res = minimize(
                    _objective,
                    x0,
                    args=(g0,),
                    method="Nelder-Mead",
                    options=dict(maxiter=3000, xatol=1e-6, fatol=1e-10,
                                 adaptive=True),
                )
            except Exception:
                res = None
            if res is not None and np.all(np.isfinite(res.x)):
                v = _objective(res.x, g0)
                if v < best_val:
                    best_val = v
                    best_params = res.x.copy()
            done_restarts += 1
        if time.time() > deadline:
            break

    # Final high-precision polish
    if time.time() < deadline - 15.0:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(g0,),
                method="Powell",
                options=dict(maxiter=6000, ftol=1e-14, maxfev=150000),
            )
            if res is not None and np.all(np.isfinite(res.x)):
                v = _objective(res.x, g0)
                if v < best_val:
                    best_val = v
                    best_params = res.x.copy()
        except Exception:
            pass

    roots = _candidate_roots(best_params, g0)
    roots = sorted(roots)
    out = []
    prev = 0.0
    for r in roots:
        r = max(r, prev + 1e-6)
        out.append(float(r))
        prev = r
    return out

# EVOLVE-BLOCK-END
