import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time

def rudin_problem10() -> np.ndarray:
    K = 10
    N_THETA = 70
    N_PHI = 130
    N_DIM = 2 * K + 1  # 21

    start_time = time.time()
    DEADLINE = 600.0
    SAFETY_MARGIN = 12.0

    def elapsed():
        return time.time() - start_time

    def time_left():
        return DEADLINE - SAFETY_MARGIN - elapsed()

    # ---------- Quadrature ----------
    x, w_theta = np.polynomial.legendre.leggauss(N_THETA)  # cos(theta) nodes in [-1,1]
    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)
    w_phi = 2.0 * np.pi / N_PHI
    # weights[i,j] = w_theta[i] * w_phi
    weights = np.kron(w_theta, np.full(N_PHI, w_phi))  # shape (N_THETA*N_PHI,)

    # ---------- Build Y matrix (N_POINTS x 21) of real orthonormal harmonics ----------
    N_POINTS = N_THETA * N_PHI
    Y = np.zeros((N_POINTS, N_DIM), dtype=np.float64)

    N_m = np.zeros(K + 1)
    for m in range(K + 1):
        N_m[m] = np.sqrt((2 * K + 1) / (4.0 * np.pi) * factorial(K - m) / factorial(K + m))

    # P_10^m(x) for m=0..10 (Condon-Shortley phase included by lpmv)
    P = np.array([lpmv(m, K, x) for m in range(K + 1)])  # (K+1, N_THETA)

    # Broadcast grids
    x_col = x[:, np.newaxis]            # (N_THETA,1)
    phi_row = phi[np.newaxis, :]        # (1, N_PHI)

    for m in range(K + 1):
        idx_pos = m + K
        Pm = P[m][:, np.newaxis]        # (N_THETA,1)
        if m == 0:
            Ym = (N_m[0] * Pm) * np.ones((1, N_PHI))
            Y[:, idx_pos] = Ym.reshape(-1)
        else:
            N_val = np.sqrt(2.0) * N_m[m]
            cos_mp = np.cos(m * phi_row)  # (1, N_PHI)
            sin_mp = np.sin(m * phi_row)
            Y[:, idx_pos] = (N_val * Pm * cos_mp).reshape(-1)
            Y[:, -m + K] = (N_val * Pm * sin_mp).reshape(-1)

    # ---------- Objective: minimize R^2 = A / B^2, with A=sum w f^4, B=sum w f^2 ----------
    # Scale invariant; we restrict to unit sphere ||c||=1.

    def eval_f(c):
        return Y @ c  # (N_POINTS,)

    def objective(c):
        f = Y @ c
        f2 = f * f
        A = np.dot(weights, f2 * f2)
        B = np.dot(weights, f2)
        if B < 1e-12:
            return 1e10
        return A / (B * B)

    def objective_grad(c):
        f = Y @ c
        f2 = f * f
        A = np.dot(weights, f2 * f2)
        B = np.dot(weights, f2)
        if B < 1e-12:
            return 1e10, np.zeros_like(c)
        val = A / (B * B)
        w_f3 = weights * (f * f2)
        w_f = weights * f
        dA = 4.0 * (Y.T @ w_f3)
        dB = 2.0 * (Y.T @ w_f)
        grad = dA / (B * B) - 2.0 * A * dB / (B ** 3)
        return val, grad

    # ---------- Riemannian gradient flow on the unit sphere ----------
    # Minimize F(c)=R^2 subject to ||c||=1.
    # Tangent gradient: g_t = grad - (grad . c) c
    # Update: c_new = (c - lr * g_t); renormalize. Use adaptive step with backtracking.
    def sphere_flow(c0, max_iter=2000, lr0=0.05, tol=1e-13, time_budget=30.0):
        c = c0 / np.linalg.norm(c0)
        F, grad = objective_grad(c)
        prev_F = F
        for it in range(max_iter):
            if it % 50 == 0 and (time.time() - start_time) > start_time + time_budget:
                break
            # Tangent gradient
            gt = grad - (np.dot(grad, c)) * c
            gnorm = np.linalg.norm(gt)
            if gnorm < tol:
                break
            # Adaptive step
            lr = lr0
            for _ in range(10):
                c_trial = c - lr * gt
                n = np.linalg.norm(c_trial)
                if n < 1e-12:
                    lr *= 0.5
                    continue
                c_trial = c_trial / n
                F_trial = objective(c_trial)
                if F_trial < F:
                    break
                lr *= 0.5
            if lr < 1e-12:
                break
            c = c_trial
            F = F_trial
            # Convergence on objective
            if abs(prev_F - F) < tol * max(1.0, F):
                break
            prev_F = F
        return c, F

    # ---------- Structured diverse starts ----------
    rng = np.random.default_rng(12345)
    starts = []
    # m=0 only
    c0 = np.zeros(N_DIM); c0[K] = 1.0
    starts.append(c0)
    # Single-m starts (cos and sin for each m>0)
    for m in range(1, K + 1):
        cpos = np.zeros(N_DIM); cpos[m + K] = 1.0
        starts.append(cpos)
        cneg = np.zeros(N_DIM); cneg[-m + K] = 1.0
        starts.append(cneg)
    # Pairs of modes
    for m in range(1, K + 1):
        cp = np.zeros(N_DIM); cp[m + K] = 1.0; cp[-m + K] = 1.0
        cp /= np.linalg.norm(cp); starts.append(cp)
        cm = np.zeros(N_DIM); cm[m + K] = 1.0; cm[-m + K] = -1.0
        cm /= np.linalg.norm(cm); starts.append(cm)
    # Random starts
    for _ in range(40):
        cr = rng.normal(size=N_DIM)
        cr /= np.linalg.norm(cr)
        starts.append(cr)

    best_c = starts[0].copy()
    best_F = objective(best_c)

    for c0 in starts:
        if time_left() < 8.0:
            break
        c_opt, F_opt = sphere_flow(c0, max_iter=1500, lr0=0.05, tol=1e-13, time_budget=12.0)
        if F_opt < best_F:
            best_F = F_opt
            best_c = c_opt

    # ---------- Global DE pre-search (parameterize by 21-dim vector, normalize) ----------
    if time_left() > 60.0:
        def de_objective(vec):
            n = np.linalg.norm(vec)
            if n < 1e-12:
                return 1e10
            c = vec / n
            return objective(c)

        bounds = [(-1.0, 1.0)] * N_DIM
        # Use a moderate popsize and maxiter, but bounded by time via callback
        de_start = time.time()
        def de_callback(xk, convergence):
            # xk is the current best; check time
            if (time.time() - de_start) > max(20.0, time_left() - 30.0):
                return True
            return False

        try:
            de_res = differential_evolution(
                de_objective,
                bounds,
                maxiter=80,
                popsize=18,
                seed=99,
                tol=1e-10,
                mutation=(0.4, 1.0),
                recombination=0.7,
                init='sobol',
                callback=de_callback,
                polish=True,
                update_interval=5,
            )
            if de_res.success or de_res.x is not None:
                c_de = de_res.x
                n = np.linalg.norm(c_de)
                if n > 1e-12:
                    c_de = c_de / n
                    c_pol, F_pol = sphere_flow(c_de, max_iter=1500, lr0=0.05, tol=1e-13, time_budget=20.0)
                    if F_pol < best_F:
                        best_F = F_pol
                        best_c = c_pol
        except Exception:
            pass

    # ---------- Final polish with L-BFGS-B under sphere equality constraint ----------
    if time_left() > 15.0:
        def obj_val(c):
            return objective(c)
        def grad_val(c):
            return objective_grad(c)[1]
        def cons_fun(c):
            return np.dot(c, c) - 1.0
        def cons_jac(c):
            return 2.0 * c
        constraints = [{'type': 'eq', 'fun': cons_fun, 'jac': cons_jac}]
        try:
            res = minimize(
                obj_val, best_c, jac=grad_val, method='SLSQP',
                constraints=constraints, options={'maxiter': 400, 'ftol': 1e-14}
            )
            c_opt = res.x
            n = np.linalg.norm(c_opt)
            if n > 1e-12:
                c_opt = c_opt / n
                F_opt = objective(c_opt)
                if F_opt < best_F:
                    best_F = F_opt
                    best_c = c_opt
        except Exception:
            pass

    # ---------- Validity checks ----------
    best_c = np.asarray(best_c, dtype=np.float64)
    if best_c.shape != (N_DIM,):
        best_c = np.ones(N_DIM, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(N_DIM, dtype=np.float64)
    n = np.linalg.norm(best_c)
    if n < 1e-12:
        best_c = np.ones(N_DIM, dtype=np.float64)
    else:
        best_c = best_c / n
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1.0 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e5 / max_abs)

    return best_c
