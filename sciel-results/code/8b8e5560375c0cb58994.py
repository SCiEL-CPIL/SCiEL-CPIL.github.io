# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize as scipy_minimize
import time

# ---------- Basis precomputation ----------
def _precompute_basis():
    k = 10
    n_theta = 70
    n_phi = 130
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    n_pts = n_theta * n_phi
    dim = 2 * k + 1

    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)

    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))

    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    sqrt2 = np.sqrt(2.0)
    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    for m in range(1, 11):
        idx_pos = m + 10
        cos_term = np.cos(m * phi)
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, idx_pos] = (val_theta[:, None] * cos_term[None, :]).ravel()
        idx_neg = -m + 10
        sin_term = np.sin(m * phi)
        B[:, idx_neg] = (val_theta[:, None] * sin_term[None, :]).ravel()
    return B, weights

B_GLOBAL, W_GLOBAL = _precompute_basis()

# ---------- Objective / gradient / Hessian ----------
def _compute_ratio(c):
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    if norm2_sq < 1e-30:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    return norm4 / norm2

def _objective_and_grad(c):
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    if norm4 < 1e-30:
        return np.inf, np.zeros_like(c)
    R = norm4 / norm2
    grad_4 = 4.0 * (B_GLOBAL.T @ (W_GLOBAL * f * f2))
    grad_2 = 2.0 * (B_GLOBAL.T @ (W_GLOBAL * f))
    grad_R = (1.0 / (4.0 * norm4**3 * norm2)) * grad_4 - (norm4 / (2.0 * norm2**3)) * grad_2
    return R, grad_R

def _objective_grad_hess(c):
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c), np.zeros((21, 21))
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    if norm4 < 1e-30:
        return np.inf, np.zeros_like(c), np.zeros((21, 21))
    R = norm4 / norm2
    G4 = B_GLOBAL.T @ (W_GLOBAL * f * f2)
    G2 = B_GLOBAL.T @ (W_GLOBAL * f)
    grad = (1.0 / (4.0 * norm4**3 * norm2)) * G4 - (norm4 / (2.0 * norm2**3)) * G2
    # Hessian of R = a*norm4 - b*norm2  with a=1/(4 norm4^3 norm2), b=norm4/(2 norm2^3)
    a = 1.0 / (4.0 * norm4**3 * norm2)
    b = norm4 / (2.0 * norm2**3)
    H4 = 4.0 * (B_GLOBAL.T @ (W_GLOBAL * (3.0 * f2 * f[:, :, None] * B_GLOBAL[:, :, None])))
    H2 = 2.0 * (B_GLOBAL.T @ (W_GLOBAL * B_GLOBAL))
    H = a * H4 - b * H2
    return R, grad, H

def _normalize_unit_l2(c):
    f = B_GLOBAL @ c
    norm2_sq = np.sum(W_GLOBAL * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

# ---------- Constrained (unit-sphere) Newton polish ----------
def _constrained_newton_polish(c, deadline, max_iter=400):
    """Project the gradient/Hessian onto the unit-sphere tangent and take Newton
    steps. This is a smooth, constraint-preserving local solver that drives the
    gradient of R (restricted to the sphere) to ~0, i.e. a true stationary point."""
    c = _normalize_unit_l2(c.copy())
    current_val = _compute_ratio(c)
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        val, grad, H = _objective_grad_hess(c)
        if not np.isfinite(val):
            break
        # Project gradient and Hessian onto tangent space T = {v : <v,c> = 0}
        gc = float(np.dot(c, grad))
        g_tan = grad - gc * c
        gnorm = np.linalg.norm(g_tan)
        if gnorm < 1e-14:
            break
        H_tan = H - np.outer(c, H @ c) - np.outer(H @ c, c) + gc * np.outer(c, c)
        # Solve Newton step with damping / regularization fallback
        step = None
        for lam in [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            try:
                M = H_tan + lam * np.eye(21)
                s = np.linalg.solve(M, -g_tan)
                if np.all(np.isfinite(s)) and np.linalg.norm(s) < 1e6:
                    step = s
                    break
            except Exception:
                continue
        if step is None:
            step = -g_tan / gnorm
        # Backtracking line search on the sphere
        t = 1.0
        improved = False
        for _bt in range(60):
            c_try = c + t * step
            c_try = _normalize_unit_l2(c_try)
            val_try = _compute_ratio(c_try)
            if np.isfinite(val_try) and val_try < current_val - 1e-22:
                c = c_try
                current_val = val_try
                improved = True
                break
            t *= 0.5
        if not improved:
            break
    return current_val, c

# ---------- Gradient descent polish ----------
def _gradient_descent_polish(c, deadline, max_iter=2000):
    c = c.copy()
    current_val = _compute_ratio(c)
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
        gc = float(np.dot(c, grad))
        g_tan = grad - gc * c
        gnorm = np.linalg.norm(g_tan)
        if gnorm < 1e-15:
            break
        step = 0.05
        improved = False
        for _bt in range(50):
            c_try = _normalize_unit_l2(c - step * g_tan)
            val_try = _compute_ratio(c_try)
            if val_try < current_val - 1e-22:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return current_val, c

# ---------- L-BFGS-B ----------
def _lgbfsb_optimize(c0, deadline, norm_func=None, maxiter=3000, ftol=1e-20, gtol=1e-18, maxfun=8000):
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        c_norm = norm_func(c)
        val, grad = _objective_and_grad(c_norm)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    try:
        result = scipy_minimize(
            obj_and_grad, c0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxfun})
        if result.success or result.fun < 1e10:
            c_opt = norm_func(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    return np.inf, c0

# ---------- Nelder-Mead ----------
def _nelder_mead_minimize(func, x0, max_iter=2000, tol=1e-15, deadline=None, norm_func=None):
    n = len(x0)
    if norm_func is not None:
        x0 = norm_func(x0).copy()
    simplex = [x0.copy()]
    base_step = 0.05
    for i in range(n):
        xi = x0.copy()
        xi[i] += base_step
        if norm_func is not None:
            xi = norm_func(xi)
        simplex.append(xi)
    fvals = [func(v) for v in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    best_val = min(fvals)
    best_point = simplex[np.argmin(fvals)].copy()
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol * (1 + abs(fvals[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        if norm_func is not None:
            xr = norm_func(xr)
        fr = func(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            if norm_func is not None:
                xe = norm_func(xe)
            fe = func(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            if norm_func is not None:
                xc = norm_func(xc)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    if norm_func is not None:
                        simplex[i] = norm_func(simplex[i])
                    fvals[i] = func(simplex[i])
        cur = min(fvals)
        if cur < best_val:
            best_val = cur
            best_point = simplex[np.argmin(fvals)].copy()
    return best_val, best_point

# ---------- Diverse symmetry-aware initializers ----------
def _generate_initial_points(dim, n_random=50):
    inits = []
    c_zonal = np.zeros(dim); c_zonal[10] = 1.0; inits.append(c_zonal)
    inits.append(np.ones(dim))
    for i in range(dim):
        c = np.zeros(dim); c[i] = 1.0; inits.append(c)
    c_even = np.zeros(dim)
    for i in range(dim):
        if (i - 10) % 2 == 0:
            c_even[i] = 1.0
    inits.append(c_even)
    c_odd = np.zeros(dim)
    for i in range(dim):
        if (i - 10) % 2 != 0:
            c_odd[i] = 1.0
    inits.append(c_odd)
    for bias in [0.5, 0.3, 0.1]:
        cb = np.zeros(dim)
        for i in range(dim):
            cb[i] = bias ** abs(i - 10)
        inits.append(cb)
    for sgm in [1.0, 2.0, 3.0]:
        cg = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            cg[i] = np.exp(-m**2 / (2 * sgm**2))
        inits.append(cg)
    for freq in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]:
        cc = np.zeros(dim)
        for i in range(dim):
            cc[i] = np.cos(freq * (i - 10))
        inits.append(cc)
        cs = np.zeros(dim)
        for i in range(dim):
            cs[i] = np.sin(freq * (i - 10))
        inits.append(cs)
    for seed in range(n_random):
        np.random.seed(seed * 12345 + 67890)
        inits.append(np.random.randn(dim))
    for seed in range(20):
        np.random.seed(seed * 98765 + 43210)
        cr = np.random.randn(dim)
        for i in range(dim):
            cr[i] *= 0.5 ** abs(i - 10)
        inits.append(cr)
    for seed in range(10):
        np.random.seed(seed * 54321 + 13579)
        cr = np.random.randn(dim)
        for i in range(dim):
            cr[i] *= 0.7 ** abs(i - 10)
        inits.append(cr)
    for seed in range(20):
        np.random.seed(seed * 24680 + 13579)
        cs = np.zeros(dim)
        nn = np.random.randint(1, 6)
        idx = np.random.choice(dim, nn, replace=False)
        cs[idx] = np.random.randn(nn)
        inits.append(cs)
    for bs in [2, 3, 4, 5]:
        for st in range(0, dim - bs + 1, max(1, bs - 1)):
            cb = np.zeros(dim); cb[st:st + bs] = 1.0; inits.append(cb)
    for ad in [0.5, 0.7, 0.9]:
        ca = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            ca[i] = (1.0 if m % 2 == 0 else -1.0) * (ad ** abs(m))
        inits.append(ca)
    for p1 in range(-10, 11, 3):
        for p2 in range(-10, 11, 3):
            if p1 != p2:
                ct = np.zeros(dim)
                for i in range(dim):
                    m = i - 10
                    ct[i] = np.exp(-(m - p1)**2) + np.exp(-(m - p2)**2)
                inits.append(ct)
    for coeffs in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]:
        cp = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            cp[i] = coeffs[0] + coeffs[1]*m + coeffs[2]*m**2
        inits.append(cp)
    for freq in np.arange(0.1, 3.0, 0.1):
        cc = np.zeros(dim)
        for i in range(dim):
            cc[i] = np.cos(freq * (i - 10) + 0.5)
        inits.append(cc)
    for bw in [3, 5, 7, 9, 11, 15, 21]:
        cb = np.zeros(dim)
        hw = bw // 2
        cb[max(0, 10 - hw):min(dim, 10 + hw + 1)] = 1.0
        inits.append(cb)
    for wt in np.arange(0.1, 1.5, 0.1):
        cm = np.zeros(dim)
        cm[10] = 1.0
        for m in [1, 2, 3, 4, 5]:
            cm[10 + m] = wt
            cm[10 - m] = wt
        inits.append(cm)
    for ph in np.arange(0, 2 * np.pi, 0.5):
        cp = np.zeros(dim)
        for i in range(dim):
            cp[i] = np.cos((i - 10) * 0.5 + ph)
        inits.append(cp)
    # extra: pure cos/sin single-m pairs and mixed zonal+tesseral tuned
    for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for w in [0.2, 0.5, 1.0, 2.0]:
            cc = np.zeros(dim); cc[10] = 1.0
            cc[10 + m] = w; cc[10 - m] = w
            inits.append(cc)
    return inits

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 615.0   # safety margin before 630s kill
    dim = 21

    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(c)

    best_coeffs = None
    best_val = np.inf

    inits = _generate_initial_points(dim, n_random=50)
    init_vals = []
    for c_init in inits:
        c_norm = _normalize_unit_l2(c_init)
        val = objective(c_norm)
        init_vals.append((val, c_norm))
        if val < best_val:
            best_val = val
            best_coeffs = c_norm.copy()
    init_vals.sort(key=lambda x: x[0])

    # Phase 1: L-BFGS-B from best starts
    n_starts = 30
    for val, c_start in init_vals[:n_starts]:
        if time.time() > deadline - 150:
            break
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_start, deadline, norm_func=_normalize_unit_l2)
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead from best starts
    for val, c_start in init_vals[:15]:
        if time.time() > deadline - 150:
            break
        try:
            val_opt, c_opt = _nelder_mead_minimize(
                objective, c_start, max_iter=2000, tol=1e-15,
                deadline=deadline, norm_func=_normalize_unit_l2)
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 3: random restarts with L-BFGS-B
    for i in range(60):
        if time.time() > deadline - 90:
            break
        if i % 5 == 0:
            np.random.seed(i * 777 + 12345)
            c_rand = np.random.randn(dim)
        elif i % 5 == 1:
            c_rand = best_coeffs + 0.05 * np.random.randn(dim)
        elif i % 5 == 2:
            c_rand = 0.8 * best_coeffs + 0.2 * np.random.randn(dim)
        elif i % 5 == 3:
            c_rand = np.random.randn(dim)
            for im in range(dim):
                c_rand[im] *= 0.5 ** abs(im - 10)
        else:
            c_rand = np.zeros(dim)
            nn = np.random.randint(1, 8)
            idx = np.random.choice(dim, nn, replace=False)
            c_rand[idx] = np.random.randn(nn)
        c_rand = _normalize_unit_l2(c_rand)
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_rand, deadline, norm_func=_normalize_unit_l2)
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 4: high-precision polishing
    if time.time() < deadline - 40:
        val_polish, c_polish = _gradient_descent_polish(best_coeffs.copy(), deadline, max_iter=2000)
        if val_polish < best_val:
            best_val = val_polish
            best_coeffs = c_polish.copy()
    # Constrained Newton polish (drives the on-sphere gradient to ~0)
    if time.time() < deadline - 20:
        val_newt, c_newt = _constrained_newton_polish(best_coeffs.copy(), deadline, max_iter=400)
        if val_newt < best_val:
            best_val = val_newt
            best_coeffs = c_newt.copy()
    # Second Newton pass for extra precision
    if time.time() < deadline - 10:
        val_newt2, c_newt2 = _constrained_newton_polish(best_coeffs.copy(), deadline, max_iter=300)
        if val_newt2 < best_val:
            best_val = val_newt2
            best_coeffs = c_newt2.copy()
    # Final ultra-tight L-BFGS-B
    if time.time() < deadline - 5:
        def obj_ultra(c):
            if not np.all(np.isfinite(c)):
                return 1e10, np.zeros_like(c)
            cn = _normalize_unit_l2(c)
            val, grad = _objective_and_grad(cn)
            if not np.isfinite(val):
                return 1e10, np.zeros_like(c)
            return val, grad
        try:
            res = scipy_minimize(obj_ultra, best_coeffs.copy(), jac=True, method='L-BFGS-B',
                                 options={'maxiter': 8000, 'ftol': 1e-25, 'gtol': 1e-20, 'maxfun': 15000})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = _normalize_unit_l2(res.x).copy()
        except Exception:
            pass

    coeffs = best_coeffs.astype(np.float64)
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
    max_c = np.max(np.abs(coeffs))
    if max_c < 1e-3:
        coeffs = coeffs / max_c * 1.0 if max_c > 0 else np.ones(dim)
    elif max_c > 1e5:
        coeffs = coeffs / max_c * 1.0
    return coeffs

# EVOLVE-BLOCK-END
