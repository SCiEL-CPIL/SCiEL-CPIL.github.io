# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize as scipy_minimize
import time

def _precompute_basis():
    k = 10
    n_theta = 70
    n_phi = 130
    dim = 21

    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    n_pts = n_theta * n_phi

    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)

    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) *
                       float(factorial(k - m) / factorial(k + m)))

    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    sqrt2 = np.sqrt(2.0)

    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    for m in range(1, 11):
        val = sqrt2 * N[m] * P[m, :]
        cos_t = np.cos(m * phi)
        sin_t = np.sin(m * phi)
        B[:, 10 + m] = (val[:, None] * cos_t[None, :]).ravel()
        B[:, 10 - m] = (val[:, None] * sin_t[None, :]).ravel()

    return B, weights

B_G, W_G = _precompute_basis()


def _compute_ratio(c):
    f = B_G @ c
    f2 = f * f
    norm2_sq = np.sum(W_G * f2)
    if norm2_sq < 1e-30:
        return np.inf
    norm4_4 = np.sum(W_G * f2 * f2)
    if norm4_4 < 1e-30:
        return np.inf
    return (norm4_4 ** 0.25) / np.sqrt(norm2_sq)


def _objective_and_grad(c):
    f = B_G @ c
    f2 = f * f
    norm2_sq = np.sum(W_G * f2)
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
    norm4_4 = np.sum(W_G * f2 * f2)
    norm4 = norm4_4 ** 0.25
    if norm4 < 1e-30:
        return np.inf, np.zeros_like(c)
    norm2 = np.sqrt(norm2_sq)
    R = norm4 / norm2
    grad_4 = 4.0 * (B_G.T @ (W_G * f * f2))
    grad_2 = 2.0 * (B_G.T @ (W_G * f))
    grad_R = grad_4 / (4.0 * norm4**3 * norm2) - (norm4 * grad_2) / (2.0 * norm2**3)
    return R, grad_R


def _normalize_unit(c):
    f = B_G @ c
    n2 = np.sum(W_G * f * f)
    if n2 < 1e-30:
        return c
    return c / np.sqrt(n2)


def _project_onto_sphere(c, r=1.0):
    n = np.linalg.norm(c)
    if n < 1e-15:
        c = np.random.randn(len(c))
        return c / np.linalg.norm(c) * r
    return c / n * r


def _init_pool(dim):
    inits = []
    # zonal
    c = np.zeros(dim); c[10] = 1.0; inits.append(c)
    # all ones
    inits.append(np.ones(dim))
    # single m
    for i in range(dim):
        c = np.zeros(dim); c[i] = 1.0; inits.append(c)
    # low-m bias
    for b in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        c = np.zeros(dim)
        for i in range(dim):
            c[i] = b ** abs(i - 10)
        inits.append(c)
    # cosine/sine patterns in m
    for fr in np.arange(0.1, 4.0, 0.15):
        c = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c[i] = np.cos(fr * m)
        inits.append(c)
        c2 = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c2[i] = np.sin(fr * m)
        inits.append(c2)
    # random normals
    for s in range(50):
        np.random.seed(s * 12345 + 67890)
        inits.append(np.random.randn(dim))
    # low-m biased random
    for s in range(40):
        np.random.seed(s * 98765 + 43210)
        c = np.random.randn(dim)
        for i in range(dim):
            c[i] *= 0.4 ** abs(i - 10)
        inits.append(c)
    # sparse random
    for s in range(40):
        np.random.seed(s * 24680 + 13579)
        c = np.zeros(dim)
        nnz = np.random.randint(1, 10)
        idx = np.random.choice(dim, nnz, replace=False)
        c[idx] = np.random.randn(nnz)
        inits.append(c)
    # two-peak
    for p1 in range(-10, 11, 2):
        for p2 in range(p1 + 2, 11, 2):
            c = np.zeros(dim)
            for i in range(dim):
                m = i - 10
                c[i] = np.exp(-(m - p1)**2 / 2.0) + np.exp(-(m - p2)**2 / 2.0)
            inits.append(c)
    # three-peak
    for p1 in range(-9, 10, 3):
        for p2 in range(p1 + 2, 10, 3):
            for p3 in range(p2 + 2, 11, 3):
                c = np.zeros(dim)
                for i in range(dim):
                    m = i - 10
                    c[i] = np.exp(-(m - p1)**2 / 2.5) + np.exp(-(m - p2)**2 / 2.5) + np.exp(-(m - p3)**2 / 2.5)
                inits.append(c)
    # alternating sign patterns
    for period in [2, 3, 4, 5, 6, 7]:
        c = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c[i] = np.sign(np.cos(2 * np.pi * m / period))
        inits.append(c)
    return inits


def _lgbfsb(c0, deadline, maxfun=3000, ftol=1e-18, gtol=1e-16):
    def obj_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        cn = _normalize_unit(c)
        val, grad = _objective_and_grad(cn)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    try:
        res = scipy_minimize(obj_grad, c0, jac=True, method='L-BFGS-B',
                             options={'maxiter': 3000, 'maxfun': maxfun,
                                      'ftol': ftol, 'gtol': gtol})
        if res.fun < 1e9:
            return res.fun, _normalize_unit(res.x)
    except Exception:
        pass
    return np.inf, c0


def _nelder_mead(func, x0, deadline, maxiter=3000, tol=1e-15):
    n = len(x0)
    x0 = _project_onto_sphere(x0)
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy(); xi[i] += 0.03
        simplex.append(_project_onto_sphere(xi))
    fvals = [func(v) for v in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    best_val = min(fvals)
    best_pt = simplex[np.argmin(fvals)].copy()
    for it in range(maxiter):
        if time.time() > deadline - 3:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol * (1 + abs(fvals[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = _project_onto_sphere(centroid + alpha * (centroid - simplex[-1]))
        fr = func(xr)
        if fr < fvals[0]:
            xe = _project_onto_sphere(centroid + gamma * (centroid - simplex[-1]))
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = _project_onto_sphere(centroid + rho * (simplex[-1] - centroid))
            else:
                xc = _project_onto_sphere(centroid - rho * (simplex[-1] - centroid))
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = _project_onto_sphere(simplex[0] + sigma * (simplex[i] - simplex[0]))
                    fvals[i] = func(simplex[i])
        cur = min(fvals)
        if cur < best_val:
            best_val = cur
            best_pt = simplex[np.argmin(fvals)].copy()
    return best_val, best_pt


def _grad_descent_polish(c, deadline, max_iter=5000):
    c = _project_onto_sphere(c.copy())
    cur = _compute_ratio(c)
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
        # project gradient to tangent space
        grad_t = grad - (grad @ c) * c
        gnorm = np.linalg.norm(grad_t)
        if gnorm < 1e-16:
            break
        grad_t = grad_t / gnorm
        step = 0.1
        improved = False
        for _bt in range(60):
            c_try = _project_onto_sphere(c - step * grad_t)
            vt = _compute_ratio(c_try)
            if vt < cur - 1e-17:
                c = c_try; cur = vt; improved = True
                break
            step *= 0.5
        if not improved:
            break
    return cur, c


def _local_restarts(c_best, deadline, n_restarts=20):
    best_val = _compute_ratio(c_best)
    best_c = c_best.copy()
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        np.random.seed(42 + i * 7919)
        noise = np.random.randn(len(c_best)) * 0.1
        c_try = _project_onto_sphere(c_best + noise)
        v, c_try = _grad_descent_polish(c_try, deadline, max_iter=2000)
        if v < best_val:
            best_val = v
            best_c = c_try.copy()
    return best_val, best_c


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 600.0
    dim = 21

    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(_normalize_unit(c))

    best_c = None
    best_val = np.inf

    inits = _init_pool(dim)
    init_sorted = []
    for c_init in inits:
        cn = _project_onto_sphere(c_init)
        v = _compute_ratio(cn)
        init_sorted.append((v, cn))
        if v < best_val:
            best_val = v
            best_c = cn.copy()
    init_sorted.sort(key=lambda x: x[0])

    # Phase 1: L-BFGS-B from best inits
    for v, cs in init_sorted[:50]:
        if time.time() > deadline - 180:
            break
        try:
            vo, co = _lgbfsb(cs, deadline, maxfun=3000)
            if vo < best_val:
                best_val = vo
                best_c = co.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead from a few starts
    for v, cs in init_sorted[:15]:
        if time.time() > deadline - 120:
            break
        try:
            vo, co = _nelder_mead(objective, cs, deadline, maxiter=2000)
            if vo < best_val:
                best_val = vo
                best_c = co.copy()
        except Exception:
            pass

    # Phase 3: gradient polish
    if time.time() < deadline - 60:
        vp, cp = _grad_descent_polish(best_c, deadline, max_iter=5000)
        if vp < best_val:
            best_val = vp
            best_c = cp.copy()
        # L-BFGS-B re-polish
        if time.time() < deadline - 30:
            try:
                vo, co = _lgbfsb(best_c, deadline, maxfun=4000, ftol=1e-20, gtol=1e-18)
                if vo < best_val:
                    best_val = vo
                    best_c = co.copy()
            except Exception:
                pass
        # final grad polish
        if time.time() < deadline - 10:
            vp, cp = _grad_descent_polish(best_c, deadline, max_iter=3000)
            if vp < best_val:
                best_val = vp
                best_c = cp.copy()

    # Phase 4: local restarts around best
    if time.time() < deadline - 15:
        try:
            vr, cr = _local_restarts(best_c, deadline, n_restarts=15)
            if vr < best_val:
                best_val = vr
                best_c = cr.copy()
        except Exception:
            pass

    coeffs = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
    mc = np.max(np.abs(coeffs))
    if mc < 1e-3:
        coeffs = coeffs / mc * 1.0 if mc > 0 else np.ones(dim)
    elif mc > 1e5:
        coeffs = coeffs / mc * 1.0
    return coeffs

# EVOLVE-BLOCK-END
