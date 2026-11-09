"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START

def _score_vec(h):
    """Exact normalized max autoconvolution for a nonnegative height vector.
    
    The autoconvolution of a step function h on N equal intervals has peak
    at lag 0 equal to sum(h^2) * (interval width). The normalization divides
    by (sum(h))^2 / N to make the score scale-invariant.
    """
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s <= 0:
        return 0.0
    # The peak of autoconvolution is sum(h_i^2) * dx where dx = 1/(2N)
    # Normalized by (sum(h) * dx)^2 / dx = sum(h)^2 * dx
    # So normalized peak = sum(h^2) * dx / (sum(h)^2 * dx) = sum(h^2) / sum(h)^2
    # But the task says "normalized maximum" - let's use sum(h^2)/(sum(h)^2/N)
    # which equals N * sum(h^2) / sum(h)^2
    return float((h * h).sum()) / (s * s / N)


def _setup():
    """Build the matrix Q so that h^T Q h = sum(h^2) for the autoconvolution peak."""
    # For a step function on N intervals, the autoconvolution at lag 0 is sum(h_i^2)
    # So Q = I (identity) for the quadratic form h^T Q h = sum(h_i^2)
    # But we need to be careful about the exact normalization.
    # Let's just use the identity matrix since the peak is simply sum(h^2)
    return np.eye(N)


_R = None


def _project_simplex(v, target):
    """Closest point to v (Euclidean) on {x>=0, sum x = target}."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    # Find rho such that u[rho] > (sv[rho] - target) / (rho+1)
    # and u[rho+1] <= (sv[rho] - target) / (rho+1)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (sv - target))[0]
    if len(rho_candidates) == 0:
        return np.zeros(n)
    rho = rho_candidates[-1]
    theta = (sv[rho] - target) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _face_minimize(idx, h0):
    """Minimize sum h^2 over {h_j, j in idx} s.t. sum h = 1, h_j >= 0.
    
    Since the objective is sum(h^2) and the constraint is sum(h) = 1 with h >= 0,
    the optimal solution on a given support is h_j = 1/|idx| for all j in idx.
    But we want to find the best support, so we use optimization to allow
    non-uniform weights that might work better with the global structure.
    """
    idx = np.array(sorted(set(int(i) for i in idx)), dtype=int)
    m = len(idx)
    if m == 0:
        return np.zeros(N)
    
    # For sum(h^2) with sum(h)=1, h>=0, the minimum on a fixed support is uniform
    # But let's still run optimization in case we want to explore non-uniform solutions
    # that might lead to better global solutions when combined with support selection
    
    # Start with uniform on the support
    x0 = np.ones(m) / m
    
    def f(x):
        return float(x @ x)
    
    def g(x):
        return 2.0 * x
    
    best_x, best_f = x0.copy(), f(x0)
    
    # Projected gradient descent with line search
    try:
        for _ in range(200):
            grad = g(best_x)
            # Newton direction for this simple quadratic
            step_dir = -grad / (2.0)  # Since Hessian is 2*I
            best_t, best_gt = 0.0, best_f
            ts = np.concatenate(([0.1, 0.5, 1.0, 2.0], np.logspace(-2, 1, 8)))
            for t in ts:
                cand = _project_simplex(best_x + t * step_dir, 1.0)
                gv = f(cand)
                if gv < best_gt - 1e-14:
                    best_gt, best_t = gv, t
            if best_t == 0.0:
                break
            nx = _project_simplex(best_x + best_t * step_dir, 1.0)
            nf = f(nx)
            if nf >= best_f - 1e-15:
                break
            if nf < best_f - 1e-14:
                best_f = nf
                best_x = nx.copy()
    except Exception:
        pass
    
    # Local polish with L-BFGS-B
    try:
        res = minimize(f, best_x, jac=g, method="L-BFGS-B",
                       options=dict(maxiter=300, gtol=1e-13, ftol=1e-16))
        xf = res.x
        xf = _project_simplex(xf, 1.0)
        if f(xf) < best_f - 1e-14:
            best_f = f(xf)
            best_x = xf.copy()
    except Exception:
        pass
    
    out = np.zeros(N)
    out[idx] = np.maximum(best_x, 0.0)
    s = out.sum()
    if s > 0:
        out /= s
    return out


def _kkt_support(R, h0):
    """Greedy active-set/KKT pass to discover the support of the interior solution."""
    n = N
    h = np.maximum(h0, 0.0)
    lam = None
    for _ in range(1200):
        Qh = R @ h
        if lam is None:
            lam = float(np.mean(Qh))
        else:
            lam = lam * 0.5 + 0.5 * float(np.mean(Qh))
        resid = Qh - lam
        sup = set(np.nonzero(h > 1e-10 * max(h.max(), 1e-300))[0])
        add = np.nonzero(resid > 1e-9)[0]
        add = [int(a) for a in add if a not in sup]
        if not add:
            break
        sup = sup | set(add)
        idx = np.array(sorted(sup), dtype=int)
        Qss = R[np.ix_(idx, idx)]
        try:
            hs = np.linalg.solve(Qss + 1e-12 * np.eye(len(idx)), np.ones(len(idx)))
        except np.linalg.LinAlgError:
            hs = np.linalg.lstsq(Qss + 1e-9 * np.eye(len(idx)), np.ones(len(idx)), rcond=None)[0]
        h = np.zeros(n)
        h[idx] = hs
        if np.any(hs < -1e-7):
            w = np.where(hs < -1e-7)[0]
            for j in w:
                idx = np.delete(idx, j)
            break
    sup = set(np.nonzero(h > 1e-10 * max(h.max(), 1e-300))[0])
    return np.array(sorted(sup), dtype=int)


def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin under the 150s kill
    
    global _R
    try:
        _R = _setup()
    except Exception:
        return [1.0] * N
    
    # Initial candidates - diverse starting points
    base = np.ones(N)
    cands = [base]
    
    # Concentrated in the middle
    for width_frac in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        w = int(N * width_frac)
        v = np.zeros(N)
        v[(N - w) // 2:(N + w) // 2] = 1.0
        cands.append(v)
    
    # Two bumps
    for sep in [0.2, 0.3, 0.4]:
        v = np.zeros(N)
        bump_w = int(N * 0.3)
        center1 = int(N * (0.5 - sep))
        center2 = int(N * (0.5 + sep))
        v[max(0, center1 - bump_w // 2):min(N, center1 + bump_w // 2)] = 1.0
        v[max(0, center2 - bump_w // 2):min(N, center2 + bump_w // 2)] = 1.0
        cands.append(v)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        sig = np.exp(-0.5 * (x / sigma) ** 2)
        cands.append(sig)
    
    # Cosine bump
    for width_frac in [0.5, 0.7, 0.9]:
        w = int(N * width_frac)
        v = np.zeros(N)
        idx = np.arange(w)
        v[(N - w) // 2:(N + w) // 2] = 0.5 * (1 + np.cos(2 * np.pi * idx / w))
        cands.append(v)
    
    # Deterministic perturbations
    rng = np.random.default_rng(12345)
    for _ in range(6):
        cands.append(np.clip(base + 0.5 * rng.standard_normal(N), 0.0, None))
    
    best_h, best_val = None, float("inf")
    for v in cands:
        vv = np.maximum(v, 0.0)
        if vv.sum() <= 0:
            continue
        sc = _score_vec(vv)
        if sc < best_val:
            best_val, best_h = sc, vv.copy()
    
    if best_h is None:
        best_h, best_val = base.copy(), _score_vec(base)
    
    # KKT support discovery + face optimization, per candidate
    for v in cands:
        if time.time() > deadline:
            break
        try:
            sup = _kkt_support(_R, v)
            if len(sup) < 2:
                continue
            hh = _face_minimize(sup, v)
            sc = _score_vec(hh)
            if sc < best_val - 1e-15:
                best_val, best_h = sc, hh
        except Exception:
            continue
    
    # Additional local refinement: try perturbing the best solution
    if best_h is not None and time.time() < deadline:
        h_curr = best_h.copy()
        s = h_curr.sum()
        if s > 0:
            h_curr /= s
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random coordinate perturbation
            rng2 = np.random.default_rng(int(time.time() * 1000) % 10000)
            idx_perturb = rng2.choice(N, size=N // 10, replace=False)
            h_new = h_curr.copy()
            for i in idx_perturb:
                h_new[i] = max(0, h_new[i] + rng2.standard_normal() * 0.1)
            sn = h_new.sum()
            if sn > 0:
                h_new /= sn
            sc = _score_vec(h_new)
            if sc < best_val - 1e-15:
                best_val, best_h = sc, h_new
    
    if best_h is None:
        best_h, best_val = base.copy(), _score_vec(base)
    
    out = np.asarray(best_h, dtype=float)
    out = np.maximum(out, 0.0)
    s = out.sum()
    if s <= 0:
        return [1.0] * N
    out = out / s
    return [float(x) for x in out]

# EVOLVE-BLOCK-END
