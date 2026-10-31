import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle index combinations
tri_idx = np.array(list(combinations(range(N), 3)), dtype=np.int64)  # (165, 3)

# Equilateral triangle vertices (unit area)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def cart_to_bary(p):
    """Convert Cartesian (x,y) to barycentric (u,v) where u=coord at V1, v=coord at V2."""
    x = p[:, 0]
    y = p[:, 1]
    v = y / H
    u = x - 0.5 * v
    return np.column_stack([u, v])

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_cart(p):
    """Areas of all 165 triangles from N cartesian points (unit-area triangle)."""
    pi = p[tri_idx[:, 0]]
    pj = p[tri_idx[:, 1]]
    pk = p[tri_idx[:, 2]]
    a = pj - pi
    b = pk - pi
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

def in_triangle(p, eps=1e-10):
    """Check if cartesian points are inside the unit equilateral triangle."""
    x = p[:, 0]
    y = p[:, 1]
    b0 = 1.0 - x          # coord at V0
    b1 = x - 0.5 * y / H  # coord at V1
    b2 = y / H            # coord at V2
    return (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)

def project_to_triangle(p, eps=1e-9):
    """Project cartesian points into the triangle interior (barycentric clipping)."""
    bary = cart_to_bary(p)
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    s = u + v
    bad = s > (1.0 - eps)
    if np.any(bad):
        scale = (1.0 - eps) / np.where(s > 0, s, 1.0)
        u[bad] *= scale[bad]
        v[bad] *= scale[bad]
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return bary_to_cart(np.column_stack([u, v]))

def min_area(p):
    return float(np.min(calc_areas_cart(p)))

def smooth_min_obj(p_flat, beta=40.0):
    """Smooth surrogate: -soft_min(areas). Vectorized, differentiable."""
    p = p_flat.reshape(N, 2)
    p = project_to_triangle(p)
    areas = calc_areas_cart(p)
    if np.any(areas < 1e-14):
        return 1e10
    scaled = -beta * areas
    m = np.max(scaled)
    if m > 700:
        return -np.min(areas)
    lse = m + np.log(np.sum(np.exp(scaled - m)))
    return lse / beta  # positive ~ -soft_min

def barrier_obj(p_flat, beta=60.0, mu=0.5):
    """Smooth min + log-barrier for triangle boundary (cartesian)."""
    p = p_flat.reshape(N, 2)
    x = p[:, 0]
    y = p[:, 1]
    # barycentric coords
    b0 = 1.0 - x
    b1 = x - 0.5 * y / H
    b2 = y / H
    # log barrier (penalize approaching edges)
    barrier = -mu * (np.log(np.clip(b0, 1e-12, None)) +
                     np.log(np.clip(b1, 1e-12, None)) +
                     np.log(np.clip(b2, 1e-12, None)))
    base = smooth_min_obj(p_flat, beta)
    if base > 1e9:
        return base + 1e6
    return base + barrier

def critical_triangles(p, frac=1.02):
    """Return indices of triangles with area near the minimum."""
    areas = calc_areas_cart(p)
    mn = np.min(areas)
    return np.where(areas <= mn * frac)[0]

def equalize_step(p, max_iter=60, tol=1e-12):
    """
    Active-set equalization: for the critical triangles, find a move that
    equalizes their areas and makes the min stationary. We solve a small
    least-squares on the area gradients w.r.t. the 2N coordinates.
    Returns improved p.
    """
    p = p.copy()
    for it in range(max_iter):
        areas = calc_areas_cart(p)
        mn = np.min(areas)
        crit = np.where(areas <= mn * 1.02)[0]
        if len(crit) < 2:
            break
        # For each critical triangle, area = 0.5*|cross|. Use signed to get gradient.
        # Build Jacobian of signed areas w.r.t. coordinates.
        m = len(crit)
        # Signed area for a triangle (i,j,k): 0.5*((xj-xi)(yk-yi)-(yj-yi)(xk-xi))
        # Gradient wrt point i: -0.5*(perp of (pj-pi)) etc. We'll do finite-diff-free analytic.
        J = np.zeros((m, 2 * N))
        for r, ti in enumerate(crit):
            i, j, k = tri_idx[ti]
            pi = p[i]; pj = p[j]; pk = p[k]
            # signed area A = 0.5*((pj-pi)x(pk-pi))
            # dA/dpi = 0.5*( (pj-pi) x d(pi) ... ) -> dA/dpi = -0.5*perp2d(pj-pi)?? Let's derive.
            # A = 0.5*( (xj-xi)(yk-yi) - (yj-yi)(xk-xi) )
            # dA/dxi = 0.5*( -(yk-yi) - (yj-yi)*(-1) ) = 0.5*( -yk+yi + yj-yi ) = 0.5*(yj - yk)
            # dA/dyi = 0.5*( (xj-xi) - (xk-xi) ) = 0.5*(xj - xk)
            # dA/dxj = 0.5*(yk-yi)
            # dA/dyj = 0.5*( -(xk-xi) ) = 0.5*(xi - xk)
            # dA/dxk = 0.5*( -(yj-yi) ) = 0.5*(yi - yj)
            # dA/dyk = 0.5*(xj-xi)
            J[r, 2*i]   = 0.5 * (p[j,1] - p[k,1])
            J[r, 2*i+1] = 0.5 * (p[j,0] - p[k,0])
            J[r, 2*j]   = 0.5 * (p[k,1] - p[i,1])
            J[r, 2*j+1] = 0.5 * (p[i,0] - p[k,0])
            J[r, 2*k]   = 0.5 * (p[i,1] - p[j,1])
            J[r, 2*k+1] = 0.5 * (p[j,0] - p[i,0])
        # We want to move p by d to make all critical areas equal to mn (raise the min).
        # Objective: maximize the min. A first-order approach: increase all critical areas equally.
        # Solve min ||J d - t*1||^2  s.t. small step, where we increase area by t.
        # Simpler: solve J d = e (equal increase) in least squares, then line search.
        # Use the null-space-free least squares: d = J^T (J J^T + lam I)^{-1} 1
        Jt = J.T
        JJt = J @ Jt
        lam = 1e-6
        try:
            rhs = np.ones(m)
            coef = np.linalg.solve(JJt + lam * np.eye(m), rhs)
            d = Jt @ coef
        except np.linalg.LinAlgError:
            d = Jt @ np.linalg.pinv(JJt) @ np.ones(m)
        # Normalize step
        nrm = np.linalg.norm(d)
        if nrm < 1e-15:
            break
        d = d / nrm
        # Line search: try several step sizes, keep best min-area
        best_p = p
        best_val = mn
        for step in [0.02, 0.01, 0.005, 0.002, 0.001, -0.01, -0.005]:
            cand = project_to_triangle(p + step * d)
            val = min_area(cand)
            if val > best_val:
                best_val = val
                best_p = cand
        if best_p is p:
            break
        if best_val - mn < tol:
            p = best_p
            break
        p = best_p
    return p

def steepest_ascent(p, max_iter=200):
    """
    Direct steepest ascent on the minimum area: at each step find the min triangle,
    compute gradient of its (signed) area, move points in the increasing direction.
    """
    p = p.copy()
    for it in range(max_iter):
        areas = calc_areas_cart(p)
        mn = np.min(areas)
        idx = np.argmin(areas)
        i, j, k = tri_idx[idx]
        # signed area
        pi = p[i]; pj = p[j]; pk = p[k]
        signed = 0.5 * ((pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0]))
        # gradient of |signed| = sign(signed)*grad(signed)
        g = np.zeros(2 * N)
        sgn = 1.0 if signed >= 0 else -1.0
        g[2*i]   = sgn * 0.5 * (p[j,1] - p[k,1])
        g[2*i+1] = sgn * 0.5 * (p[j,0] - p[k,0])
        g[2*j]   = sgn * 0.5 * (p[k,1] - p[i,1])
        g[2*j+1] = sgn * 0.5 * (p[i,0] - p[k,0])
        g[2*k]   = sgn * 0.5 * (p[i,1] - p[j,1])
        g[2*k+1] = sgn * 0.5 * (p[j,0] - p[i,0])
        nrm = np.linalg.norm(g)
        if nrm < 1e-15:
            break
        d = g / nrm
        best_p = p
        best_val = mn
        for step in [0.01, 0.005, 0.002, 0.001, -0.005]:
            cand = project_to_triangle(p + step * d)
            val = min_area(cand)
            if val > best_val:
                best_val = val
                best_p = cand
        if best_p is p:
            break
        if best_val - mn < 1e-13:
            p = best_p
            break
        p = best_p
    return p

def refine_local(p_init, time_limit=4.0):
    """Multi-stage local refinement using trust-region + equalization + ascent."""
    start = time.time()
    p = project_to_triangle(p_init.copy())
    bounds = [(0.0, 1.0)] * (2 * N)

    # Stage A: trust-region on barrier objective (smooth)
    try:
        res = minimize(
            barrier_obj,
            p.flatten(),
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': 80, 'gtol': 1e-10, 'verbose': 0}
        )
        if res.x is not None:
            cand = project_to_triangle(res.x.reshape(N, 2))
            if min_area(cand) > min_area(p):
                p = cand
    except Exception:
        pass
    if time.time() - start > time_limit * 0.35:
        return p

    # Stage B: active-set equalization
    try:
        cand = equalize_step(p, max_iter=80)
        if min_area(cand) > min_area(p):
            p = cand
    except Exception:
        pass
    if time.time() - start > time_limit * 0.6:
        return p

    # Stage C: steepest ascent on min triangle
    try:
        cand = steepest_ascent(p, max_iter=150)
        if min_area(cand) > min_area(p):
            p = cand
    except Exception:
        pass
    if time.time() - start > time_limit * 0.8:
        return p

    # Stage D: another equalization + ascent polish
    try:
        cand = equalize_step(p, max_iter=60)
        cand = steepest_ascent(cand, max_iter=100)
        if min_area(cand) > min_area(p):
            p = cand
    except Exception:
        pass

    # Stage E: trust-region polish on smooth min
    try:
        res = minimize(
            smooth_min_obj,
            p.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            args=(80.0,),
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        cand = project_to_triangle(res.x.reshape(N, 2))
        if min_area(cand) > min_area(p):
            p = cand
    except Exception:
        pass

    return p

def hex_init(scale):
    """Hexagonal lattice points inside triangle, pick N."""
    pts = []
    max_k = int(1.0 / scale) + 1
    for k in range(max_k + 1):
        w = k * scale
        if w > 1.0 + 1e-9:
            break
        n_in_row = int((1.0 - w) / scale) + 1
        for j in range(n_in_row):
            u = j * scale
            v = (1.0 - w) - u
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1.0 + 1e-9:
                pts.append([u, v])
    pts = np.array(pts)
    if len(pts) == 0:
        return None
    if len(pts) == N:
        return bary_to_cart(pts)
    # Choose subset: prefer spread. Use farthest-point sampling.
    rng = np.random.default_rng(7)
    # start from a corner
    chosen = [0]
    for _ in range(N - 1):
        d = np.min([np.linalg.norm(pts - pts[c], axis=1) for c in chosen], axis=0)
        # add farthest
        chosen.append(int(np.argmax(d)))
    return bary_to_cart(pts[chosen])

def random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1.0 - u)
    return bary_to_cart(np.column_stack([u, v]))

def perturbed_init(base, sigma, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base.shape)
    return project_to_triangle(base + noise)

def centroid_corner_init():
    """Structured: centroid + 3 corners + midpoints, then fill."""
    bary = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.5],
        [1.0/3, 1.0/3],
        [0.25, 0.25], [0.5, 0.0], [0.0, 0.5],
        [0.75, 0.25], [0.25, 0.75], [0.5, 0.5]
    ])
    bary = np.clip(bary, 1e-6, 1.0)
    s = bary.sum(axis=1, keepdims=True)
    bad = s > 1.0
    if np.any(bad):
        bary[bad] *= (1.0 / s[bad])
    return bary_to_cart(bary)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_p = None
    best_val = -np.inf

    def update(p):
        nonlocal best_p, best_val
        p = project_to_triangle(p)
        val = min_area(p)
        if val > best_val:
            best_val = val
            best_p = p.copy()
            return True
        return False

    # 1. Structured initializations + local refine
    inits = []
    for sc in [0.18, 0.22, 0.27, 0.33, 0.4]:
        h = hex_init(sc)
        if h is not None:
            inits.append(h)
    inits.append(centroid_corner_init())
    for s in [11, 22, 33]:
        inits.append(random_init(s))

    for init in inits:
        if time.time() > deadline - 20:
            break
        tlim = min(6.0, max(1.5, deadline - time.time() - 15))
        cand = refine_local(init, time_limit=tlim)
        update(cand)

    # 2. Multi-start: perturb best + fresh randoms, with local refine
    rng = np.random.default_rng(999)
    rc = 0
    while time.time() < deadline - 12:
        rc += 1
        if best_p is not None and rc % 2 == 0:
            sigma = 0.03 + 0.02 * (rc % 4)
            init = perturbed_init(best_p, sigma, seed=5000 + rc)
        else:
            init = random_init(7000 + rc)
        tlim = min(4.5, max(1.0, deadline - time.time() - 10))
        cand = refine_local(init, time_limit=tlim)
        update(cand)

    # 3. Final high-precision polish on best
    if best_p is not None and time.time() < deadline - 8:
        try:
            cand = equalize_step(best_p, max_iter=120)
            cand = steepest_ascent(cand, max_iter=200)
            update(cand)
        except Exception:
            pass

    if best_p is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                smooth_min_obj,
                best_p.flatten(),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                args=(150.0,),
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            update(project_to_triangle(res.x.reshape(N, 2)))
        except Exception:
            pass

    if best_p is not None and time.time() < deadline - 3:
        try:
            cand = equalize_step(best_p, max_iter=100)
            update(cand)
        except Exception:
            pass

    if best_p is not None and time.time() < deadline - 1.0:
        try:
            cand = steepest_ascent(best_p, max_iter=300)
            update(cand)
        except Exception:
            pass

    if best_p is None:
        best_p = random_init(0)

    return project_to_triangle(best_p)

# EVOLVE-BLOCK-END
