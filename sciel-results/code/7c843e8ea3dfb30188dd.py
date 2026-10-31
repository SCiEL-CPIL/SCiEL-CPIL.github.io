import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)          # (165,3)
M = tri_idx.shape[0]                     # 165

H = np.sqrt(3) / 2

# --- Build the linear-area coefficient matrix A (M x 22) once.
# area_j = |det| ; but sign depends on ordering. We want area_j >= t.
# Since area_j = |det_j| and det_j is linear in (u,v) of the 3 points,
# we have area_j >= t  <=>  det_j >= t  OR  det_j <= -t.
# To keep a single LP, we fix an orientation per triangle by choosing the
# sign that makes det positive at a reference feasible point, then require
# det_j >= t (we can flip the row if needed at each LP call based on current
# sign). Simpler robust approach: use both constraints? No -- that would force
# |det|>=t via two linear constraints which is NOT convex.
#
# Correct convex approach: the feasible set {x : min_j |det_j(x)| >= t} is the
# intersection over j of {|det_j| >= t}, each of which is NON-convex (union of
# two halfspaces). So a single LP cannot capture the hard-min exactly.
#
# Therefore we use the epigraph with a FIXED orientation: pick, for the current
# iterate, the sign s_j = sign(det_j(x0)) for each triangle, then maximize t
# subject to s_j * det_j(x) >= t. This is a valid LP and is a local (in the
# orientation sense) relaxation; polishing with Newton on the active set
# recovers the true local optimum of the piecewise-linear objective.

def build_A():
    """Return matrix A (M x 22) such that det_j = A[j,:] @ z, where
    z = [u0,v0,u1,v1,...,u10,v10]."""
    A = np.zeros((M, 2 * N))
    for m, (i, j, k) in enumerate(tri_idx):
        # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
        # with w = 1 - u - v.
        # Expand each term in (u,v) of i,j,k.
        # We'll compute symbolically by evaluating A rows via finite construction:
        # det is linear in each variable, so we can get coefficients by
        # probing. But cleaner: derive formula.
        # Let's just compute using the linearity: set basis vectors.
        # Coefficient of u_p: det with u_p=1 (others 0) minus det with u_p=0.
        # We'll do it numerically once.
        pass
    # Numeric construction:
    for m, (i, j, k) in enumerate(tri_idx):
        def det_at(z):
            u = z[0::2]; v = z[1::2]; w = 1 - u - v
            ui, vi, wi = u[i], v[i], w[i]
            uj, vj, wj = u[j], v[j], w[j]
            uk, vk, wk = u[k], v[k], w[k]
            return ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
        for p in range(N):
            for c in range(2):
                z0 = np.zeros(2 * N)
                z1 = np.zeros(2 * N)
                z0[2*p + c] = 1.0
                z1[2*p + c] = -1.0
                # linear => coeff = (det(z0)-det(z1))/2  (since z1 = -z0)
                A[m, 2*p + c] = (det_at(z0) - det_at(z1)) / 2.0
    return A

A = build_A()

def clamp_bary(bary, eps=1e-9):
    u = np.clip(bary[:, 0].copy(), eps, 1 - eps)
    v = np.clip(bary[:, 1].copy(), eps, 1 - eps)
    s = u + v
    bad = s > (1 - eps)
    if np.any(bad):
        scale = (1 - eps) / np.where(s > 0, s, 1.0)
        u[bad] *= scale[bad]
        v[bad] *= scale[bad]
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    return np.column_stack([u, v])

def bary_to_cart(bary):
    u = bary[:, 0]; v = bary[:, 1]; w = 1 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def areas_from_bary(bary):
    u = bary[:, 0]; v = bary[:, 1]; w = 1 - u - v
    ui, vi, wi = u[tri_idx[:, 0]], v[tri_idx[:, 0]], w[tri_idx[:, 0]]
    uj, vj, wj = u[tri_idx[:, 1]], v[tri_idx[:, 1]], w[tri_idx[:, 1]]
    uk, vk, wk = u[tri_idx[:, 2]], v[tri_idx[:, 2]], w[tri_idx[:, 2]]
    det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    return det  # signed

def hard_min_val(bary):
    return np.min(np.abs(areas_from_bary(bary)))

def solve_active_lp(bary0, tol=1e-12):
    """Maximize t subject to s_j * (A_j . z) >= t, 0<=u,v<=1, u+v<=1.
    z = [u0..u10, v0..v10] (22 vars) + t (1 var) = 23 vars.
    s_j = sign(det_j(bary0)) (use +1 if zero).
    Returns (bary, t)."""
    z0 = np.zeros(2 * N)
    z0[0::2] = bary0[:, 0]
    z0[1::2] = bary0[:, 1]
    det0 = A @ z0
    s = np.sign(det0)
    s[s == 0] = 1.0

    # Variables: z (22) then t (1). Index of t = 22.
    nvar = 2 * N + 1
    t_idx = 2 * N

    # Objective: maximize t  =>  minimize -t
    c = np.zeros(nvar)
    c[t_idx] = -1.0

    # Inequality constraints:  -s_j * A_j . z + t <= 0   for all j
    # i.e.  t - s_j*(A_j . z) <= 0
    A_ub = np.zeros((M, nvar))
    A_ub[:, 0::2] = -s[:, None] * A[:, 0::2]
    A_ub[:, 1::2] = -s[:, None] * A[:, 1::2]
    A_ub[:, t_idx] = 1.0
    b_ub = np.zeros(M)

    # Bounds: 0 <= u_i <= 1, 0 <= v_i <= 1, t free (use large bound)
    lb = np.full(nvar, -np.inf)
    ub = np.full(nvar, np.inf)
    for p in range(N):
        lb[2*p] = 0.0; ub[2*p] = 1.0
        lb[2*p+1] = 0.0; ub[2*p+1] = 1.0
    lb[t_idx] = -1.0
    ub[t_idx] = 1.0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)),
                  method='highs')
    if not res.success:
        return bary0, hard_min_val(bary0)

    z = res.x[:2 * N]
    t = res.x[t_idx]
    bary = np.column_stack([z[0::2], z[1::2]])
    bary = clamp_bary(bary)
    # re-evaluate true min area
    val = hard_min_val(bary)
    return bary, val

def polish_active(bary0, iters=30, tol=1e-12):
    """Iterate: solve active LP, then re-orient and repeat. Stop when t stable."""
    bary = bary0.copy()
    best = hard_min_val(bary)
    for _ in range(iters):
        bary_new, t = solve_active_lp(bary, tol=tol)
        val = hard_min_val(bary_new)
        if val > best:
            best = val
            bary = bary_new
        else:
            # try a tiny perturbation to escape degenerate orientation
            rng = np.random.default_rng(abs(hash(tuple(bary.flatten().round(12)))) % (2**31))
            bary_try = clamp_bary(bary + rng.normal(0, 1e-4, bary.shape))
            bary_try, val_try = solve_active_lp(bary_try)
            if val_try > best:
                best = val_try
                bary = bary_try
            else:
                break
    return bary, best

def generate_hexagonal_init():
    best_bary = None
    best_val = -np.inf
    for scale in [0.18, 0.22, 0.26, 0.30, 0.34, 0.38]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for _ in range(3):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = hard_min_val(bary)
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    return best_bary

def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_local(bary_init, time_limit=5.0):
    start = time.time()
    bary, val = polish_active(bary_init, iters=25)
    if time.time() - start < time_limit * 0.5:
        bary2, val2 = polish_active(bary, iters=20)
        if val2 > val:
            bary, val = bary2, val2
    return bary, val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = hard_min_val(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()

    # 1. Hexagonal seed
    hex_init = generate_hexagonal_init()
    b, v = refine_local(hex_init, time_limit=10.0)
    update(b)

    # 2. Differential evolution to find diverse basins (coarse), then active-set polish
    if time.time() < deadline - 40:
        def de_obj(x):
            bary = clamp_bary(x.reshape(N, 2))
            return -hard_min_val(bary)
        bounds = [(0, 1)] * (2 * N)
        try:
            res = differential_evolution(de_obj, bounds, seed=42, maxiter=25,
                                         popsize=10, tol=1e-5, polish=False,
                                         init='sobol')
            de_bary = clamp_bary(res.x.reshape(N, 2))
            b, v = refine_local(de_bary, time_limit=10.0)
            update(b)
        except Exception:
            pass

    # 3. Multi-start with active-set polish
    rng = np.random.default_rng(7)
    i = 0
    while time.time() < deadline - 15:
        i += 1
        if best_bary is not None and i % 2 == 0:
            init = clamp_bary(best_bary + rng.normal(0, 0.03, best_bary.shape))
        else:
            init = generate_random_init(seed=1000 + i)
        b, v = refine_local(init, time_limit=4.0)
        update(b)

    # 4. Final high-iteration polish on best
    if best_bary is not None and time.time() < deadline - 3:
        b, v = polish_active(best_bary, iters=60)
        update(b)
        if time.time() < deadline - 1:
            b, v = polish_active(best_bary, iters=40)
            update(b)

    return bary_to_cart(best_bary)
