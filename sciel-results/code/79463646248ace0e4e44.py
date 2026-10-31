import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2
TRIS = np.array(list(combinations(range(N), 3)))
TRI_A = TRIS[:, 0]
TRI_B = TRIS[:, 1]
TRI_C = TRIS[:, 2]


def bary_to_cart(bary):
    """bary (n,3) -> cartesian (n,2) for triangle (0,0),(1,0),(0.5,H)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = bary[:, 2] * H
    return np.column_stack([x, y])


def cart_to_bary(cart):
    c = cart[:, 1] / H
    b = cart[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def clamp_bary(bary):
    bary = np.clip(bary, 0.0, 1.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return bary / s


def all_areas_cart(cart):
    pa = cart[TRI_A]
    pb = cart[TRI_B]
    pc = cart[TRI_C]
    return 0.5 * np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                        - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0]))


def hard_min_bary(flat):
    bary = clamp_bary(flat.reshape(N, 3))
    return float(np.min(all_areas_cart(bary_to_cart(bary))))


def soft_min_obj(flat, beta=120.0):
    """Smooth surrogate of min area (differentiable)."""
    bary = clamp_bary(flat.reshape(N, 3))
    areas = all_areas_cart(bary_to_cart(bary))
    m = areas.min()
    shifted = np.clip(-beta * (areas - m), -500, 500)
    logsum = np.log(np.sum(np.exp(shifted)))
    return m - logsum / beta


def _delaunay_smooth_obj(flat, beta=120.0, lam=0.03, gamma=40.0):
    """
    Hybrid objective: soft-min area PLUS a variance-regularization term on the
    smallest areas. Encourages a *balanced* set of critical triangles, which
    smooths the landscape and helps escape shallow local minima.
    """
    bary = clamp_bary(flat.reshape(N, 3))
    cart = bary_to_cart(bary)
    areas = all_areas_cart(cart)
    m = areas.min()
    shifted = np.clip(-beta * (areas - m), -500, 500)
    logsum = np.log(np.sum(np.exp(shifted)))
    softmin = m - logsum / beta

    # Delaunay-aware: penalize spread among the smallest triangles.
    try:
        tri = Delaunay(cart)
        d_areas = 0.5 * np.abs((cart[tri.simplices[:, 1], 0] - cart[tri.simplices[:, 0], 0])
                               * (cart[tri.simplices[:, 2], 1] - cart[tri.simplices[:, 0], 1])
                               - (cart[tri.simplices[:, 1], 1] - cart[tri.simplices[:, 0], 1])
                               * (cart[tri.simplices[:, 2], 0] - cart[tri.simplices[:, 0], 0]))
        k = min(20, len(d_areas))
        small = np.sort(d_areas)[:k]
        var_term = np.var(small)
    except Exception:
        small = np.sort(areas)[:20]
        var_term = np.var(small)

    # A soft-min of the *variance* (so it's smooth), scaled by lam.
    sm = small.min()
    sh2 = np.clip(-gamma * (small - sm), -500, 500)
    lse2 = np.log(np.sum(np.exp(sh2)))
    var_soft = sm - lse2 / gamma
    return softmin - lam * var_soft


def _neg_hybrid(flat, beta, lam, gamma):
    return -_delaunay_smooth_obj(flat, beta, lam, gamma)


def _neg_soft(flat, beta):
    return -soft_min_obj(flat, beta)


def _neg_hard(flat):
    return -hard_min_bary(flat)


def generate_hexagonal_init():
    pts = []
    pts.append([1.0, 0.0, 0.0])
    pts.append([0.0, 1.0, 0.0])
    pts.append([0.0, 0.0, 1.0])
    pts.append([0.5, 0.5, 0.0])
    pts.append([0.0, 0.5, 0.5])
    pts.append([0.5, 0.0, 0.5])
    pts.append([1 / 3, 1 / 3, 1 / 3])
    cx, cy, cz = 1 / 3, 1 / 3, 1 / 3
    r = 0.13
    for angle in [0, np.pi / 3, 2 * np.pi / 3, np.pi]:
        da = r * np.cos(angle)
        db = r * np.sin(angle)
        dc = -da - db
        p = [cx + da, cy + db, cz + dc]
        p = np.array(p)
        p = p / max(p.sum(), 1e-12)
        pts.append(p)
    return np.array(pts)


def generate_grid_init():
    pts = []
    for i in range(N):
        # spread along a rough triangular lattice
        row = i // 4
        col = i % 4
        a = 1.0 - 0.25 * row - 0.18 * col
        b = 0.25 * row + 0.18 * col
        c = 0.25 * row + 0.18 * col
        p = np.array([a, b, c])
        p = np.clip(p, 0, 1)
        p = p / p.sum()
        pts.append(p)
    return np.array(pts)


def generate_random_init(rng):
    r1 = rng.random(N)
    r2 = rng.random(N)
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    bary = np.column_stack([1 - r1 - r2, r1, r2])
    return bary


def generate_perturbed_init(base, rng, scale):
    bary = base + rng.normal(0, scale, size=base.shape)
    return clamp_bary(bary)


def _local_polish_bary(bary, time_left_fn, budget=20.0):
    """L-BFGS-B on hybrid + Nelder-Mead on hard, from a bary starting point."""
    best_bary = bary.copy()
    best_score = hard_min_bary(best_bary)
    start = time.time()

    for beta in [80.0, 200.0]:
        if time_left_fn() < 8.0:
            break
        try:
            res = minimize(
                _neg_hybrid,
                best_bary.flatten(),
                args=(beta, 0.03, 40.0),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (N * 3),
                options={'maxiter': 120, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            cand = clamp_bary(res.x.reshape(N, 3))
            s = hard_min_bary(cand)
            if s > best_score:
                best_score = s
                best_bary = cand.copy()
        except Exception:
            pass

    # Nelder-Mead on the hard objective for exact polish.
    if time_left_fn() > 6.0:
        try:
            res = minimize(
                _neg_hard,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            cand = clamp_bary(res.x.reshape(N, 3))
            s = hard_min_bary(cand)
            if s > best_score:
                best_score = s
                best_bary = cand.copy()
        except Exception:
            pass

    return best_bary, best_score


def _simulated_annealing(bary, time_left_fn, budget=25.0):
    """SA on the hard objective with adaptive step size; accepts worse moves."""
    best_bary = bary.copy()
    best_score = hard_min_bary(best_bary)
    cur_bary = bary.copy()
    cur_score = best_score
    start = time.time()
    T = 0.02
    step = 0.06
    rng = np.random.default_rng(12345)

    while time_left_fn() > 3.0:
        for _ in range(300):
            if time_left_fn() < 3.0:
                break
            move = rng.normal(0, step, size=(N, 3))
            cand = clamp_bary(cur_bary + move)
            s = hard_min_bary(cand)
            delta = s - cur_score
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                cur_bary = cand
                cur_score = s
                if s > best_score:
                    best_score = s
                    best_bary = cand.copy()
        T *= 0.95
        step *= 0.97
    return best_bary, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin under 150s

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)

    best_bary = None
    best_score = -1.0

    def consider(bary):
        nonlocal best_bary, best_score
        bary = clamp_bary(bary)
        s = hard_min_bary(bary)
        if s > best_score:
            best_score = s
            best_bary = bary.copy()
        return s

    # --- Stage 1: build diverse initial pool ---
    inits = [generate_hexagonal_init(), generate_grid_init()]
    for scale in [0.03, 0.08, 0.15, 0.25]:
        inits.append(generate_perturbed_init(generate_hexagonal_init(), rng, scale))
    for _ in range(8):
        inits.append(generate_random_init(rng))

    scored = [(init, consider(init)) for init in inits]
    scored.sort(key=lambda x: x[1], reverse=True)

    # --- Stage 2: global differential evolution on hybrid objective ---
    if time_left() > 45.0:
        try:
            de_res = differential_evolution(
                _neg_hybrid,
                bounds=[(0.0, 1.0)] * (N * 3),
                args=(120.0, 0.03, 40.0),
                seed=42,
                maxiter=60,
                popsize=12,
                tol=1e-8,
                polish=False,
                update_interval=1,
                callback=lambda xk, convergence: (convergence and time_left() < 45.0)
            )
            consider(de_res.x.reshape(N, 3))
        except Exception:
            pass

    # --- Stage 3: local polish on the top candidates ---
    for bary_init, init_score in scored[:6]:
        if time_left() < 15.0:
            break
        try:
            pb, ps = _local_polish_bary(bary_init, time_left)
            consider(pb)
        except Exception:
            pass

    # --- Stage 4: simulated annealing from current best ---
    if best_bary is not None and time_left() > 20.0:
        try:
            sb, ss = _simulated_annealing(best_bary, time_left)
            consider(sb)
        except Exception:
            pass

    # --- Stage 5: final exact polish ---
    if best_bary is not None and time_left() > 8.0:
        try:
            res = minimize(
                _neg_hard,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            consider(res.x.reshape(N, 3))
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_hexagonal_init()
        best_score = consider(best_bary)

    return bary_to_cart(clamp_bary(best_bary))


if __name__ == "__main__":
    pts = construct()
    print("Best score:", hard_min_bary(cart_to_bary(pts)))
