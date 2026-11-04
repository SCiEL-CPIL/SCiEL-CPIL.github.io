"""Circle packing in a perimeter-4 rectangle; maximize sum of radii for 21 circles."""
import numpy as np
from scipy.optimize import linprog

N = 21
EPS = 1e-6

# --- geometry helpers ---
def perimeter(w, h):
    return 2.0 * (w + h)

def make_rect(radii):
    """Choose a rectangle with perimeter 4 that fits the circles (centers in [r, W-r])."""
    rmax = max(radii)
    # candidate aspect ratios
    best = None
    for ar in np.linspace(0.5, 2.0, 26):
        W = 2.0 * ar
        H = 2.0 / ar
        if W < 2 * rmax or H < 2 * rmax:
            continue
        # feasible? need each circle to fit within bounds
        ok = True
        if best is None or (W + H) < best[0]:
            best = (W + H, W, H)
    if best is None:
        W = H = 1.0
    else:
        _, W, H = best
    return W, H

# --- objective: maximize sum r_i subject to disjointness (linearized) + bounds ---
def linear_packing(radii, centers, W, H, n_iter=40, tol=1e-9):
    """
    Iteratively linearize circle-circle constraints and solve LP.
    Returns improved radii, centers.
    """
    n = len(radii)
    r = np.array(radii, dtype=float).copy()
    c = np.array(centers, dtype=float).copy()  # (n,2)
    best = r.copy()
    best_c = c.copy()
    best_val = r.sum()

    for it in range(n_iter):
        # build linearized constraints: for each pair, (dx,dy) dot (u_i - u_j) >= r_i + r_j - eps
        # variables: x = [r_0..r_{n-1}, c_0x,c_0y,...,c_{n-1}x,c_{n-1}y]
        nv = 2 * n
        # objective: maximize sum r -> minimize -sum r
        c_obj = np.zeros(nv)
        c_obj[:n] = -1.0

        A_ub = []
        b_ub = []

        # circle-circle: for each pair (i,j) with i<j
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    d = np.array([1.0, 0.0])
                    dist = 1.0
                u = d / dist
                # constraint: (x_i - x_j)*u_x + (y_i - y_j)*u_y >= r_i + r_j
                # => r_i + r_j - (x_i-x_j)*u_x - (y_i-y_j)*u_y <= 0
                row = np.zeros(nv)
                row[i] = 1.0
                row[j] = 1.0
                row[n + 2*i] = -u[0]
                row[n + 2*i + 1] = -u[1]
                row[n + 2*j] = u[0]
                row[n + 2*j + 1] = u[1]
                A_ub.append(row)
                b_ub.append(EPS)

        # bounds: 0 <= r_i <= max_r ; r_i <= x_i <= W - r_i ; r_i <= y_i <= H - r_i
        max_r = min(W, H) / 2.0
        lb = np.zeros(nv)
        ub = np.zeros(nv)
        ub[:n] = max_r
        # center bounds depend on r_i; use loose bounds for LP, enforce in polish
        for i in range(n):
            ub[n + 2*i] = W
            ub[n + 2*i + 1] = H
            lb[n + 2*i] = 0.0
            lb[n + 2*i + 1] = 0.0

        res = linprog(c_obj, A_ub=np.array(A_ub) if A_ub else None,
                      b_ub=np.array(b_ub) if b_ub else None,
                      bounds=list(zip(lb, ub)), method='highs')
        if not res.success:
            break
        x = res.x
        r_new = x[:n]
        c_new = x[n:].reshape(n, 2)
        # clip tiny negatives
        r_new = np.clip(r_new, 0.0, max_r)
        val = r_new.sum()
        if val > best_val + 1e-12:
            best_val = val
            best = r_new.copy()
            best_c = c_new.copy()
        r = r_new
        c = c_new
        # check convergence
        if abs(val - best_val) < tol and it > 5:
            break

    return best, best_c

# --- force-directed layout ---
def force_layout(radii, centers, W, H, n_iter=8000, repulse=50.0, contact=1.02, seed=0):
    """
    Minimize overlap via repulsive forces, while pulling circles to fill space.
    Returns radii, centers.
    """
    n = len(radii)
    r = np.array(radii, dtype=float).copy()
    c = np.array(centers, dtype=float).copy()
    rng = np.random.default_rng(seed)

    # initial positions: jittered grid
    side = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        ix = i % side
        iy = i // side
        c[i, 0] = (ix + 0.5) / side * (W - 2*r[i]) + r[i] + rng.uniform(-0.02, 0.02)
        c[i, 1] = (iy + 0.5) / side * (H - 2*r[i]) + r[i] + rng.uniform(-0.02, 0.02)
        c[i, 0] = np.clip(c[i, 0], r[i], W - r[i])
        c[i, 1] = np.clip(c[i, 1], r[i], H - r[i])

    # adaptive step
    step = 0.02
    for it in range(n_iter):
        # forces
        F = np.zeros_like(c)
        # repulsion between overlapping (or near) circles
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                min_dist = (r[i] + r[j]) * contact
                if dist < 1e-12:
                    d = rng.normal(size=2)
                    dist = np.linalg.norm(d)
                if dist < min_dist:
                    overlap = min_dist - dist
                    u = d / dist
                    # push apart
                    F[i] += repulse * overlap * u
                    F[j] -= repulse * overlap * u
        # wall repulsion
        for i in range(n):
            if c[i, 0] < r[i]:
                F[i, 0] += repulse * (r[i] - c[i, 0])
            elif c[i, 0] > W - r[i]:
                F[i, 0] -= repulse * (c[i, 0] - (W - r[i]))
            if c[i, 1] < r[i]:
                F[i, 1] += repulse * (r[i] - c[i, 1])
            elif c[i, 1] > H - r[i]:
                F[i, 1] -= repulse * (c[i, 1] - (H - r[i]))

        # move
        c += step * F
        # clamp to bounds
        for i in range(n):
            c[i, 0] = np.clip(c[i, 0], r[i], W - r[i])
            c[i, 1] = np.clip(c[i, 1], r[i], H - r[i])

        # decay step
        step *= 0.999

    return r, c

# --- polish: ensure disjointness by scaling down if needed ---
def ensure_disjoint(r, c, W, H):
    """If any overlap, scale radii down slightly until disjoint."""
    n = len(r)
    for _ in range(200):
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                need = r[i] + r[j]
                if d < need:
                    max_overlap = max(max_overlap, need - d)
        if max_overlap < EPS:
            break
        # scale all radii down a tiny bit
        r *= (1.0 - max_overlap * 0.01)
    # also ensure within bounds
    for i in range(n):
        c[i, 0] = np.clip(c[i, 0], r[i], W - r[i])
        c[i, 1] = np.clip(c[i, 1], r[i], H - r[i])
    return r, c

def construct():
    import time
    t0 = time.time()
    DEADLINE = 120.0

    # Start with equal radii
    # Perimeter 4 => W+H=2. For 21 circles, a good starting radius:
    # area of rectangle ~ (W*H), 21 circles of radius r: 21*pi*r^2 <= W*H
    # Try W=1.4, H=0.6 => area=0.84; 21*pi*r^2=0.84 => r^2=0.84/(21*pi)=0.01273 => r~0.1128
    W, H = 1.4, 0.6
    r0 = 0.10
    radii = np.full(N, r0)
    centers = np.zeros((N, 2))

    # initial grid
    side = int(np.ceil(np.sqrt(N)))
    for i in range(N):
        ix = i % side
        iy = i // side
        centers[i, 0] = (ix + 0.5) / side * (W - 2*r0) + r0
        centers[i, 1] = (iy + 0.5) / side * (H - 2*r0) + r0

    # --- multi-restart force-directed + LP polish ---
    best_score = -1.0
    best_r = radii.copy()
    best_c = centers.copy()
    best_W, best_H = W, H

    restarts = 6
    for rs in range(restarts):
        if time.time() - t0 > DEADLINE - 10.0:
            break
        # vary aspect ratio
        ar = 0.7 + 0.3 * rs / max(1, restarts - 1)
        W_r = 2.0 * ar
        H_r = 2.0 / ar
        # ensure fits
        if W_r < 2 * r0 or H_r < 2 * r0:
            continue

        # force layout
        r_f, c_f = force_layout(radii, centers, W_r, H_r, n_iter=6000, repulse=50.0, contact=1.02, seed=rs)

        # LP polish
        r_lp, c_lp = linear_packing(r_f, c_f, W_r, H_r, n_iter=30)

        # ensure disjoint
        r_lp, c_lp = ensure_disjoint(r_lp, c_lp, W_r, H_r)

        score = r_lp.sum()
        if score > best_score:
            best_score = score
            best_r = r_lp.copy()
            best_c = c_lp.copy()
            best_W, best_H = W_r, H_r

    # final: verify and return
    r = best_r
    c = best_c
    W, H = best_W, best_H

    # final disjointness check
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(c[i] - c[j])
            if d < r[i] + r[j] - 1e-6:
                # shrink both slightly
                shrink = (r[i] + r[j] - d) / 2.0 + 1e-6
                r[i] -= shrink
                r[j] -= shrink
    # bounds
    for i in range(n):
        c[i, 0] = np.clip(c[i, 0], r[i], W - r[i])
        c[i, 1] = np.clip(c[i, 1], r[i], H - r[i])

    out = np.zeros((n, 3))
    out[:, 0] = c[:, 0]
    out[:, 1] = c[:, 1]
    out[:, 2] = r
    return out
