import time
import numpy as np

_N = 8
_FRAC = 1.0 / _N
_IDX8 = np.arange(_N) / _N
_C = np.linspace(0.0, 1.0, 401)  # 401 points in [0,1]


def _union_len(C, base):
    """Vectorized union length of N intervals [base[i] + c*i/N, +1/N] for each c in C."""
    shifts = C[:, None] * _IDX8[None, :]
    l = base[None, :] + shifts
    r = l + _FRAC
    order = np.argsort(l, axis=1)
    ls = np.take_along_axis(l, order, 1)
    rs = np.take_along_axis(r, order, 1)
    union = (rs[:, 0] - ls[:, 0]).copy()
    cur = rs[:, 0:1]
    for k in range(1, _N):
        s = ls[:, k:k + 1]
        e = rs[:, k:k + 1]
        ne = np.maximum(cur, e)
        union = union + np.maximum(ne - np.maximum(cur, s), 0.0)
        cur = ne
    return union


def _v_to_fg(v):
    f = np.zeros(_N)
    g = np.zeros(_N)
    f[1:] = v[:7]
    g[1:] = v[7:]
    return f, g


def _obj(v):
    f, g = _v_to_fg(v)
    uf = _union_len(_C, f)
    ug = _union_len(_C, g)
    val = float(np.mean(uf * ug))
    if not np.isfinite(val):
        return 1e9
    return val


def _keich_v():
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N)
    for idx in range(N):
        bits = np.array(list(str(bin(idx))[2:]), dtype=int)
        bits = np.pad(bits, (pos - len(bits), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * bits * (2.0 ** -j_terms) / pos)
    keich = keich - keich[0]
    return np.concatenate([keich[1:], keich[1:]])


def kakeya_needle_3d8() -> np.ndarray:
    N = _N

    def build(f, g):
        i = np.arange(N * N) // N
        j = np.arange(N * N) % N
        return np.stack([f[i], g[j]], axis=1)

    start = time.time()
    LIMIT = 630.0
    DEADLINE = LIMIT - 30.0

    # Start with the known good Keich seed
    keich = _keich_v()
    best_v = keich.copy()
    best_val = _obj(keich)

    # Discrete lattice search:
    # We quantize the 14 degrees of freedom (f[1..7], g[1..7]) to a grid.
    # The range [-1.5, 1.5] with step 0.25 gives 13 levels per dim.
    # 13^14 is too large for exhaustive, but we can do a structured search.
    # However, a better discrete strategy for this specific problem is to
    # recognize that the optimal positions often align with the shear steps.
    
    # Let's try a "snap to grid" refinement from the Keich seed and random starts.
    # But the directive asks for a MATERIALLY different strategy.
    # The current program uses continuous optimization.
    # I will implement a **Discrete Simulated Annealing** or **Grid Search** on a coarse lattice.
    
    # Actually, the most powerful discrete approach for this specific Kakeya problem
    # is to realize that the optimal f and g are often permutations or simple shifts
    # of the indices. 
    # Let's try searching over all permutations of the indices 0..7 for f and g?
    # No, the positions are continuous.
    
    # Let's do a **Coarse Grid Search** with **Local Discrete Optimization**.
    # Grid points: -1.5, -1.25, ..., 1.5 (step 0.25). 13 points.
    # We will evaluate the objective on this grid for a subset of dimensions at a time?
    # No, that's coordinate descent.
    
    # Let's try **Random Discrete Search**:
    # Sample 100,000 random configurations from a discrete grid (step 0.25) 
    # and keep the best. This is fast because _obj is vectorized.
    
    # Even better: **Structured Discrete Search**.
    # The Keich construction is known to be near-optimal.
    # Let's perturb the Keich construction on a discrete grid.
    
    # Let's try a **Full Grid Search** on a reduced dimensionality?
    # No.
    
    # Let's try **Discrete Coordinate Descent** from the Keich seed.
    # Start at Keich. For each dimension, try all grid points and pick the best.
    # Repeat until no improvement.
    
    grid_step = 0.25
    grid_min = -2.0
    grid_max = 2.0
    grid_points = np.arange(grid_min, grid_max + grid_step, grid_step)
    
    def discrete_coord_descent(v0):
        v = v0.copy()
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for d in range(14):
                best_val_dim = _obj(v)
                best_v_dim = v.copy()
                for gp in grid_points:
                    v[d] = gp
                    val = _obj(v)
                    if val < best_val_dim - 1e-12:
                        best_val_dim = val
                        best_v_dim = v.copy()
                if best_val_dim < _obj(v) - 1e-12:
                    v = best_v_dim
                    improved = True
        return v, best_val_dim

    # Run discrete coordinate descent from Keich
    v_keich, val_keich = discrete_coord_descent(keich)
    if val_keich < best_val:
        best_v = v_keich
        best_val = val_keich

    # Run discrete coordinate descent from a few random discrete starts
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() > DEADLINE:
            break
        v0 = rng.choice(grid_points, size=14)
        v_rand, val_rand = discrete_coord_descent(v0)
        if val_rand < best_val:
            best_v = v_rand
            best_val = val_rand

    # Final polish with continuous L-BFGS-B if time remains
    if time.time() < DEADLINE - 10:
        try:
            from scipy import optimize
            bounds = [(-3.0, 3.0)] * 14
            r = optimize.minimize(_obj, best_v, method="L-BFGS-B", bounds=bounds,
                                  options=dict(maxiter=200, maxfun=1000, ftol=1e-12, gtol=1e-10))
            if r is not None:
                val = _obj(r.x)
                if val < best_val:
                    best_v = r.x
                    best_val = val
        except Exception:
            pass

    f, g = _v_to_fg(best_v)
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
        kv = _keich_v()
        f = np.concatenate([[0.0], kv[:7]])
        g = np.concatenate([[0.0], kv[7:14]])
    pos = build(f, g)
    if pos.shape != (N * N, 2) or not np.all(np.isfinite(pos)):
        kv = _keich_v()
        pos = build(np.concatenate([[0.0], kv[:7]]), np.concatenate([[0.0], kv[7:14]]))
    return pos
