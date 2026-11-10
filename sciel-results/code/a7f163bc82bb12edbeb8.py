import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 40s margin for safety

    N = 8
    w = 1.0 / N
    slopes = np.arange(N, dtype=np.float64) / N

    def get_union_length(c, z):
        starts = z + c * slopes
        order = np.argsort(starts, kind="stable")
        s_sorted = starts[order]
        e_sorted = s_sorted + w
        length = 0.0
        current_end = -np.inf
        for s, e in zip(s_sorted, e_sorted):
            if s > current_end:
                length += (e - s)
                current_end = e
            else:
                if e > current_end:
                    length += (e - current_end)
                    current_end = e
        return length

    def get_criticals(z):
        crit = [0.0, 1.0]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = i - j
                c1 = N * (z[j] - z[i]) / denom
                if 0.0 < c1 < 1.0:
                    crit.append(c1)
                c2 = N * (z[j] - z[i] + w) / denom
                if 0.0 < c2 < 1.0:
                    crit.append(c2)
                c3 = N * (z[j] - z[i] - w) / denom
                if 0.0 < c3 < 1.0:
                    crit.append(c3)
        return sorted(set(crit))

    def compute_cost(z):
        if not np.all(np.isfinite(z)):
            return 1e10
        crit = get_criticals(z)
        total = 0.0
        for k in range(len(crit) - 1):
            c0 = crit[k]
            c1 = crit[k + 1]
            h = c1 - c0
            if h < 1e-12:
                continue
            L0 = get_union_length(c0, z)
            L1 = get_union_length(c1, z)
            A = (L1 - L0) / h
            B = L0 - A * c0
            total += (A**2 * (c1**3 - c0**3)) / 3.0 + (A * B * (c1**2 - c0**2)) + (B**2 * h)
        return total

    def compute_grad(z):
        if not np.all(np.isfinite(z)):
            return np.zeros(N)
        crit = get_criticals(z)
        grad = np.zeros(N, dtype=np.float64)
        for k in range(len(crit) - 1):
            c0 = crit[k]
            c1 = crit[k + 1]
            h = c1 - c0
            if h < 1e-12:
                continue
            L0 = get_union_length(c0, z)
            L1 = get_union_length(c1, z)
            A = (L1 - L0) / h
            B = L0 - A * c0
            # d/dc [ (A c + B)^2 ] = 2 A (A c + B); integrate c^{2m} over [c0,c1]
            I0 = h
            I1 = (c1**2 - c0**2) / 2.0
            I2 = (c1**3 - c0**3) / 3.0
            dcdz = 2.0 * A * (A * (A * I2 + B * I1) + B * (A * I1 + B * I0))
            # dL/dc at each endpoint (subgradient at kinks)
            for c in (c0, c1):
                starts = z + c * slopes
                order = np.argsort(starts, kind="stable")
                s_sorted = starts[order]
                e_sorted = s_sorted + w
                max_end = np.maximum.accumulate(e_sorted)
                min_start = np.minimum.accumulate(s_sorted)
                active = np.where((e_sorted > min_start) & (s_sorted < max_end))[0]
                dLdc = np.zeros(N, dtype=np.float64)
                if active.size > 0:
                    dLdc[order[active[0]]] += 1.0
                    dLdc[order[active[-1]]] -= 1.0
                grad += dcdz * dLdc
        return grad

    # Keich initial guess (x = y symmetry)
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    best_z = keich.copy()
    best_cost = compute_cost(keich)

    rng = np.random.default_rng(42)
    inits = [keich.copy()]
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        if best_cost < 1e-2:
            z0 = best_z + rng.normal(0, 0.01, N)
        else:
            z0 = rng.uniform(-0.5, 0.5, N)
        inits.append(z0)

    for z0 in inits:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(compute_cost, z0, jac=compute_grad, method="L-BFGS-B",
                           options={"maxiter": 4000, "gtol": 1e-12, "ftol": 1e-14,
                                    "maxls": 40, "maxcor": 20})
            if np.all(np.isfinite(res.x)) and res.fun < best_cost:
                best_cost = res.fun
                best_z = res.x.copy()
        except Exception:
            pass

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = best_z[i_flat]
    base_y = best_z[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions
