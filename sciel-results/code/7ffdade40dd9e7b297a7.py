import numpy as np
import scipy.optimize as so
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    X0 = keich.copy()
    Y0 = keich.copy()

    def calc_union_length_vec(X, c):
        """Compute union length of intervals [X[i] + c*i/N, X[i] + c*i/N + delta] for i=0..N-1"""
        lefts = X + c * np.arange(N) / N
        rights = lefts + delta
        endpoints = np.empty(2 * N, dtype=np.float64)
        types = np.empty(2 * N, dtype=np.int8)
        endpoints[0::2] = lefts
        endpoints[1::2] = rights
        types[0::2] = 0
        types[1::2] = 1
        order = np.lexsort((types, endpoints))
        sorted_ends = endpoints[order]
        sorted_types = types[order]
        count = 0
        start_pos = 0.0
        total = 0.0
        for val, t in zip(sorted_ends, sorted_types):
            if t == 0:
                if count == 0:
                    start_pos = val
                count += 1
            else:
                count -= 1
                if count == 0:
                    total += val - start_pos
        return total

    def compute_volume(X, Y):
        A_x = np.empty(2 * N)
        B_x = np.empty(2 * N)
        for i in range(N):
            A_x[2*i] = X[i]
            B_x[2*i] = i / N
            A_x[2*i+1] = X[i] + delta
            B_x[2*i+1] = i / N
        
        A_y = np.empty(2 * N)
        B_y = np.empty(2 * N)
        for i in range(N):
            A_y[2*i] = Y[i]
            B_y[2*i] = i / N
            A_y[2*i+1] = Y[i] + delta
            B_y[2*i+1] = i / N
        
        crit_x = [0.0, 1.0]
        for i in range(2 * N):
            for j in range(i + 1, 2 * N):
                if B_x[i] != B_x[j]:
                    c = (A_x[j] - A_x[i]) / (B_x[i] - B_x[j])
                    if 0.0 < c < 1.0:
                        crit_x.append(c)
        
        crit_y = [0.0, 1.0]
        for i in range(2 * N):
            for j in range(i + 1, 2 * N):
                if B_y[i] != B_y[j]:
                    c = (A_y[j] - A_y[i]) / (B_y[i] - B_y[j])
                    if 0.0 < c < 1.0:
                        crit_y.append(c)
        
        all_crit = sorted(set(crit_x + crit_y))
        
        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c_start = all_crit[k]
            c_end = all_crit[k + 1]
            h = c_end - c_start
            if h < 1e-12:
                continue
            
            Lx_s = calc_union_length_vec(X, c_start)
            Lx_e = calc_union_length_vec(X, c_end)
            Ly_s = calc_union_length_vec(Y, c_start)
            Ly_e = calc_union_length_vec(Y, c_end)
            
            dx = Lx_e - Lx_s
            dy = Ly_e - Ly_s
            
            term = Lx_s * Ly_s
            term += 0.5 * (Lx_s * dy + Ly_s * dx)
            term += (1.0 / 3.0) * dx * dy
            
            total_vol += h * term
        
        return total_vol

    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume(X, Y)

    p0 = np.concatenate([X0, Y0])

    start_time = time.time()
    deadline = start_time + 520.0

    best_p = p0.copy()
    best_v = objective(p0)

    np.random.seed(42)
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > deadline - 80:
            break
        if r == 0:
            p_init = p0.copy()
        else:
            scale = 0.05 * (1.0 + 0.5 * np.sin(r * 0.7))
            p_init = p0 + np.random.normal(0, scale, 2 * N)

        try:
            res = so.minimize(objective, p_init, method='Powell',
                              options={'maxiter': 400, 'xtol': 1e-9, 'ftol': 1e-9})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 40:
        try:
            res2 = so.minimize(objective, best_p, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < best_v:
                best_v = res2.fun
                best_p = res2.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 10:
        try:
            res3 = so.minimize(objective, best_p, method='COBYLA',
                               options={'maxiter': 200, 'tol': 1e-12})
            if res3.fun < best_v:
                best_v = res3.fun
                best_p = res3.x.copy()
        except Exception:
            pass

    X_opt = best_p[:N]
    Y_opt = best_p[N:]

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions
