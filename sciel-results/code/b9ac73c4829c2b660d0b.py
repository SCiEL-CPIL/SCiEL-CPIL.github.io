import numpy as np
import time
import random
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 3
    start = time.time()
    deadline = start + 140  # Safety margin under 150s wall-clock

    # Precompute the 18 index triples (U-row, V-row, W-row)
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (9, R)
    # W shape: (p*n, R) = (6, R)
    # Tensor T_{i,j,k} = U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    triples = np.array(triples)  # (18, 3)

    def residuals(x, R):
        U = x[0:6 * R].reshape(6, R)
        V = x[6 * R:6 * R + 9 * R].reshape(9, R)
        W = x[6 * R + 9 * R:6 * R + 15 * R].reshape(6, R)
        U_g = U[triples[:, 0], :]
        V_g = V[triples[:, 1], :]
        W_g = W[triples[:, 2], :]
        return (U_g * V_g * W_g).sum(axis=1) - 1.0

    def compute_s(Uh, Vh, Wh):
        return (Uh[triples[:, 0], :] * Vh[triples[:, 1], :] * Wh[triples[:, 2], :]).sum(axis=1)

    def coord_polish(Uh, Vh, Wh, R, max_iter=100, time_limit=5.0):
        # Local exact search in half-unit integer space; target sum == 8 (since entries are half-integers, 1 = 8/8? No.)
        # Wait, the current code uses Uh = round(U * 2). So U = Uh / 2.
        # Product U*V*W = (Uh/2)*(Vh/2)*(Wh/2) = Uh*Vh*Wh / 8.
        # We want sum(U*V*W) = 1 => sum(Uh*Vh*Wh) = 8.
        s = compute_s(Uh, Vh, Wh).astype(np.int64)
        err = int(((s - 8) ** 2).sum())
        if err == 0:
            return True
        
        mats = [(Uh, 0), (Vh, 1), (Wh, 2)]
        t_start = time.time()
        
        for _ in range(max_iter):
            if time.time() - t_start > time_limit:
                break
            improved = False
            for M, role in mats:
                rows, cols = M.shape
                # Iterate over all entries
                for idx in range(rows * cols):
                    row = idx // cols
                    r = idx % cols
                    mask = triples[:, role] == row
                    if not mask.any():
                        continue
                    t_idx = np.where(mask)[0]
                    
                    if role == 0:
                        c = Vh[triples[t_idx, 1], r] * Wh[triples[t_idx, 2], r]
                    elif role == 1:
                        c = Uh[triples[t_idx, 0], r] * Wh[triples[t_idx, 2], r]
                    else:
                        c = Uh[triples[t_idx, 0], r] * Vh[triples[t_idx, 1], r]
                    c = c.astype(np.int64)
                    
                    base = M[idx]
                    best_delta = 0
                    best_err = err
                    # Try small deltas
                    for d in (-2, -1, 1, 2):
                        s_new = s.copy()
                        s_new[t_idx] += d * c
                        e = int(((s_new - 8) ** 2).sum())
                        if e < best_err:
                            best_err = e
                            best_delta = d
                            
                    if best_delta != 0:
                        M[idx] = base + best_delta
                        s[t_idx] += best_delta * c
                        err = best_err
                        improved = True
                        if err == 0:
                            return True
            if not improved:
                break
        return err == 0

    def to_list(M):
        out = []
        for row in M:
            r2 = []
            for v in row:
                fv = v / 2.0
                if abs(fv - round(fv)) < 1e-9:
                    r2.append(int(round(fv)))
                else:
                    r2.append(fv)
            out.append(r2)
        return out

    def try_rank(R, budget):
        rng = random.Random(R * 7919 + 13)
        size = 21 * R
        end = min(time.time() + budget, deadline)
        best_cost = float('inf')
        attempts = 0
        while time.time() < end:
            attempts += 1
            init = np.array([rng.uniform(-1.0, 1.0) for _ in range(size)])
            try:
                res = least_squares(lambda x: residuals(x, R), init,
                                    method='lm', max_nfev=100)
            except Exception:
                continue
            if not np.all(np.isfinite(res.x)):
                continue
            best_cost = min(best_cost, float(res.cost))
            
            # If cost is very low, try to round and polish
            if res.cost < 1e-6:
                x = res.x
                U = x[0:6 * R].reshape(6, R)
                V = x[6 * R:6 * R + 9 * R].reshape(9, R)
                W = x[6 * R + 9 * R:6 * R + 15 * R].reshape(6, R)
                
                # Round to half-integers
                Uh = np.round(U * 2).astype(np.int64)
                Vh = np.round(V * 2).astype(np.int64)
                Wh = np.round(W * 2).astype(np.int64)
                
                ok = np.all(compute_s(Uh, Vh, Wh) == 8)
                if not ok:
                    ok = coord_polish(Uh, Vh, Wh, R, time_limit=2.0)
                
                if ok:
                    return {"U": to_list(Uh), "V": to_list(Vh), "W": to_list(Wh)}
            
            # Early abort if clearly infeasible
            if attempts > 100 and best_cost > 1e-3:
                break
        return None

    best = None
    best_R = None

    # Try ranks in order of likelihood of feasibility and value
    # Rank 15 and 16 are the most likely candidates for improvement over 18.
    # Rank 14 is very difficult.
    plan = [(15, 30), (16, 30), (14, 20)]
    
    for R, budget in plan:
        if time.time() >= deadline - 5:
            break
        sol = try_rank(R, budget)
        if sol is not None:
            if best is None or R < best_R:
                best, best_R = sol, R
            if R == 14:
                break

    if best is None:
        # Guaranteed fallback: schoolbook rank n*m*p = 18.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    return best
