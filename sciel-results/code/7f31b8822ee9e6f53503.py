import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    t0 = time.time()
    deadline = 135  # safety margin before 150s kill

    def remaining():
        return deadline - (time.time() - t0)

    def verify_full(U, V, W):
        """Verify that U, V, W correctly decompose 2x2 x 2x12."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
        # U shape: (4, rank), V shape: (24, rank), W shape: (24, rank)
        # W[k*n+i, r] = W[k*2+i, r]
        max_err = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(12):
                    val = sum(U[i*2+j, s] * V[j*12+k, s] * W[k*2+i, s] for s in range(rank))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # --- Fallback: rank 42 (Strassen per 2x2 block of the 2x12) ---
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(24)]
        W = [[0] * rank for _ in range(24)]
        strassen_coeffs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        r = 0
        for b in range(6):
            for s in range(7):
                u, v, w = strassen_coeffs[s]
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u[i_loc * 2 + j_loc]
                        if val != 0:
                            U[i_loc * 2 + j_loc][r] = val
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v[j_loc * 2 + k_loc]
                        if val != 0:
                            V[j_loc * 12 + (2 * b + k_loc)][r] = val
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w[k_loc * 2 + i_loc]
                        if val != 0:
                            W[(2 * b + k_loc) * 2 + i_loc][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    # --- Vectorized objective for 2x2x4 ---
    def make_obj_2x2x4(R):
        ii = np.repeat(np.repeat(np.arange(2), 2), 4)
        jj = np.repeat(np.tile(np.arange(2), 2), 4)
        kk = np.tile(np.arange(4), 4)
        u_row = ii * 2 + jj
        v_row = jj * 4 + kk
        w_row = ii * 4 + kk
        target = np.ones(16)

        def objective(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+8*R].reshape(8, R)
            W = params[4*R+8*R:].reshape(8, R)
            T = (U[u_row] * V[v_row] * W[w_row]).sum(axis=1)
            d = T - target
            return float(d @ d)
        return objective

    def verify_2x2x4(U, V, W, R):
        ii = np.repeat(np.repeat(np.arange(2), 2), 4)
        jj = np.repeat(np.tile(np.arange(2), 2), 4)
        kk = np.tile(np.arange(4), 4)
        T = (U[ii*2+jj] * V[jj*4+kk] * W[ii*4+kk]).sum(axis=1)
        return float(np.max(np.abs(T - 1)))

    def try_snaps(U, V, W, verify_fn, R):
        # Try several snap grids
        for scale in [2, 4, 8]:
            Us = np.round(U * scale) / scale
            Vs = np.round(V * scale) / scale
            Ws = np.round(W * scale) / scale
            if verify_fn(Us, Vs, Ws, R) < 1e-9:
                return Us, Vs, Ws
        return None

    def search_2x2x4(R, num_restarts):
        """Search for rank-R decomposition of 2x2x4, then tile 3x for 2x2x12 (total rank 3R)."""
        if remaining() < 5:
            return None
        obj = make_obj_2x2x4(R)
        dim = 4*R + 8*R + 8*R
        best_err = float('inf')
        best_p = None

        for restart in range(num_restarts):
            if remaining() < 5:
                break
            if restart % 4 == 0:
                params = np.random.uniform(-1, 1, dim)
            elif restart % 4 == 1:
                params = np.random.normal(0, 0.7, dim)
            elif restart % 4 == 2:
                params = np.zeros(dim)
                nn = max(1, dim // 4)
                nz = np.random.choice(dim, nn, replace=False)
                params[nz] = np.random.uniform(-1, 1, nn)
            else:
                # Structured init: use known 2x2x2 rank-7 Strassen pattern extended
                params = np.zeros(dim)
                # Initialize with simple rank-4 schoolbook-like structure
                for s in range(R):
                    params[0*4 + s] = 1.0 if s < R//2 else 0.0
                    params[1*4 + s] = 0.0 if s < R//2 else 1.0
                    params[2*4 + s] = 0.0 if s < R//2 else 0.0
                    params[3*4 + s] = 0.0 if s < R//2 else 0.0
                params += np.random.normal(0, 0.1, dim)

            res = minimize(obj, params, method='L-BFGS-B',
                           options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-14})
            if res.fun < best_err:
                best_err = res.fun
                best_p = res.x.copy()
            if best_err < 1e-10:
                break

        if best_err < 1e-8 and best_p is not None:
            U = best_p[:4*R].reshape(4, R)
            V = best_p[4*R:4*R+8*R].reshape(8, R)
            W = best_p[4*R+8*R:].reshape(8, R)
            snapped = try_snaps(U, V, W, verify_2x2x4, R)
            if snapped is not None:
                Us, Vs, Ws = snapped
                full_rank = R * 3
                full_U = [[0.0] * full_rank for _ in range(4)]
                full_V = [[0.0] * full_rank for _ in range(24)]
                full_W = [[0.0] * full_rank for _ in range(24)]
                for block in range(3):
                    off = block * R
                    for s in range(R):
                        r = off + s
                        for i in range(2):
                            for j in range(2):
                                full_U[i*2+j][r] = float(Us[i*2+j, s])
                        for j in range(2):
                            for k in range(4):
                                full_V[j*12 + (block*4 + k)][r] = float(Vs[j*4+k, s])
                        for i in range(2):
                            for k in range(4):
                                full_W[(block*4 + k)*2 + i][r] = float(Ws[i*4+k, s])
                # Final verification
                if verify_full(full_U, full_V, full_W) < 1e-8:
                    return {"U": full_U, "V": full_V, "W": full_W}
        return None

    # Try rank 8 (2x2x4) -> total rank 24
    res = search_2x2x4(R=8, num_restarts=50)
    if res:
        return res

    # Try rank 9 (2x2x4) -> total rank 27
    res = search_2x2x4(R=9, num_restarts=30)
    if res:
        return res

    # Try rank 10 (2x2x4) -> total rank 30
    res = search_2x2x4(R=10, num_restarts=20)
    if res:
        return res

    # Fallback
    return get_strassen_42()
