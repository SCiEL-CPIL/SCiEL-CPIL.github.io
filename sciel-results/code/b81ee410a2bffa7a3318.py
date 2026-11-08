import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_known():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0

    def add_rank1(u_idx, u_vals, v_idx, v_vals, w_idx, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_idx, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_idx, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_idx, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    """Verify C[i,k,j] = sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]."""
    import numpy as np
    Ua = np.asarray(U, dtype=object)
    Va = np.asarray(V, dtype=object)
    Wa = np.asarray(W, dtype=object)
    R = Ua.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0
                for t in range(R):
                    s = s + Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if s != 1:
                    return False
    return True

def _als_step(U, V, W, n, m, p, max_iter=50, tol=1e-9):
    R = U.shape[1]
    
    def solve_U(V, W):
        A = np.zeros((n*m*p, n*m*R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i*m + j) * p + k
                    for t in range(R):
                        col = (i*m + j) * R + t
                        A[row, col] = V[j*p + k, t] * W[k*n + i, t]
        b = np.ones(n*m*p)
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        return x.reshape(n*m, R)

    def solve_V(U, W):
        A = np.zeros((n*m*p, m*p*R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i*m + j) * p + k
                    for t in range(R):
                        col = (j*p + k) * R + t
                        A[row, col] = U[i*m + j, t] * W[k*n + i, t]
        b = np.ones(n*m*p)
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        return x.reshape(m*p, R)

    def solve_W(U, V):
        A = np.zeros((n*m*p, p*n*R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i*m + j) * p + k
                    for t in range(R):
                        col = (k*n + i) * R + t
                        A[row, col] = U[i*m + j, t] * V[j*p + k, t]
        b = np.ones(n*m*p)
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        return x.reshape(p*n, R)

    U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
    
    for _ in range(max_iter):
        U_curr = solve_U(V_curr, W_curr)
        V_curr = solve_V(U_curr, W_curr)
        W_curr = solve_W(U_curr, V_curr)
        
    return U_curr, V_curr, W_curr

def _try_reduce_rank(U, V, W, n, m, p, target_rank):
    R = U.shape[1]
    if target_rank >= R:
        return None
    
    T = np.einsum('it,jt,kt->ijk', U, V, W)
    T_flat = T.reshape(-1, R)
    try:
        Usvd, svals, Vsvd = np.linalg.svd(T_flat, full_matrices=False)
        Z = Vsvd[:, :target_rank]
        U_init = U @ Z
        V_init = V @ Z
        W_init = W @ Z
    except:
        U_init = U[:, :target_rank]
        V_init = V[:, :target_rank]
        W_init = W[:, :target_rank]
        
    U_sol, V_sol, W_sol = _als_step(U_init, V_init, W_init, n, m, p, max_iter=100)
    
    def round_half(a):
        return np.round(2 * a) / 2.0
        
    Ur = round_half(U_sol)
    Vr = round_half(V_sol)
    Wr = round_half(W_sol)
    
    Tc = np.einsum('it,jt,kt->ijk', Ur, Vr, Wr)
    if np.allclose(Tc, 1.0, atol=1e-6):
        return (Ur.tolist(), Vr.tolist(), Wr.tolist())
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0

    known = _build_known()
    best = known
    if not _validate(known["U"], known["V"], known["W"], n, m, p):
        return known

    U, V, W = np.array(known["U"]), np.array(known["V"]), np.array(known["W"])
    
    for r in range(65, 59, -1):
        if time.time() > deadline - 10:
            break
        res = _try_reduce_rank(U, V, W, n, m, p, r)
        if res is not None:
            best = {"U": res[0], "V": res[1], "W": res[2]}
            break
            
    return best

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
