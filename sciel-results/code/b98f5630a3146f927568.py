"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (U_size, V_size, W_size)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit

    def verify(U, V, W):
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def als_decompose(rank, max_iter=100, tol=1e-14):
        U = np.random.randn(U_size, rank) * 0.5
        V = np.random.randn(V_size, rank) * 0.5
        W = np.random.randn(W_size, rank) * 0.5
        
        prev_cost = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                return None

            # Update U
            M_vw = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            T_flat = T.reshape(U_size, V_size * W_size)
            try:
                U_new_T = np.linalg.lstsq(M_vw.T, T_flat.T, rcond=None)[0]
                U_new = U_new_T.T
            except Exception:
                return None

            # Update V
            N_uw = np.einsum('ic,kc->ikc', U_new, W).reshape(U_size * W_size, rank)
            V_new = np.zeros((V_size, rank))
            for j in range(V_size):
                t_j = T[:, j, :].reshape(-1, 1)
                try:
                    V_new[j] = np.linalg.lstsq(N_uw.T, t_j, rcond=None)[0].flatten()
                except Exception:
                    return None

            # Update W
            P_uv = np.einsum('ic,jc->ijc', U_new, V_new).reshape(U_size * V_size, rank)
            W_new = np.zeros((W_size, rank))
            for k in range(W_size):
                t_k = T[:, :, k].reshape(-1, 1)
                try:
                    W_new[k] = np.linalg.lstsq(P_uv.T, t_k, rcond=None)[0].flatten()
                except Exception:
                    return None

            T_calc = np.zeros((U_size, V_size, W_size))
            for c in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U_new[:, c], V_new[:, c], W_new[:, c])
            cost = np.sum((T_calc - T)**2)
            
            if cost < tol:
                return U_new, V_new, W_new
            if cost > prev_cost * (1 + 1e-10):
                break
            prev_cost = cost
            
            U, V, W = U_new, V_new, W_new
            
        return U, V, W

    # Try ranks from 17 down to 10
    for rank in range(17, 9, -1):
        if time.time() > deadline:
            break
            
        max_restarts = 30 if rank <= 16 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart * 7 + rank * 13)
            result = als_decompose(rank)
            
            if result is not None:
                U_opt, V_opt, W_opt = result
                snapped = snap_and_verify(U_opt, V_opt, W_opt)
                if snapped:
                    U_final, V_final, W_final = snapped
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
