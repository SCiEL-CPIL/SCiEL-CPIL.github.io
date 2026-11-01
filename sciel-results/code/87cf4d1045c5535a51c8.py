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

    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        # U, V, W are numpy arrays of shape (size, rank)
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
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def als_decompose(rank, max_iter=50, tol=1e-12):
        # Initialize randomly
        # Scale initialization to help convergence
        U = np.random.randn(U_size, rank) * 0.5
        V = np.random.randn(V_size, rank) * 0.5
        W = np.random.randn(W_size, rank) * 0.5
        
        prev_cost = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                return None

            # Update U
            # T_flat (U, V*W) = U_new (U, rank) @ M_T (rank, V*W)
            # M (V*W, rank) where M[(j,k), c] = V[j,c] * W[k,c]
            # Note: The product V[j,c] * W[k,c] creates a tensor of shape (V_size, W_size, rank)
            # We need to map (j, k) to a single index.
            # T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
            # For fixed i, T[i, j, k] is linear in U[i,c].
            # Let's reshape T to (U_size, V_size*W_size).
            # M_vw (V_size*W_size, rank)
            # M_vw[j*W_size + k, c] = V[j,c] * W[k,c]
            M_vw = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            T_flat = T.reshape(U_size, V_size * W_size)
            
            # Solve: U_new @ M_vw.T = T_flat
            # U_new (U_size, rank), M_vw.T (rank, V_size*W_size), T_flat (U_size, V_size*W_size)
            # This is a linear system in U_new rows? No.
            # U_new[i, :] @ M_vw.T = T_flat[i, :]
            # Let A = M_vw.T (rank, V_size*W_size).
            # U_new[i, :] is a row vector x. x @ A = b (where b = T_flat[i,:]).
            # This is equivalent to A.T @ x.T = b.T.
            # So we solve A.T @ X = B where X is U_new.T (rank, U_size) and B is T_flat.T (V_size*W_size, U_size).
            try:
                # lstsq solves A.T @ X = B
                # A.T is (V_size*W_size, rank)
                # X is (rank, U_size)
                # B is (V_size*W_size, U_size)
                U_new_T = np.linalg.lstsq(M_vw.T, T_flat.T, rcond=None)[0]
                U_new = U_new_T.T
            except Exception:
                return None

            # Update V
            # T[i, j, k] = sum_c V[j,c] (U[i,c] W[k,c])
            # For fixed j, T[:, j, :] (U_size, W_size) is linear in V[j,c].
            # Let N (U_size*W_size, rank) where N[i*W_size + k, c] = U_new[i,c] * W[k,c]
            N_uw = np.einsum('ic,kc->ikc', U_new, W).reshape(U_size * W_size, rank)
            V_new = np.zeros((V_size, rank))
            for j in range(V_size):
                # T[:, j, :] is (U_size, W_size). Flatten to (U_size*W_size, 1)
                t_j = T[:, j, :].reshape(-1, 1)
                # Solve N_uw.T @ v_j.T = t_j.T ?
                # sum_c V[j,c] N_uw[idx, c] = T_flat_j[idx]
                # N_uw.T (rank, U_size*W_size) @ V[j].T (rank, 1) = T_flat_j.T (U_size*W_size, 1)
                try:
                    V_new[j] = np.linalg.lstsq(N_uw.T, t_j, rcond=None)[0].flatten()
                except Exception:
                    return None

            # Update W
            # T[i, j, k] = sum_c W[k,c] (U[i,c] V[j,c])
            # For fixed k, T[:, :, k] (U_size, V_size) is linear in W[k,c].
            # Let P (U_size*V_size, rank) where P[i*V_size + j, c] = U_new[i,c] * V_new[j,c]
            P_uv = np.einsum('ic,jc->ijc', U_new, V_new).reshape(U_size * V_size, rank)
            W_new = np.zeros((W_size, rank))
            for k in range(W_size):
                # T[:, :, k] is (U_size, V_size). Flatten to (U_size*V_size, 1)
                t_k = T[:, :, k].reshape(-1, 1)
                # P_uv.T (rank, U_size*V_size) @ W[k].T (rank, 1) = T_flat_k.T (U_size*V_size, 1)
                try:
                    W_new[k] = np.linalg.lstsq(P_uv.T, t_k, rcond=None)[0].flatten()
                except Exception:
                    return None

            # Calculate cost
            T_calc = np.zeros((U_size, V_size, W_size))
            for c in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U_new[:, c], V_new[:, c], W_new[:, c])
            cost = np.sum((T_calc - T)**2)
            
            if cost < tol:
                return U_new, V_new, W_new
            if cost > prev_cost:
                # Diverging, stop
                break
            prev_cost = cost
            
            U, V, W = U_new, V_new, W_new
            
        return U, V, W

    # Try ranks from 14 upwards
    # Theoretical minimum rank for 2x3x3 multiplication is at least max(n*m, m*p, p*n) = 9?
    # Actually Strassen-like bounds. Schoolbook is 18.
    # Known best for 2x3x3 is likely lower than 18.
    # Let's try a range.
    for rank in range(10, 19):
        if time.time() > deadline:
            break
            
        max_restarts = 20 if rank < 16 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart + rank) # Deterministic for reproducibility in search
            result = als_decompose(rank)
            
            if result is not None:
                U_opt, V_opt, W_opt = result
                snapped = snap_and_verify(U_opt, V_opt, W_opt)
                if snapped:
                    U_final, V_final, W_final = snapped
                    # Convert to list of lists for return
                    U_list = U_final.tolist()
                    V_list = V_final.tolist()
                    W_list = W_final.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
