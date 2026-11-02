import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (n*m, m*p, p*n)
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
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def verify(U, V, W):
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5 (allowing integer and half-integer entries)
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def local_grid_search(U, V, W, max_iters=100):
        """
        Given a snapped candidate, try to improve it by flipping signs or adding/subtracting 0.5
        to entries that reduce the error.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Calculate initial error
        def calc_error():
            T_calc = np.zeros((U_size, V_size, W_size))
            for c in range(U.shape[1]):
                T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            return np.sum((T_calc - T)**2)
        
        current_error = calc_error()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try flipping signs or adjusting U
            for i in range(U_size):
                for c in range(U.shape[1]):
                    orig_val = U[i, c]
                    # Try +/- 0.5
                    for delta in [-0.5, 0.5]:
                        if orig_val + delta in [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]: # Simple check
                            U[i, c] = orig_val + delta
                            new_error = calc_error()
                            if new_error < current_error - 1e-12:
                                current_error = new_error
                                improved = True
                            else:
                                U[i, c] = orig_val
                    # Try flipping sign
                    if orig_val != 0:
                        U[i, c] = -orig_val
                        new_error = calc_error()
                        if new_error < current_error - 1e-12:
                            current_error = new_error
                            improved = True
                        else:
                            U[i, c] = orig_val

            # Try flipping signs or adjusting V
            for i in range(V_size):
                for c in range(V.shape[1]):
                    orig_val = V[i, c]
                    for delta in [-0.5, 0.5]:
                        if orig_val + delta in [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]:
                            V[i, c] = orig_val + delta
                            new_error = calc_error()
                            if new_error < current_error - 1e-12:
                                current_error = new_error
                                improved = True
                            else:
                                V[i, c] = orig_val
                    if orig_val != 0:
                        V[i, c] = -orig_val
                        new_error = calc_error()
                        if new_error < current_error - 1e-12:
                            current_error = new_error
                            improved = True
                        else:
                            V[i, c] = orig_val

            # Try flipping signs or adjusting W
            for i in range(W_size):
                for c in range(W.shape[1]):
                    orig_val = W[i, c]
                    for delta in [-0.5, 0.5]:
                        if orig_val + delta in [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]:
                            W[i, c] = orig_val + delta
                            new_error = calc_error()
                            if new_error < current_error - 1e-12:
                                current_error = new_error
                                improved = True
                            else:
                                W[i, c] = orig_val
                    if orig_val != 0:
                        W[i, c] = -orig_val
                        new_error = calc_error()
                        if new_error < current_error - 1e-12:
                            current_error = new_error
                            improved = True
                        else:
                            W[i, c] = orig_val

            if not improved:
                break

        if current_error < 1e-9:
            return U, V, W
        return None

    def als_decompose(rank, max_iter=100, tol=1e-12, use_grid_init=True):
        # Initialize
        if use_grid_init:
            # Sparse integer initialization
            U = np.random.choice([-1, 0, 1], size=(U_size, rank))
            V = np.random.choice([-1, 0, 1], size=(V_size, rank))
            W = np.random.choice([-1, 0, 1], size=(W_size, rank))
        else:
            U = np.random.randn(U_size, rank)
            V = np.random.randn(V_size, rank)
            W = np.random.randn(W_size, rank)
        
        prev_cost = np.inf
        for it in range(max_iter):
            # Update U
            M = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            try:
                U_new = np.linalg.lstsq(M, T.reshape(U_size, -1).T, rcond=None)[0].T
            except:
                return None

            # Update V
            N = np.einsum('ic,kc->ikc', U_new, W).reshape(U_size * W_size, rank)
            V_new = np.zeros((V_size, rank))
            for j in range(V_size):
                t_j = T[:, j, :].reshape(-1, 1)
                try:
                    V_new[j] = np.linalg.lstsq(N, t_j, rcond=None)[0].flatten()
                except:
                    return None

            # Update W
            P = np.einsum('ic,jc->ijc', U_new, V_new).reshape(U_size * V_size, rank)
            W_new = np.zeros((W_size, rank))
            for k in range(W_size):
                t_k = T[:, :, k].reshape(-1, 1)
                try:
                    W_new[k] = np.linalg.lstsq(P, t_k, rcond=None)[0].flatten()
                except:
                    return None

            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            cost = 0
            for c in range(rank):
                cost += np.sum((np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]) - T)**2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
                
        return U, V, W

    def polish(U, V, W, rank, max_iter=20, tol=1e-14):
        """
        High-precision polish pass.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            # Update U
            M = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            try:
                U_new = np.linalg.lstsq(M, T.reshape(U_size, -1).T, rcond=None)[0].T
            except:
                break

            # Update V
            N = np.einsum('ic,kc->ikc', U_new, W).reshape(U_size * W_size, rank)
            V_new = np.zeros((V_size, rank))
            for j in range(V_size):
                t_j = T[:, j, :].reshape(-1, 1)
                try:
                    V_new[j] = np.linalg.lstsq(N, t_j, rcond=None)[0].flatten()
                except:
                    break

            # Update W
            P = np.einsum('ic,jc->ijc', U_new, V_new).reshape(U_size * V_size, rank)
            W_new = np.zeros((W_size, rank))
            for k in range(W_size):
                t_k = T[:, :, k].reshape(-1, 1)
                try:
                    W_new[k] = np.linalg.lstsq(P, t_k, rcond=None)[0].flatten()
                except:
                    break
            
            U, V, W = U_new, V_new, W_new
            
            cost = 0
            for c in range(rank):
                cost += np.sum((np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]) - T)**2)
            
            if cost < tol:
                break
                
        return U, V, W

    # Try ranks from 14 upwards
    # Rank 14 is the known optimal rank for 2x3x3
    best_rank = n * m * p
    best_decomp = None

    # Prioritize rank 14
    for rank in [14, 13, 15, 16, 17]:
        if time.time() > deadline:
            break
            
        # More restarts for lower ranks
        max_restarts = 50 if rank == 14 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Use a deterministic seed based on rank and restart for reproducibility
            np.random.seed(rank * 100 + restart)
            
            U, V, W = als_decompose(rank, use_grid_init=True)
            if U is None:
                continue

            # Polish
            U, V, W = polish(U, V, W, rank)
            
            # Snap and Verify
            result = snap_and_verify(U, V, W)
            if result:
                U_final, V_final, W_final = result
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            
            # If snapping failed, try local grid search on the snapped version
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            refined = local_grid_search(U_snap, V_snap, W_snap)
            if refined:
                U_final, V_final, W_final = refined
                if verify(U_final, V_final, W_final):
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    # Fallback to schoolbook
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
