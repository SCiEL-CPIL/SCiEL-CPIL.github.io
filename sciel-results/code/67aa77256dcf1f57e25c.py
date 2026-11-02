# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.linalg import lstsq

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
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
        
        # Try integer snapping as well, in case 0.5 is not needed or rounding went wrong
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int
            
        return None

    def als_decompose(T, U, V, W, num_iters=50):
        """
        Alternating Least Squares for Tensor Decomposition.
        U, V, W are initialized.
        """
        rank = U.shape[1]
        
        for _ in range(num_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for c in range(rank):
                # Residual T - sum_{k != c} U_k V_k W_k
                T_res = T.copy()
                for k in range(rank):
                    if k != c:
                        T_res -= np.einsum('i,j,k->ijk', U[:, k], V[:, k], W[:, k])
                
                Vc = V[:, c]
                Wc = W[:, c]
                
                B_row = np.outer(Vc, Wc).flatten()
                norm_B_sq = np.dot(B_row, B_row)
                
                if norm_B_sq < 1e-12:
                    U[:, c] = 0
                    continue
                    
                U[:, c] = (T_res.reshape(U_size, -1) @ B_row) / norm_B_sq
                
            # Update V
            for c in range(rank):
                T_res = T.copy()
                for k in range(rank):
                    if k != c:
                        T_res -= np.einsum('i,j,k->ijk', U[:, k], V[:, k], W[:, k])
                        
                Uc = U[:, c]
                Wc = W[:, c]
                
                B_row = np.outer(Uc, Wc).flatten()
                norm_B_sq = np.dot(B_row, B_row)
                
                if norm_B_sq < 1e-12:
                    V[:, c] = 0
                    continue
                    
                V[:, c] = (T_res.transpose(1, 0, 2).reshape(V_size, -1) @ B_row) / norm_B_sq

            # Update W
            for c in range(rank):
                T_res = T.copy()
                for k in range(rank):
                    if k != c:
                        T_res -= np.einsum('i,j,k->ijk', U[:, k], V[:, k], W[:, k])
                        
                Uc = U[:, c]
                Vc = V[:, c]
                
                B_row = np.outer(Uc, Vc).flatten()
                norm_B_sq = np.dot(B_row, B_row)
                
                if norm_B_sq < 1e-12:
                    W[:, c] = 0
                    continue
                    
                W[:, c] = (T_res.transpose(2, 0, 1).reshape(W_size, -1) @ B_row) / norm_B_sq

        return U, V, W

    def generate_initial_guesses(rank, num_guesses, use_random=True):
        """
        Generate initial guesses.
        Mix of schoolbook-merged and random Gaussian.
        """
        schoolbook_rank = n * m * p
        U_sb = np.zeros((U_size, schoolbook_rank))
        V_sb = np.zeros((V_size, schoolbook_rank))
        W_sb = np.zeros((W_size, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        
        guesses = []
        
        # 1. Random Gaussian guesses
        num_random = num_guesses // 2
        for _ in range(num_random):
            U_new = np.random.randn(U_size, rank)
            V_new = np.random.randn(V_size, rank)
            W_new = np.random.randn(W_size, rank)
            guesses.append((U_new, V_new, W_new))
            
        # 2. Schoolbook merged guesses
        num_merged = num_guesses - num_random
        for _ in range(num_merged):
            if rank >= schoolbook_rank:
                continue
            leaders = np.random.choice(schoolbook_rank, rank, replace=False)
            groups = {l: [l] for l in leaders}
            remaining = [i for i in range(schoolbook_rank) if i not in leaders]
            for idx in remaining:
                leader = np.random.choice(leaders)
                groups[leader].append(idx)
                
            U_new = np.zeros((U_size, rank))
            V_new = np.zeros((V_size, rank))
            W_new = np.zeros((W_size, rank))
            
            for new_col_idx, leader in enumerate(leaders):
                cols = groups[leader]
                U_new[:, new_col_idx] = np.sum(U_sb[:, cols], axis=1)
                V_new[:, new_col_idx] = np.sum(V_sb[:, cols], axis=1)
                W_new[:, new_col_idx] = np.sum(W_sb[:, cols], axis=1)
                
            # Add small random noise
            U_new += np.random.randn(U_size, rank) * 0.1
            V_new += np.random.randn(V_size, rank) * 0.1
            W_new += np.random.randn(W_size, rank) * 0.1
            guesses.append((U_new, V_new, W_new))
            
        return guesses

    # Try ranks from 15 downwards? No, 14 is lower but harder.
    # Known rank is 15. Try 15 first with high effort.
    best_result = get_schoolbook()
    best_rank = 18

    # Prioritize rank 15
    target_ranks = [15, 14, 16, 17]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Allocate more effort to rank 15
        if rank == 15:
            max_restarts = 200
            max_iters = 100
        elif rank == 14:
            max_restarts = 50
            max_iters = 100
        else:
            max_restarts = 50
            max_iters = 50

        guesses = generate_initial_guesses(rank, max_restarts)
        
        for U0, V0, W0 in guesses:
            if time.time() > deadline:
                break
                
            try:
                # Run ALS
                U_opt, V_opt, W_opt = als_decompose(T, U0.copy(), V0.copy(), W0.copy(), num_iters=max_iters)
                
                # Snap and verify
                result = snap_and_verify(U_opt, V_opt, W_opt)
                if result:
                    U_final, V_final, W_final = result
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                    
            except Exception:
                continue

    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
