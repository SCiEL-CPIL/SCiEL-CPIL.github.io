"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

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
                
                # Solve for U[:, c]
                # We want U_c such that sum_i U_c[i] (V_c W_c)^T[i] = T_res ... wait.
                # T_res[i, j, k] = sum_l U_c[l] V_c[j] W_c[k] ? No.
                # The term is U_c[i] V_c[j] W_c[k].
                # So for fixed j, k, T_res[:, j, k] = U_c * (V_c[j] W_c[k]).
                # This is a linear system for U_c.
                # Let M be a matrix where M[l, (j,k)] = V_c[j] W_c[k].
                # Then T_res_flat = M @ U_c.
                
                Vc = V[:, c]
                Wc = W[:, c]
                
                # Construct M
                # M shape: (U_size, V_size * W_size)
                # M[i, j*W_size + k] = Vc[j] * Wc[k]
                M = np.outer(np.ones(U_size), np.outer(Vc, Wc).flatten()).T # This is wrong shape/logic
                # Let's do it properly:
                # We need to solve min || T_res - U_c \otimes V_c \otimes W_c ||
                # This is equivalent to min || T_res_flat - (V_c \otimes W_c) \otimes U_c || ? No.
                # The vectorization of U_c \otimes V_c \otimes W_c is (V_c \otimes W_c) \otimes U_c?
                # Actually, vec(U_c \otimes V_c \otimes W_c) = (W_c \otimes V_c \otimes U_c) ?
                # Let's just use the property:
                # T_res[i,j,k] = U_c[i] * V_c[j] * W_c[k]
                # For a fixed (j,k), the vector over i is U_c * (V_c[j]*W_c[k]).
                # So T_res[:, j, k] = (V_c[j]*W_c[k]) * U_c.
                # This is a rank-1 update to T_res[:, j, k].
                # We can solve for U_c using least squares on all (j,k) pairs.
                
                # Stack all (j,k) residuals
                # Residual vector r of length V_size * W_size * U_size?
                # Let's flatten T_res to shape (U_size, V_size * W_size)
                T_res_2d = T_res.reshape(U_size, V_size * W_size)
                
                # We want U_c such that U_c @ A = T_res_2d, where A is (V_size*W_size, 1)? No.
                # T_res_2d[i, idx(j,k)] = U_c[i] * (V_c[j] W_c[k])
                # So T_res_2d = U_c * B, where B is a row vector?
                # No, B depends on i? No, B[j,k] = V_c[j] W_c[k] is independent of i.
                # So T_res_2d[i, :] = U_c[i] * B
                # This means T_res_2d is a rank-1 matrix: U_c * B.
                # We can solve for U_c and B simultaneously? No, B is fixed by V_c, W_c.
                # So we have T_res_2d = U_c * B.
                # This is a linear system: T_res_2d = U_c B.
                # Since B is a row vector (1 x V_size*W_size), this is underdetermined if U_size < V_size*W_size?
                # U_size=6, V_size*W_size=54.
                # We want to find U_c (6x1) such that U_c B approx T_res_2d.
                # This is equivalent to finding the projection of T_res_2d onto the space spanned by rank-1 matrices with right factor B.
                # Actually, simply:
                # min_U || T_res_2d - U B ||_F
                # Let B_row = np.outer(Vc, Wc).flatten() (shape 54)
                # T_res_2d (6x54)
                # We want U (6x1) such that U B_row approx T_res_2d.
                # This is equivalent to (B_row^T B_row) U^T = T_res_2d B_row^T ?
                # Let's check dimensions.
                # (U B_row)_{i, j} = U_i B_row_j.
                # || T - U B ||^2 = sum_{i,j} (T_{ij} - U_i B_j)^2
                # d/du_i = 2 sum_j (U_i B_j - T_{ij}) B_j = 2 (U_i ||B||^2 - sum_j T_{ij} B_j)
                # So U_i = (sum_j T_{ij} B_j) / ||B||^2.
                
                B_row = np.outer(Vc, Wc).flatten()
                norm_B_sq = np.dot(B_row, B_row)
                
                if norm_B_sq < 1e-12:
                    # If B is zero, U is not identifiable, leave as is or set to 0
                    U[:, c] = 0
                    continue
                    
                # U_i = sum_j T_res_2d[i,j] B_j / norm_B_sq
                U[:, c] = (T_res_2d @ B_row) / norm_B_sq
                
            # Update V
            for c in range(rank):
                T_res = T.copy()
                for k in range(rank):
                    if k != c:
                        T_res -= np.einsum('i,j,k->ijk', U[:, k], V[:, k], W[:, k])
                        
                Uc = U[:, c]
                Wc = W[:, c]
                
                # T_res[i,j,k] = V_c[j] * (U_c[i] W_c[k])
                # For fixed (i,k), T_res[i,:,k] = V_c * (Uc[i] Wc[k])
                # Reshape T_res to (V_size, U_size * W_size)
                T_res_2d = T_res.transpose(1, 0, 2).reshape(V_size, U_size * W_size)
                
                B_row = np.outer(Uc, Wc).flatten()
                norm_B_sq = np.dot(B_row, B_row)
                
                if norm_B_sq < 1e-12:
                    V[:, c] = 0
                    continue
                    
                V[:, c] = (T_res_2d @ B_row) / norm_B_sq

            # Update W
            for c in range(rank):
                T_res = T.copy()
                for k in range(rank):
                    if k != c:
                        T_res -= np.einsum('i,j,k->ijk', U[:, k], V[:, k], W[:, k])
                        
                Uc = U[:, c]
                Vc = V[:, c]
                
                # T_res[i,j,k] = W_c[k] * (Uc[i] Vc[j])
                # For fixed (i,j), T_res[i,j,:] = W_c * (Uc[i] Vc[j])
                # Reshape T_res to (W_size, U_size * V_size)
                T_res_2d = T_res.transpose(2, 0, 1).reshape(W_size, U_size * V_size)
                
                B_row = np.outer(Uc, Vc).flatten()
                norm_B_sq = np.dot(B_row, B_row)
                
                if norm_B_sq < 1e-12:
                    W[:, c] = 0
                    continue
                    
                W[:, c] = (T_res_2d @ B_row) / norm_B_sq

        return U, V, W

    def generate_initial_guesses(rank, num_guesses):
        """
        Generate initial guesses by merging columns of the schoolbook decomposition.
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
        if rank >= schoolbook_rank:
            return guesses
            
        for _ in range(num_guesses):
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
                
            # Add small random noise to break symmetry
            U_new += np.random.randn(U_size, rank) * 0.1
            V_new += np.random.randn(V_size, rank) * 0.1
            W_new += np.random.randn(W_size, rank) * 0.1
            
            guesses.append((U_new, V_new, W_new))
            
        return guesses

    # Try ranks from 14 upwards
    # Theoretical lower bound is around 13-14. Schoolbook is 18.
    best_result = get_schoolbook()
    best_rank = 18

    for rank in range(14, 18):
        if time.time() > deadline:
            break
            
        # Number of restarts
        max_restarts = 20 if rank < 17 else 5
        
        guesses = generate_initial_guesses(rank, max_restarts)
        
        for U0, V0, W0 in guesses:
            if time.time() > deadline:
                break
                
            try:
                # Run ALS
                U_opt, V_opt, W_opt = als_decompose(T, U0.copy(), V0.copy(), W0.copy(), num_iters=30)
                
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
