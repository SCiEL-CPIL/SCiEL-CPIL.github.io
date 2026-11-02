import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Strassen rank-7 terms
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Build rank-35 Strassen solution
    rank = 7 * (p // 2)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U[idx, col] = a_vec[idx]
            V[0 * p + 2 * k, col] = b_vec[0]
            V[0 * p + 2 * k + 1, col] = b_vec[1]
            V[1 * p + 2 * k, col] = b_vec[2]
            V[1 * p + 2 * k + 1, col] = b_vec[3]
            W[2 * k * n + 0, col] = c_vec[0]
            W[2 * k * n + 1, col] = c_vec[1]
            W[(2 * k + 1) * n + 0, col] = c_vec[2]
            W[(2 * k + 1) * n + 1, col] = c_vec[3]
            col += 1

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    def verify(U_mat, V_mat, W_mat):
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i * m + k, k * p + j, j * n + i] = 1.0
        result = np.zeros((n * m, m * p, p * n))
        for k in range(U_mat.shape[1]):
            result += np.einsum('a,b,c->abc', U_mat[:, k], V_mat[:, k], W_mat[:, k])
        return np.allclose(result, T, atol=1e-6)

    # Target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    def tensor_norm_diff(U_mat, V_mat, W_mat):
        result = np.zeros((n * m, m * p, p * n))
        for k in range(U_mat.shape[1]):
            result += np.einsum('a,b,c->abc', U_mat[:, k], V_mat[:, k], W_mat[:, k])
        return np.linalg.norm(result - T)

    # Try to find rank 34 solution
    if time.time() < deadline - 20:
        target_rank = 34
        rng = np.random.default_rng(42)
        
        # Initialize with random values
        # Or better, initialize with the rank 35 solution and remove one column
        # Removing a column from a valid decomposition usually increases error,
        # but if we re-optimize, we might converge to a valid rank 34 decomposition if it exists.
        
        best_err = float('inf')
        best_U_34 = None
        best_V_34 = None
        best_W_34 = None

        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            if restart == 0:
                # Use rank 35 solution, drop last column
                U_init = U[:, :target_rank].copy()
                V_init = V[:, :target_rank].copy()
                W_init = W[:, :target_rank].copy()
            else:
                U_init = rng.standard_normal((n * m, target_rank))
                V_init = rng.standard_normal((m * p, target_rank))
                W_init = rng.standard_normal((p * n, target_rank))
            
            # ALS Optimization
            U_cur = U_init
            V_cur = V_init
            W_cur = W_init
            
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                # Minimize || T - sum u_r (v_r x w_r) ||
                # Reshape T to (n*m, m*p*p*n)
                T_mat = T.reshape(n * m, -1)
                
                # Construct the system for U
                # T_mat = U * (V x W)^T
                # (V x W) is a matrix of shape (m*p*p*n, target_rank)
                # Actually, let's use the vectorized form
                # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
                # For fixed V, W, we solve for U.
                # T_{ijk} - sum_{r' != r} U_{ir'} V_{jr'} W_{kr} ... this is complex.
                
                # Simpler ALS step:
                # Update U:
                # For each i, U[i, :] is solution to min || T[i, :, :] - U[i, :] * (V * W) ||
                # Let M = V * W (element-wise product flattened?)
                # T[i] (vector of length m*p*p*n) = U[i] (vector of length R) * M^T?
                # T_{ijk} = sum_r U_{ir} (V_{jr} W_{kr})
                # Let Z_{jr, kr} = V_{jr} W_{kr}.
                # T[i] = sum_r U_{ir} Z_r (where Z_r is a vector of length m*p*p*n)
                # So T_mat (n*m x L) = U (n*m x R) * Z^T (R x L)
                # Z is constructed by outer product of V and W columns?
                # Z[:, r] = np.kron(V[:, r], W[:, r])
                
                Z = np.zeros((m * p * p * n, target_rank))
                for r in range(target_rank):
                    Z[:, r] = np.kron(V_cur[:, r], W_cur[:, r])
                
                # Solve U = T_mat * (Z^T Z)^-1 Z^T T_mat^T
                # Use lstsq for stability
                try:
                    U_cur, _, _, _ = np.linalg.lstsq(Z, T_mat, rcond=None)
                except:
                    break
                
                # Update V
                # T_{ijk} = sum_r V_{jr} (U_{ir} W_{kr})
                # T[j] (vector of length n*m*p*n) = V[j] (vector of length R) * Y^T
                # Y[:, r] = np.kron(U_cur[:, r], W_cur[:, r])
                # Reshape T to (m*p, n*m*p*n)
                T_mat_V = T.reshape(m * p, -1)
                
                Y = np.zeros((n * m * p * n, target_rank))
                for r in range(target_rank):
                    Y[:, r] = np.kron(U_cur[:, r], W_cur[:, r])
                
                try:
                    V_cur, _, _, _ = np.linalg.lstsq(Y, T_mat_V, rcond=None)
                except:
                    break
                    
                # Update W
                # T_{ijk} = sum_r W_{kr} (U_{ir} V_{jr})
                # T[k] (vector of length n*m*m*p) = W[k] (vector of length R) * X^T
                # X[:, r] = np.kron(U_cur[:, r], V_cur[:, r])
                # Reshape T to (p*n, n*m*m*p)
                T_mat_W = T.reshape(p * n, -1)
                
                X = np.zeros((n * m * m * p, target_rank))
                for r in range(target_rank):
                    X[:, r] = np.kron(U_cur[:, r], V_cur[:, r])
                
                try:
                    W_cur, _, _, _ = np.linalg.lstsq(X, T_mat_W, rcond=None)
                except:
                    break

            # Check error
            err = tensor_norm_diff(U_cur, V_cur, W_cur)
            if err < best_err:
                best_err = err
                best_U_34 = U_cur
                best_V_34 = V_cur
                best_W_34 = W_cur
                
            # If error is very small, we found a solution
            if best_err < 1e-6:
                break

        if best_err < 1e-6:
            # Verify again to be sure
            if verify(best_U_34, best_V_34, best_W_34):
                best_U, best_V, best_W = best_U_34, best_V_34, best_W_34
                best_rank = target_rank

    # Fallback to schoolbook if needed
    def schoolbook():
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_sb[i * m + k, col] = 1
                    V_sb[k * p + j, col] = 1
                    W_sb[j * n + i, col] = 1
                    col += 1
        return U_sb, V_sb, W_sb, rank_sb

    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W, best_rank = schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
