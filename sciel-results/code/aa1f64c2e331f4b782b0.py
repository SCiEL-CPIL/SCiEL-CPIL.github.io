import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[a, b, c]
    # a indexes A (n x m), b indexes B (m x p), c indexes C with trace convention
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # W index for C[i, k] is k * n + i (trace convention)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i  # trace convention: W[k*n+i, r]
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Determine the tensor rank by examining the matricization
    # For a CP decomposition T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # The matricization T_(1) of shape (n*m, m*p*p*n) has rank equal to the tensor rank
    # (if the decomposition is generic)
    
    T_mat = T.reshape(n * m, m * p * p * n)
    
    # Compute SVD to find the effective rank
    U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
    
    # Find the numerical rank
    tol = 1e-10
    num_rank = np.sum(s_vals > tol * s_vals[0])
    
    print(f"Tensor matricization rank: {num_rank}")
    
    # Try to find a decomposition of rank = num_rank
    # Use the SVD components as initial guess and refine with ALS
    
    def try_decompose(rank, max_iter=300, tol=1e-12):
        """Find rank-r CP decomposition using ALS."""
        np.random.seed(42)
        
        # Initialize using SVD of the matricization
        if rank <= num_rank:
            # Use top-r singular vectors
            U_init = U_svd[:, :rank] * np.sqrt(s_vals[:rank])
            V_init = Vt_svd[:rank, :].T * np.sqrt(s_vals[:rank])
            # Reshape V_init back to (m*p*p*n, rank) and then to (m*p, p*n, rank)
            # Actually, let's just use random init for simplicity and robustness
            pass
        
        # Use structured random initialization
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Fix V and W, solve for U
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each (b,c): T[:,b,c] = U * (V[b,:] * W[c,:])
            # M[b*c + c, r] = V[b,r] * W[c,r]
            M_u = (V.reshape(m * p, 1, rank) * W.reshape(1, p * n, rank)).reshape(m * p * p * n, rank)
            U_new, _, _, _ = np.linalg.lstsq(M_u, T_mat.T, rcond=None)
            U_new = U_new.T
            
            # Fix U and W, solve for V
            T_mat_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            M_v = (U.reshape(n * m, 1, rank) * W.reshape(1, p * n, rank)).reshape(n * m * p * n, rank)
            V_new, _, _, _ = np.linalg.lstsq(M_v, T_mat_v.T, rcond=None)
            V_new = V_new.T
            
            # Fix U and V, solve for W
            T_mat_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            M_w = (U.reshape(n * m, 1, rank) * V.reshape(1, m * p, rank)).reshape(n * m * m * p, rank)
            W_new, _, _, _ = np.linalg.lstsq(M_w, T_mat_w.T, rcond=None)
            W_new = W_new.T
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-15)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
        final_err = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-15)
        
        if final_err < 1e-10:
            return best_U, best_V, best_W, final_err
        return None
    
    # Try ranks from the estimated rank upward
    # The true rank is at most num_rank, but ALS might need a few more iterations
    # Start from num_rank and go up
    
    best_result = None
    best_rank = None
    
    for rank in range(num_rank, num_rank + 10):
        result = try_decompose(rank)
        if result is not None:
            U, V, W, err = result
            best_result = (U, V, W)
            best_rank = rank
            print(f"Found rank-{rank} decomposition with error {err:.2e}")
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Round to half-integers for cleanliness
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify the rounded version
        T_recon = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
        err_rounded = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-15)
        
        if err_rounded < 1e-10:
            U_list = [[int(round(x * 2)) / 2 for x in row] for row in U_rounded]
            V_list = [[int(round(x * 2)) / 2 for x in row] for row in V_rounded]
            W_list = [[int(round(x * 2)) / 2 for x in row] for row in W_rounded]
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Use unrounded values
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: use the known rank-22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    return {"U": U, "V": V, "W": W}

