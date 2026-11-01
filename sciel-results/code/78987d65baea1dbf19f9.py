"""Program for 2x3x7 matrix multiplication tensor decomposition."""
import numpy as np
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # Note: The indices in the problem description are:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Decomposition: sum_r U[:, r] * V[:, r] * W[:, r]^T = T
    # So T is a 3-way tensor.
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # We will try to find a decomposition of rank r.
    # We'll use CP-ALS (Alternating Least Squares).
    
    def matmul_tensor_to_matrix(T, mode):
        """
        Reshape T to a matrix for ALS.
        mode 0: (n*m, m*p*p*n) -> U
        mode 1: (m*p, n*m*p*n) -> V
        mode 2: (p*n, n*m*m*p) -> W
        """
        if mode == 0:
            return T.reshape(n * m, -1)
        elif mode == 1:
            # Move axis 1 to front
            T_perm = np.transpose(T, (1, 0, 2))
            return T_perm.reshape(m * p, -1)
        elif mode == 2:
            # Move axis 2 to front
            T_perm = np.transpose(T, (2, 0, 1))
            return T_perm.reshape(p * n, -1)

    def solve_als(T, U, V, W, mode):
        """
        Solve for one factor matrix given the other two.
        """
        if mode == 0:
            # Solve for U
            # T_0 = U * (V * W^T)^T ?
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Fix V, W.
            # T_0[a, (b, c)] = sum_r U[a, r] (V[b, r] W[c, r])
            # Let M_0 = V * W^T ? No.
            # We need to form the matrix M such that T_0 = U M^T
            # T_0[a, idx] = sum_r U[a, r] M[idx, r]
            # where idx is flattened (b, c).
            # M[idx, r] = V[b, r] W[c, r].
            # This is the Khatri-Rao product of V and W.
            M = np.kron(V, W) # This is not exactly right shape for direct matmul without reshaping T correctly.
            # Let's do it explicitly to be safe.
            # T_0 shape: (nm, mp*pn)
            # M shape: (mp*pn, rank)
            # M[idx, r] = V[b, r] * W[c, r] where idx = b * pn + c
            M = np.zeros((m * p * p * n, U.shape[1]))
            for b in range(m * p):
                for c in range(p * n):
                    idx = b * (p * n) + c
                    M[idx, :] = V[b, :] * W[c, :]
            T_0 = matmul_tensor_to_matrix(T, 0)
            # Solve U M^T = T_0
            # U = T_0 M (M^T M)^-1
            M_T_M = M.T @ M
            M_T_T = M.T @ T_0
            # Regularize
            M_T_M += 1e-6 * np.eye(U.shape[1])
            U_new = np.linalg.solve(M_T_M, M_T_T).T
            return U_new
        elif mode == 1:
            # Solve for V
            # T_1[b, (a, c)] = sum_r V[b, r] (U[a, r] W[c, r])
            # M[idx, r] = U[a, r] W[c, r] where idx = a * pn + c
            M = np.zeros((n * m * p * n, V.shape[1]))
            for a in range(n * m):
                for c in range(p * n):
                    idx = a * (p * n) + c
                    M[idx, :] = U[a, :] * W[c, :]
            T_1 = matmul_tensor_to_matrix(T, 1)
            M_T_M = M.T @ M
            M_T_T = M.T @ T_1
            M_T_M += 1e-6 * np.eye(V.shape[1])
            V_new = np.linalg.solve(M_T_M, M_T_T).T
            return V_new
        elif mode == 2:
            # Solve for W
            # T_2[c, (a, b)] = sum_r W[c, r] (U[a, r] V[b, r])
            # M[idx, r] = U[a, r] V[b, r] where idx = a * mp + b
            M = np.zeros((n * m * m * p, W.shape[1]))
            for a in range(n * m):
                for b in range(m * p):
                    idx = a * (m * p) + b
                    M[idx, :] = U[a, :] * V[b, :]
            T_2 = matmul_tensor_to_matrix(T, 2)
            M_T_M = M.T @ M
            M_T_T = M.T @ T_2
            M_T_M += 1e-6 * np.eye(W.shape[1])
            W_new = np.linalg.solve(M_T_M, M_T_T).T
            return W_new

    def compute_residual(T, U, V, W):
        """
        Compute Frobenius norm of T - sum_r U[:,r] V[:,r] W[:,r]^T
        """
        # Reconstruct
        # R[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is O(n*m * m*p * p*n * rank)
        # For small dims, this is fine.
        R = np.zeros_like(T)
        for r in range(U.shape[1]):
            R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(T - R)

    def round_and_verify(U, V, W):
        """
        Round to half-integers and verify.
        """
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify
        R = np.zeros_like(T)
        for r in range(U_r.shape[1]):
            R += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
        
        if np.allclose(R, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    # Try different ranks
    # Theoretical lower bound is n*p = 14.
    # Let's try ranks from 15 to 30.
    # If we fail, we fall back to the structural decomposition.
    
    best_result = None
    best_rank = float('inf')
    
    # Use random seeds for reproducibility in search, but different starts might help
    # We'll try a few random initializations for each rank.
    
    for r in range(15, 35):
        for attempt in range(3):
            # Initialize U, V, W
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize columns to avoid scaling issues
            for i in range(r):
                norm_u = np.linalg.norm(U[:, i])
                norm_v = np.linalg.norm(V[:, i])
                norm_w = np.linalg.norm(W[:, i])
                if norm_u > 1e-9: U[:, i] /= norm_u
                if norm_v > 1e-9: V[:, i] /= norm_v
                if norm_w > 1e-9: W[:, i] /= norm_w
                
            # ALS iterations
            for it in range(100):
                U = solve_als(T, U, V, W, 0)
                V = solve_als(T, U, V, W, 1)
                W = solve_als(T, U, V, W, 2)
                
                # Optional: Normalize
                for i in range(r):
                    norm_u = np.linalg.norm(U[:, i])
                    norm_v = np.linalg.norm(V[:, i])
                    norm_w = np.linalg.norm(W[:, i])
                    if norm_u > 1e-9: U[:, i] /= norm_u
                    if norm_v > 1e-9: V[:, i] /= norm_v
                    if norm_w > 1e-9: W[:, i] /= norm_w

            res = compute_residual(T, U, V, W)
            if res < 1e-5:
                result = round_and_verify(U, V, W)
                if result is not None:
                    if r < best_rank:
                        best_rank = r
                        best_result = result
                    break # Found a valid one for this rank, try lower rank? 
                         # Actually, we iterate r from low to high, so first found is best.
            else:
                # If residual is small but not exact, maybe rounding failed.
                # Try to snap more aggressively?
                pass
                
        if best_result is not None:
            break # Found the minimal rank in this range

    if best_result is None:
        # Fallback to structural decomposition (Rank 39)
        # We can copy the logic from the previous program here or just return a known valid one.
        # For now, let's construct the Rank 39 one again if numerical fails.
        # To save code size, we might just return the previous program's output if we can't find better.
        # But we must return a valid decomposition.
        
        # Let's hardcode the Rank 39 logic or just use a simple schoolbook (Rank 42) if needed?
        # Schoolbook is always valid.
        U_cols = []
        V_cols = []
        W_cols = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    
                    U_col[i * m + j] = 1.0
                    V_col[j * p + k] = 1.0
                    W_col[k * n + i] = 1.0
                    
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
                    
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        best_result = (U, V, W)
        best_rank = n * m * p

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
