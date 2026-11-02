import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Dimensions of factor matrices
    dim_U = n * m  # 9
    dim_V = m * p  # 21
    dim_W = p * n  # 21

    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U_mat, V_mat, W_mat):
        """Verify that U,V,W exactly decompose the multiplication tensor."""
        U_mat = np.asarray(U_mat, dtype=float)
        V_mat = np.asarray(V_mat, dtype=float)
        W_mat = np.asarray(W_mat, dtype=float)
        r = U_mat.shape[1]
        # Target tensor: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val - 1) > 1e-9:
                        return False
        # Check that all other entries are 0
        for a in range(dim_U):
            for b in range(dim_V):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                if j != j2:
                    continue
                for c in range(dim_W):
                    kk = c // n
                    ii = c % n
                    if i == ii and j == j2 and k == kk:
                        continue
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val) > 1e-9:
                        return False
        return True

    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_U, rank_schoolbook), dtype=int)
    V_school = np.zeros((dim_V, rank_schoolbook), dtype=int)
    W_school = np.zeros((dim_W, rank_schoolbook), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                col = i * m * p + j * p + k
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    # Try to find a lower rank decomposition using ALS
    # We search for ranks from 20 to 40
    # If we find a valid one, we return it immediately (since lower is better)
    
    np.random.seed(42)
    
    # To speed up ALS, we can precompute some structures or just use lstsq directly
    # ALS for Tensor Decomposition (CP model)
    # Minimize || T - sum_t u_t v_t^T w_t^T ||_F
    # Update U: For each a, T[:, b, c] is not directly separable easily in standard ALS without unfolding.
    # Unfold T to mode-1: T_(1) is (dim_U, dim_V * dim_W).
    # T_(1)[a, b*dim_W + c] = T[a, b, c]
    # We want T_(1) approx U @ (V ⊙ W)^T where ⊙ is Khatri-Rao product.
    # So, for each a, we solve: T_(1)[a, :] approx U[a, :] @ (V ⊙ W)^T
    # This is a linear system: U[a, :] * ((V ⊙ W)^T) = T_(1)[a, :]
    # Let M = V ⊙ W (shape (dim_V*dim_W, r))
    # Then U[a, :] is the least squares solution to U[a, :] @ M.T = T_(1)[a, :]
    # Actually, U[a, :] is a row vector. Let x = U[a, :].
    # x @ M.T = T_row
    # (M @ x.T) = T_row.T
    # So x.T = lstsq(M, T_row.T)
    
    # Similarly for V and W.
    
    def run_als(rank, iterations=50):
        # Initialize
        U = np.random.randn(dim_U, rank)
        V = np.random.randn(dim_V, rank)
        W = np.random.randn(dim_W, rank)
        
        # Unfold T
        T_U = T.reshape(dim_U, -1) # (dim_U, dim_V*dim_W)
        T_V = T.transpose(1, 0, 2).reshape(dim_V, -1) # (dim_V, dim_U*dim_W)
        T_W = T.transpose(2, 0, 1).reshape(dim_W, -1) # (dim_W, dim_U*dim_V)
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            # Khatri-Rao product of V and W
            # V is (dim_V, rank), W is (dim_W, rank)
            # Kr(V, W) is (dim_V*dim_W, rank)
            # Row i of Kr(V, W) corresponds to V[i//dim_W, :] * W[i%dim_W, :]
            # We can compute this efficiently
            # M_U = np.zeros((dim_V*dim_W, rank))
            # for b in range(dim_V):
            #   for c in range(dim_W):
            #     M_U[b*dim_W+c, :] = V[b, :] * W[c, :]
            
            # Vectorized Khatri-Rao
            # V[:, :, None] * W[None, :, :] -> (dim_V, dim_W, rank)
            # Reshape to (dim_V*dim_W, rank)
            M_U = (V[:, :, None] * W[None, :, :]).reshape(dim_V * dim_W, rank)
            
            # Solve for each row of U
            # U[a, :] . M_U.T = T_U[a, :]
            # M_U @ U[a, :].T = T_U[a, :].T
            # Use lstsq
            try:
                # We can solve all rows at once if we use matrix multiplication
                # But lstsq is per right-hand side.
                # Since dim_U is small (9), we can loop.
                for a in range(dim_U):
                    U[a, :] = np.linalg.lstsq(M_U, T_U[a, :], rcond=None)[0]
            except Exception:
                return None

            # Update V
            M_V = (U[:, :, None] * W[None, :, :]).reshape(dim_U * dim_W, rank)
            try:
                for b in range(dim_V):
                    V[b, :] = np.linalg.lstsq(M_V, T_V[b, :], rcond=None)[0]
            except Exception:
                return None

            # Update W
            M_W = (U[:, :, None] * V[None, :, :]).reshape(dim_U * dim_V, rank)
            try:
                for c in range(dim_W):
                    W[c, :] = np.linalg.lstsq(M_W, T_W[c, :], rcond=None)[0]
            except Exception:
                return None

        return U, V, W

    # Search for lower rank
    # Start from a low rank and go up
    # 23 is the rank for 3x3x3. For 3x3x7, it might be higher.
    # Let's try a range.
    for rank in range(20, 45):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
                
            U, V, W = run_als(rank, iterations=30)
            if U is None:
                continue
                
            # Check if we can snap to half-integers
            # Multiply by 2, round, check residual
            U2 = U * 2
            V2 = V * 2
            W2 = W * 2
            
            U_r = np.round(U2)
            V_r = np.round(V2)
            W_r = np.round(W2)
            
            # Check if the rounded values are close to the original
            if not (np.allclose(U, U_r / 2, atol=1e-6) and 
                    np.allclose(V, V_r / 2, atol=1e-6) and 
                    np.allclose(W, W_r / 2, atol=1e-6)):
                continue
                
            # Verify the exact decomposition with the rounded values
            if verify(U_r / 2, V_r / 2, W_r / 2):
                # Success!
                best_U = U_r / 2
                best_V = V_r / 2
                best_W = W_r / 2
                best_rank = rank
                found = True
                break
                
        if found:
            # We found a valid decomposition of this rank.
            # Since we iterate from low to high, this is the best so far.
            # We can break if we are confident lower ranks are impossible,
            # but for now, we return the first valid one found in the ascending loop.
            # Actually, the loop goes 20, 21, ... so the first valid one is the lowest.
            break

    # Convert to list for output
    def to_list(mat):
        return np.asarray(mat).tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
