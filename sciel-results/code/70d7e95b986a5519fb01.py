"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 64
    U_base = [[0] * rank_schoolbook for _ in range(n * m)]
    V_base = [[0] * rank_schoolbook for _ in range(m * p)]
    W_base = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using Strassen-like tricks
    # For 2x2 matrices, Strassen gives rank 7 instead of 8
    # For 2x2 by 2x16, we can potentially use 7 per block of 2 columns
    # But the structure is more complex due to the rectangular nature
    
    # Let's try a direct search for rank < 64
    # We'll use a greedy/optimization approach
    
    # First, let's try rank 56 (7 * 8, using Strassen on pairs of columns)
    # Actually, let me think about this more carefully
    
    # The tensor is T[i,j,k] = A[i,j'] * B[j',k] summed over j'
    # We want to decompose this as sum_c U[i,j,c] * V[j',k,c] * W[i,k,c]
    
    # For small ranks, we can try random initialization + gradient-free optimization
    # or use known tensor rank results
    
    # Known result: For 2x2 matrix multiplication, rank is 7 (Strassen)
    # For 2x2 by 2x2, the tensor rank is 7
    # For 2x2 by 2x16, it's like 8 independent 2x2 multiplications
    # but with shared structure
    
    # Let me try a construction based on Kronecker product of Strassen
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 2x2 by 2x16, we have 8 blocks of 2x2
    # If we could share some of the Strassen terms...
    
    # Actually, let's just do a numerical search for low-rank decomposition
    # using a simplified approach
    
    import numpy as np
    
    # Build the full tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        # Check if close to T
        diff = np.abs(recon - T)
        return np.max(diff) < 1e-9
    
    def try_rank(rank):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline - 10.0:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Use a simple iterative refinement
        # We want to minimize ||T - einsum(U,V,W)||_F^2
        # This is a tensor ring / CP decomposition problem
        
        # Use ALS (Alternating Least Squares)
        max_iter = 200
        tol = 1e-8
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Fix V, W, update U
            # T_reshaped[i, j*k*rank] = sum_{j',k',c} T[i,j',k'] * V[j',c] * W[k',c]
            # We want U[i,c] such that sum_c U[i,c] * (V*W)[i,c] = T[i]
            
            # More carefully:
            # T[i,j',k'] = sum_c U[i,c] * V[j',c] * W[k',c]
            # For fixed i: T[i,:,:] = sum_c U[i,c] * outer(V[:,c], W[:,c]).T
            # So T[i,j',k'] = sum_c U[i,c] * V[j',c] * W[k',c]
            
            # Let M[i,j',k'] = T[i,j',k']
            # For each i, we have M[i] = sum_c U[i,c] * (V[:,c] * W[:,c])
            # where V[:,c] * W[:,c] means element-wise product of column c of V and column c of W
            
            # This is: M[i, j', k'] = sum_c U[i,c] * V[j',c] * W[k',c]
            # Reshape: M[i, j'*p+k'] = sum_c U[i,c] * (V[:,c] * W[:,c])[j'*p+k']
            
            # So M_reshaped[i, :] = U[i, :] @ (V*W)_mat where (V*W)_mat[c, j'*p+k'] = V[j',c]*W[k',c]
            
            # Compute V*W matrix
            VW = V[:, :, None] * W[:, None, :]  # (m*p, p*n, rank)
            VW_mat = VW.transpose(2, 0, 1).reshape(rank, m * p * p * n)  # (rank, m*p*p*n)
            # Wait, this isn't right. Let me redo.
            
            # Actually, let's use a different approach. Use the fact that:
            # T[i, j', k'] = sum_c U[i,c] * V[j',c] * W[k',c]
            
            # For fixed j', k': T[:, j', k'] = U @ diag(V[j',:] * W[k',:])
            # So we can solve for U column by column
            
            # Simpler: use Numpy's lstsq
            # T_3d has shape (n*m, m*p, p*n)
            # We want: T_3d[i, j', k'] = sum_c U[i,c] * V[j',c] * W[k',c]
            
            # Let's reshape T to (n*m, m*p*p*n) by flattening the last two dims
            # T_flat[i, j'*p*n + k'] = T[i, j', k']
            
            # Then T_flat[i, :] = sum_c U[i,c] * (V[:,c] * W[:,c])_flat
            # where (V[:,c] * W[:,c])_flat[j'*p*n + k'] = V[j',c] * W[k',c]
            
            # So T_flat = U @ G where G[c, j'*p*n+k'] = V[j',c] * W[k',c]
            
            T_flat = T.reshape(n * m, m * p * p * n)
            
            # Update U: U = T_flat @ G^T (G @ G^T)^{-1}
            G = np.zeros((rank, m * p * p * n))
            for c in range(rank):
                G[c, :] = (V[:, c][:, None] * W[:, c][None, :]).reshape(-1)
            
            try:
                GtG = G.T @ G
                GtT = G.T @ T_flat.T
                U = np.linalg.solve(GtG + 1e-8 * np.eye(rank), GtT).T
            except np.linalg.LinAlgError:
                U = np.linalg.pinv(GtG) @ GtT
            
            # Update V: For fixed i, k': T[i, :, k'] = V @ diag(U[i,:] * W[k',:])
            # T_flat2[i, j'*p + k'] ... this is getting complex
            
            # Let's use a simpler approach: just return the schoolbook for now
            # and try a few specific constructions
            
            if iteration % 20 == 0:
                recon = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.linalg.norm(recon - T) / np.linalg.norm(T)
                if err < 1e-6:
                    break
        
        # Final check
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < 1e-9:
            return U, V, W
        return None
    
    # Try to find a better rank
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = rank_schoolbook
    
    # Try ranks from 40 to 63
    for rank in range(40, 64):
        if time.time() > deadline - 15.0:
            break
        
        result = try_rank(rank)
        if result is not None:
            U_try, V_try, W_try = result
            if verify_decomposition(U_try, V_try, W_try, rank):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = rank
                break
    
    # Convert to list format
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
