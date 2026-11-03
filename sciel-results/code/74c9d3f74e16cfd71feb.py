"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i, j, k] = 1 if we need A[i,j]*B[j,k]
    # The tensor for matrix multiplication is T[i, j, k] = 1 for all valid indices
    # Actually, the multiplication tensor is defined such that:
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T has entries T[i, j, k] = 1 for all i, j, k
    # Wait, that's not right. The tensor is a 3-way tensor where T[i, j, k] = 1
    # because the product involves A[i,j] * B[j,k] contributing to C[i,k].
    
    # The tensor T is of shape (n, m, p) with T[i, j, k] = 1 for all i, j, k
    # This is because each term A[i,j]*B[j,k] contributes to C[i,k]
    
    # Actually, let me reconsider. The matrix multiplication tensor T is defined as:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The rank of this tensor is the bilinear complexity of matrix multiplication.
    
    # The decomposition is:
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # where U has shape (n, rank), V has shape (m, rank), W has shape (p, rank)
    
    # Wait, but the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the output factor follows W[k*n+i, r]
    
    # Let me re-read the task. The tensor decomposition is for the multiplication tensor.
    # The standard form is:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # The tensor T has indices (i, j, k) and T[i, j, k] = 1
    # The decomposition is T[i, j, k] = sum_r U[i, m, j, r] * V[j, p, k, r] * W[k, n, i, r]
    
    # Hmm, that doesn't make sense either. Let me think about this differently.
    
    # The factor matrices are:
    # U: shape (n*m, rank) - indexed by (i, j) pairs
    # V: shape (m*p, rank) - indexed by (j, k) pairs  
    # W: shape (p*n, rank) - indexed by (k, i) pairs
    
    # The decomposition should satisfy:
    # For the matrix multiplication tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i, j, k
    
    # This is because the tensor entry T[i, j, k] = 1 (every product term is present)
    
    # So we need to find matrices U, V, W with integer/half-integer entries such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # The schoolbook solution has rank = n*m*p = 3*5*6 = 90
    # Each term corresponds to one (i, j, k) triple
    
    # Let me try to find a lower rank decomposition using ALS
    
    def tensor_residual(U, V, W, rank):
        """Compute the residual tensor T - T_decomposed."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    residual[i, j, k] = 1.0 - val
        return residual
    
    def tensor_residual_fast(U, V, W, rank):
        """Compute the residual tensor using vectorized operations."""
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a tensor contraction
        result = np.zeros((n, m, p))
        for r in range(rank):
            # U[:, :, r] has shape (n, m)
            # V[:, :, r] has shape (m, p)
            # W[:, :, r] has shape (p, n)
            # We need element-wise: for each (i,j,k), U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is an outer product-like operation
            # U[i,j,r] * V[j,k,r] gives a (n, p) matrix for each r, but we need to multiply by W[k,i,r]
            
            # Let's do it as: for each r, compute the contribution
            # contrib[i, j, k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
            # = U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
            
            # We can use einsum: np.einsum('ijr,jkr,kir->ijk', U_reshaped[:,:,r:r+1], V_reshaped[:,:,r:r+1], W_reshaped[:,:,r:r+1])
            # But that's not quite right because the indices are mixed
            
            # Let me just do it with broadcasting
            # U_part: (n, m, 1)
            # V_part: (1, m, p)
            # W_part: (p, 1, n) -> need to transpose to (n, 1, p)? No...
            
            # Actually: U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Let's fix r and compute:
            U_r = U_reshaped[:, :, r]  # (n, m)
            V_r = V_reshaped[:, :, r]  # (m, p)
            W_r = W_reshaped[:, :, r]  # (p, n)
            
            # We want result[i, j, k] = U_r[i, j] * V_r[j, k] * W_r[k, i]
            # This is: for each i, j, k: U_r[i,j] * V_r[j,k] * W_r[k,i]
            
            # We can do: 
            # temp[i, j, k] = U_r[i, j] * V_r[j, k]  -> this is (n, m, p) via broadcasting
            # U_r[:, :, np.newaxis] * V_r[np.newaxis, :, :] gives (n, m, p)
            temp = U_r[:, :, np.newaxis] * V_r[np.newaxis, :, :]  # (n, m, p)
            # Now multiply by W_r[k, i] which is (p, n), so we need W_r.T[i, k] = W_r[k, i]
            # W_r.T has shape (n, p), so W_r.T[np.newaxis, :, :] is (1, n, p)
            # But we need to index it as [i, k] for each (i, j, k)
            # So: temp[i, j, k] * W_r.T[i, k]
            # W_r.T[i, k] can be broadcast as: W_r.T[:, np.newaxis, :] has shape (n, 1, p)
            result += temp * W_r.T[:, np.newaxis, :]
        
        return 1.0 - result
    
    def objective(params, rank):
        """Objective function for optimization."""
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        residual = tensor_residual_fast(U, V, W, rank)
        return np.sum(residual**2)
    
    # Start with the schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    best_score = rank_school
    
    # Try to find lower rank decompositions using ALS
    # We'll try ranks from a lower bound up to the schoolbook rank
    # The theoretical lower bound for 3x5x6 is unknown, but let's try from around 30
    
    for target_rank in range(20, min(rank_school, 50)):
        if time.time() > deadline - 30:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)
        
        # ALS iterations
        for iteration in range(100):
            if time.time() > deadline - 30:
                break
            
            # Update U: for each (i, j), solve least squares
            for i in range(n):
                for j in range(m):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # Fix U[i*m+j, :] and solve for it
                    # The equation is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]) = 1 for all k
                    # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product, shape (rank,))
                    # We need: U[i*m+j, :] @ A_k = 1 for all k
                    # This is a linear system: U[i*m+j, :] @ A_matrix = ones
                    # where A_matrix[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A_matrix = np.zeros((p, target_rank))
                    for k in range(p):
                        A_matrix[k, :] = V_init[j*p + k, :] * W_init[k*n + i, :]
                    
                    # Solve: U_row @ A_matrix.T = ones(p)
                    # U_row is (1, rank), A_matrix.T is (rank, p)
                    # So: U_row @ A_matrix.T = ones
                    # This is underdetermined if rank > p, so use least squares
                    try:
                        U_init[i*m + j, :], residuals, rank_svd, sv = np.linalg.lstsq(A_matrix.T, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V: for each (j, k), solve least squares
            for j in range(m):
                for k in range(p):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                    # Let B_i = U[i*m+j, :] * W[k*n+i, :]
                    # V[j*p+k, :] @ B_matrix.T = ones(n)
                    B_matrix = np.zeros((n, target_rank))
                    for i in range(n):
                        B_matrix[i, :] = U_init[i*m + j, :] * W_init[k*n + i, :]
                    try:
                        V_init[j*p + k, :], residuals, rank_svd, sv = np.linalg.lstsq(B_matrix.T, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W: for each (k, i), solve least squares
            for k in range(p):
                for i in range(n):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    # Let C_j = U[i*m+j, :] * V[j*p+k, :]
                    # W[k*n+i, :] @ C_matrix.T = ones(m)
                    C_matrix = np.zeros((m, target_rank))
                    for j in range(m):
                        C_matrix[j, :] = U_init[i*m + j, :] * V_init[j*p + k, :]
                    try:
                        W_init[k*n + i, :], residuals, rank_svd, sv = np.linalg.lstsq(C_matrix.T, np.ones(m), rcond=None)
                    except:
                        pass
            
            # Check convergence
            if iteration % 10 == 0:
                residual = tensor_residual_fast(U_init, V_init, W_init, target_rank)
                error = np.sum(residual**2)
                if error < 1e-10:
                    # Found a valid decomposition
                    best_U, best_V, best_W, best_rank = U_init, V_init, W_init, target_rank
                    best_score = target_rank
                    break
        
        # After ALS, try to polish with exact solve
        if time.time() > deadline - 30:
            break
        
        # Check if we have a good solution
        residual = tensor_residual_fast(U_init, V_init, W_init, target_rank)
        error = np.sum(residual**2)
        if error < 1e-8:
            best_U, best_V, best_W, best_rank = U_init, V_init, W_init, target_rank
            best_score = target_rank
            break
    
    # Convert to integer/half-integer if possible
    def to_half_integer(matrix):
        """Convert matrix entries to nearest half-integer."""
        result = np.round(matrix * 2) / 2
        return result
    
    U_final = to_half_integer(best_U)
    V_final = to_half_integer(best_V)
    W_final = to_half_integer(best_W)
    
    # Verify the decomposition
    residual = tensor_residual_fast(U_final, V_final, W_final, best_rank)
    error = np.sum(residual**2)
    
    if error > 1e-8:
        # Fall back to schoolbook
        U_final = U_school
        V_final = V_school
        W_final = W_school
        best_rank = rank_school
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
