"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition: Rank = n * m * p
    rank_school = n * m * p
    
    # Build the matrix multiplication tensor T[i,j,k] = delta_{i,k}
    # For multiplying A (n x m) by B (m x p): C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = 1 if this term contributes, else 0
    # Actually the tensor is: T[i,j,k] such that C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, let me re-read the convention.
    
    # The tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i, j, k] = 1 (for the standard multiplication tensor)
    # where i in [0,n), j in [0,m), k in [0,p)
    # And C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In the factorization: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't look right. Let me re-read.
    
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The multiplication is: for each r, we have a term
    # sum_r U[row_U, r] * V[row_V, r] * W[row_W, r]
    # where row_U indexes into A's flattened form, row_V into B's, row_W into C's
    
    # Actually for matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # The tensor decomposition: C[i,k] = sum_r U[i*m + ?, r] * V[?, r] * W[k*n + i, r]
    # Hmm, the indexing is tricky. Let me think about this differently.
    
    # The standard matrix multiplication tensor M_{n,m,p} is defined by:
    # M[i, j, k] = 1 for all i,j,k (it's the "1" tensor in a sense)
    # No wait, that's not right either.
    
    # Let me think of it as: we want to compute C = A @ B where A is n x m, B is m x p
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # In tensor form, this is a trilinear form. The multiplication tensor T has
    # T[i, j, k] = 1 (for the coefficient of A[i,j]*B[j,k] in C[i,k])
    
    # A rank-r decomposition means: T[i,j,k] = sum_{r=1}^r T_r[i,j,k]
    # where each T_r is a rank-1 tensor: T_r[i,j,k] = U_r[i] * V_r[j] * W_r[k]
    
    # But here the factors are indexed differently:
    # U has shape (n*m, rank) - so U[i*m + j, r]
    # V has shape (m*p, rank) - so V[j*p + k, r]  
    # W has shape (p*n, rank) - so W[k*n + i, r]
    
    # And the reconstruction is: C[i,k] = sum_r U[i*m + ?, r] * V[?, r] * W[k*n + i, r]
    # This doesn't quite work because U and V have different index structures.
    
    # Actually, I think the correct interpretation is:
    # The multiplication can be written as:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (some coefficient depending on r)
    # 
    # More precisely, a rank-r decomposition of the multiplication tensor means:
    # For each r in [0, rank), we have coefficients such that:
    # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # 
    # No, that's not right. Let me look at the schoolbook example:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # where col corresponds to a specific (i,j,k) triple.
    
    # So the reconstruction must be:
    # C[i,k] = sum_r U[i*m + j1, r] * V[j2*p + k2, r] * W[k3*n + i3, r] * ...
    
    # Actually, I think the correct formula is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # That doesn't make sense dimensionally.
    
    # Let me reconsider. The standard way to express matrix multiplication as a tensor:
    # The tensor M has entries M[i,j,k] = 1 (all ones) for i in [0,n), j in [0,m), k in [0,p)
    # This is because C[i,k] = sum_j M[i,j,k] * A[i,j] * B[j,k]
    
    # A rank-r decomposition of M means: M[i,j,k] = sum_{r=0}^{r-1} u_r[i,j] * v_r[j,k] * w_r[i,k]
    # Wait, that's not standard either.
    
    # Standard tensor rank decomposition: M[i,j,k] = sum_r a_r[i] * b_r[j] * c_r[k]
    # But here the dimensions are coupled: i and j form the row of A, j and k form the row of B.
    
    # I think the correct interpretation based on the factor shapes is:
    # U has rows indexed by (i,j) pairs from A, so U[i*m+j, r]
    # V has rows indexed by (j,k) pairs from B, so V[j*p+k, r]
    # W has rows indexed by (k,i) pairs from C, so W[k*n+i, r]
    
    # And the multiplication is:
    # C[i,k] = sum_r U[i*m + j_U, r] * V[j_V*p + k_V, r] * W[k_W*n + i_W, r]
    # where the indices must match appropriately.
    
    # Actually, I think the correct formula involves: for each r, the term contributes
    # to C[i,k] as: (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # No, that gives a product of two sums times a scalar, which is quadratic in the inputs.
    
    # Let me look at this from a different angle. The matrix multiplication tensor
    # M_{n,m,p} is an element of R^{n*m} x R^{m*p} x R^{n*p} (or similar).
    
    # Actually, I recall now: the standard formulation is that we view matrix multiplication
    # as a bilinear map. The tensor has rank r if we can write:
    # C[i,k] = sum_{r=0}^{r-1} sum_j A[i,j] * B[j,k] * T_r[i,j,k]
    # where T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[i,k]
    
    # No, that's still not quite right.
    
    # Let me just work with the schoolbook example and generalize.
    # In schoolbook: for each (i,j,k), we have one term with:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    # The reconstruction formula must be such that when we sum over col (rank dimension),
    # we get C[i,k] = sum_j A[i,j] * B[j,k].
    
    # I believe the formula is:
    # C[i,k] = sum_r (U[i*m + j, r]) * (V[j*p + k, r]) * (W[k*n + i, r]) * A[i,j] * B[j,k]
    # But summed over j as well? That would be:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    # For schoolbook: U[i*m+j, col] = 1 only when col corresponds to (i,j,k),
    # V[j*p+k, col] = 1 only when col corresponds to (i,j,k),
    # W[k*n+i, col] = 1 only when col corresponds to (i,j,k).
    
    # So for a fixed (i,k), summing over j and col:
    # C[i,k] = sum_j sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] * A[i,j] * B[j,k]
    # 
    # For col corresponding to (i',j',k'): U[i*m+j, col] = 1 iff i'=i and j'=j
    # V[j*p+k, col] = 1 iff j'=j and k'=k
    # W[k*n+i, col] = 1 iff k'=k and i'=i
    # 
    # So all three are 1 iff i'=i, j'=j, k'=k. Thus:
    # C[i,k] = sum_j A[i,j] * B[j,k] * 1 = sum_j A[i,j] * B[j,k] ✓
    
    # Great! So the formula is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    # This means the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal 1 for all i,j,k (since C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k])
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of n*m*p = 112 equations in 3*rank*(n*m + m*p + p*n) / 3 variables... 
    # Actually, we have rank*(n*m + m*p + p*n) variables total.
    
    # For rank r, we have r*(n*m + m*p + p*n) = r*(16 + 28 + 28) = r*72 variables
    # And n*m*p = 112 constraints.
    
    # So we need r >= ceil(112/72) = 2 (from counting), but the actual minimum rank
    # of the 4x4x4 matrix multiplication tensor is known to be 49 (for 4x4x4).
    # For 4x4x7, it's likely higher.
    
    # Let me implement a search that tries to find low-rank solutions.
    
    def try_decomposition(rank, max_time=10.0):
        """Try to find a rank-r decomposition."""
        start = time.time()
        
        # Number of variables: rank * (n*m + m*p + p*n)
        num_U = n * m
        num_V = m * p
        num_W = p * n
        total_vars = rank * (num_U + num_V + num_W)
        
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # This is a system of n*m*p equations.
        
        # Strategy: use iterative refinement. Start with random initialization,
        # then use gradient descent or least squares to minimize the error.
        
        # Create the linear system: for each (i,j,k), we have:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is nonlinear in the variables (products of three variables).
        # We can linearize by fixing two factors and solving for the third.
        
        # Alternative: use the fact that for fixed r, the term is:
        # U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        # Let's use a simple approach: initialize randomly, then iteratively
        # update each factor to minimize the error.
        
        U = np.random.randn(num_U, rank) * 0.1
        V = np.random.randn(num_V, rank) * 0.1
        W = np.random.randn(num_W, rank) * 0.1
        
        # Scale to make the sum closer to 1
        scale = (num_U * num_V * num_W) ** 0.5
        U /= scale
        V /= scale  
        W /= scale
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(500):
            if time.time() - start > max_time:
                break
            
            # Compute current tensor values
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Update U: for each (i,j), we want U[i*m+j, :] to minimize
            # sum_{k} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
            
            # This is a linear least squares problem for U[i*m+j, :]:
            # Let a_r = V[j*p+k, r] * W[k*n+i, r] for each k
            # Then we want: sum_r U[i*m+j, r] * a_r ≈ 1 for each k
            # So: U[i*m+j, :] @ A = 1 where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            for i in range(n):
                for j in range(m):
                    # Build matrix A of shape (p, rank)
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A[k, r] = V[j*p + k, r] * W[k*n + i, r]
                    
                    # Solve: A @ U[i*m+j, :] = 1 (vector of ones of length p)
                    # Least squares solution
                    if rank <= p:
                        U[i*m+j, :], residuals, rank_ls, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    else:
                        # Underdetermined, use pseudoinverse
                        U[i*m+j, :] = np.linalg.pinv(A) @ np.ones(p)
            
            # Update V: for each (j,k), we want V[j*p+k, :] to minimize
            # sum_{i} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A[i, r] = U[i*m + j, r] * W[k*n + i, r]
                    
                    if rank <= n:
                        V[j*p+k, :], residuals, rank_ls, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    else:
                        V[j*p+k, :] = np.linalg.pinv(A) @ np.ones(n)
            
            # Update W: for each (k,i), we want W[k*n+i, :] to minimize
            # sum_{j} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i*m + j, r] * V[j*p + k, r]
                    
                    if rank <= m:
                        W[k*n+i, :], residuals, rank_ls, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    else:
                        W[k*n+i, :] = np.linalg.pinv(A) @ np.ones(m)
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_error < 1e-10:
                break
        
        if best_error < 1e-8:
            # Convert to lists with half-integer rounding if needed
            U_list = [[float(best_U[i, r]) for r in range(rank)] for i in range(num_U)]
            V_list = [[float(best_V[i, r]) for r in range(rank)] for i in range(num_V)]
            W_list = [[float(best_W[i, r]) for r in range(rank)] for i in range(num_W)]
            
            # Try to round to nearest half-integer
            U_rounded = [[round(2*x)/2 for x in row] for row in U_list]
            V_rounded = [[round(2*x)/2 for x in row] for row in V_list]
            W_rounded = [[round(2*x)/2 for x in row] for row in W_list]
            
            # Check if rounded version still works
            error_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_rounded[i*m+j][r] * V_rounded[j*p+k][r] * W_rounded[k*n+i][r]
                        error_rounded += (val - 1.0) ** 2
            
            if error_rounded < 1e-8:
                return U_rounded, V_rounded, W_rounded, rank
            
            # Otherwise use the unrounded version
            return U_list, V_list, W_list, rank
        
        return None
    
    # Try ranks from low to high
    # Known lower bound: for 4x4x4, min rank is 49. For 4x4x7, likely >= 49.
    # Upper bound: schoolbook = 112
    
    # Start with a reasonable range
    for rank in range(40, 113):
        if time.time() > deadline - 5:
            break
        
        result = try_decomposition(rank, max_time=min(15.0, deadline - time.time() - 2))
        if result is not None:
            U, V, W, r = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

