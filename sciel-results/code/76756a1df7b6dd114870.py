"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition search for 2x3x7 matrix multiplication.
    # Uses the schoolbook decomposition as a safe fallback (rank=42).
    # Attempts to find a lower-rank decomposition via random search with verification.
    
    import numpy as np
    import time
    import random
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Build the multiplication tensor T[i,j,k] for A(2x3) * B(3x7) -> C(2x7)
    # T is indexed by (i, j, k) where i in [0,2), j in [0,3), k in [0,7)
    # The tensor value is 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, the multiplication tensor M_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T[i,j,k] = 1 for all i,j,k (since each A_{i,j}B_{j,k} contributes)
    # Wait, no. The standard tensor for matrix multiplication C = AB is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor M has entries M[i,j,k] = 1 for all valid i,j,k
    # This is a 2 x 3 x 7 tensor with all entries 1.
    # 
    # A rank-r decomposition means we can write:
    # M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]  (or similar factorization)
    # Actually the standard form is:
    # M_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c} for square matrices
    # For rectangular: C[i,k] = sum_j sum_c U[i,j,c] V[j,k,c] ... no.
    #
    # The correct factorization for rectangular matrix multiplication:
    # C = A * B, where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_c U[i,c] * V[k,c] ... no, that's not right either.
    #
    # The standard tensor decomposition for multiplication:
    # T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}? No.
    #
    # Let me think again. The multiplication tensor M_{n,m,p} has entries:
    # M[i, j, k] = delta_{contribution} 
    # For C[i,k] = sum_j A[i,j] B[j,k], the tensor is:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because each term A[i,j]*B[j,k] contributes to C[i,k].
    #
    # A rank-r decomposition is:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]? 
    # No, the factors should be:
    # T[i,j,k] = sum_c U[i, c] * V[j, c] * W[k, c]  (for the case n=m=p)
    # For rectangular, it's:
    # T[i,j,k] = sum_c U[i, j, c] * V[j, k, c] ... this doesn't seem right.
    #
    # Actually, the standard formulation is:
    # We have three vectors of variables: a (length n*m), b (length m*p), c (length n*p)
    # The bilinear form is: c = M(a, b) where c[i,k] = sum_j a[i,j] b[j,k]
    # A rank-r algorithm is:
    # c = sum_c U_c(a) * V_c(b) * W_c
    # where U_c(a) is a linear form in a, V_c(b) is a linear form in b, and W_c is a vector.
    # Equivalently:
    # c[i,k] = sum_c (sum_{i',j} U_c[i',j] a[i',j]) * (sum_{j',k'} V_c[j',k'] b[j',k']) * W_c[i,k]
    # But this must equal sum_j a[i,j] b[j,k].
    #
    # The factor matrices in the task description:
    # U: shape (n*m, rank) - U[row, c] where row = i*m + j
    # V: shape (m*p, rank) - V[row, c] where row = j*p + k  
    # W: shape (p*n, rank) - W[row, c] where row = k*n + i (trace convention)
    #
    # The decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,i} * delta_{k,k} * ... 
    # Actually: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # No wait, the bilinear form is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] A[i,j] B[j,k]
    # This must equal sum_j A[i,j] B[j,k]
    # So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] ... 
    #
    # I think the correct interpretation is:
    # The tensor M has entries M[i,j,k] = 1 for all i,j,k
    # And we want M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # 
    # Wait, that would be a trilinear form, but matrix multiplication is bilinear.
    # 
    # Let me reconsider. The standard formulation for matrix multiplication rank:
    # C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r algorithm computes C using r multiplications:
    # C = sum_c M_c * N_c
    # where M_c and N_c are matrices computed from A and B using additions only.
    # Equivalently:
    # C[i,k] = sum_c (sum_{i',j} U_c[i',j] A[i',j]) (sum_{j',k'} V_c[j',k'] B[j',k'])
    # But this gives a rank-r decomposition where each "multiplication" is of two linear forms.
    #
    # In the factor matrix notation:
    # U[i*m+j, c] = coefficient of A[i,j] in the c-th linear form for the left factor
    # V[j*p+k, c] = coefficient of B[j,k] in the c-th linear form for the right factor
    # W[k*n+i, c] = coefficient for the output C[i,k] from the c-th multiplication
    #
    # Then: C[i,k] = sum_c (sum_{i',j} U[i'*m+j, c] A[i',j]) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])
    # And we need this to equal sum_j A[i,j] B[j,k]
    #
    # So the condition is:
    # sum_c U[i'*m+j, c] V[j'*p+k', c] W[k'*n+i', c] ... no, W multiplies the product.
    #
    # Actually, I think the formula is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j} U[i'*m+j, c] A[i',j]) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_c W[k*n+i, c] U[i'*m+j, c] V[j'*p+k', c] = delta_{i,i'} delta_{j,j'} delta_{k,k'} * ... 
    #
    # This is getting complex. Let me just use the verification approach:
    # Construct U, V, W and verify that the decomposition works.
    #
    # The verification: for random A (n x m) and B (m x p), compute C1 = A @ B
    # and C2[i,k] = sum_c sum_i' sum_j sum_j' sum_k' W[k*n+i, c] * U[i'*m+j, c] * A[i',j] * V[j'*p+k', c] * B[j',k']
    # Check if C1 == C2.
    
    def verify(U, V, W, rank):
        """Verify the decomposition with random test matrices."""
        for _ in range(5):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(rank):
                left = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        left[i, j] = U[i * m + j][c]
                right = np.zeros((m, p))
                for j in range(m):
                    for k in range(p):
                        right[j, k] = V[j * p + k][c]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W[k * n + i][c] * (left @ right)[i, k]
            if not np.allclose(C_expected, C_computed, atol=1e-10):
                return False
        return True
    
    # Schoolbook fallback
    def schoolbook():
        rank = schoolbook_rank
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
    
    deadline = time.time() + 100  # Leave 20s safety margin
    
    # Try to find a lower rank via random search
    # Start from rank 30 and go up to 41
    best_result = schoolbook()
    
    for rank in range(30, min(42, schoolbook_rank)):
        if time.time() > deadline:
            break
        
        max_attempts = 200
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with values from {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if verify(U, V, W, rank):
                found = True
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks if possible
            # Actually, we're going from low to high, so this is the first valid one
            break
    
    return best_result
# EVOLVE-BLOCK-END
