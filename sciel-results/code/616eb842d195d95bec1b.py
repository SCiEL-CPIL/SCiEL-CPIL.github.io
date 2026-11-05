"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Build the tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # Actually, the tensor is T[i, j, k] for the multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # In the factorization: C[i,k] = sum_c U[i*m+j? , c] * ... 
    # Let me re-read the convention: W[k*n+i, r]
    # So C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]? 
    # No, that's not right. Let me re-read.
    
    # The decomposition is: (A*B)[i,k] = sum_c U[i*m + j?, c] ... 
    # Actually, the standard form is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j'*p+k, c] * B[j',k]) * W[k*n+i, c]?
    # That doesn't make sense either.
    
    # Let me re-read the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition should satisfy: for all A (n×m), B (m×p), C = A*B (n×p)
    # C[i,k] = sum_{c} sum_{j} U[i*m+j, c] * A[i,j] * V[j*p+k, c] * W[k*n+i, c]?
    # No, that's not standard.
    
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # (A*B)[i,k] = sum_c U_c[i] * V_c[k] * (sum_j A[i,j] * B[j,k] * W_c[j])? 
    # Hmm, let me think differently.
    
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 (for the multiplication tensor).
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]? 
    # But U has shape (n*m, rank), so U[i*m+j, c]. V has shape (m*p, rank), so V[j*p+k, c]. 
    # W has shape (p*n, rank), so W[k*n+i, c].
    
    # So the decomposition is: for all i, j, k:
    # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is the condition for the tensor to be exactly decomposed.
    
    # Let me verify with schoolbook:
    # U[i*m+j, i*m+j*p+k] = 1, V[j*p+k, i*m+j*p+k] = 1, W[k*n+i, i*m+j*p+k] = 1
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 when c = i*m+j*p+k, else 0.
    # Yes, that works.
    
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # And for other combinations, it should be 0? No, the tensor T[i,j,k] = 1 for all valid i,j,k.
    # Wait, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in their ranges.
    # So the condition is just that the sum equals 1 for all valid (i,j,k).
    
    # Let me try to find a lower rank decomposition.
    
    # Strategy: Use a numerical approach
    # 1. Try different ranks
    # 2. For each rank, use alternating least squares
    # 3. Project to half-integers and verify
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def project_to_half_integers(mat):
        """Project a matrix to nearest half-integers."""
        return np.round(np.array(mat) * 2) / 2
    
    # Try to find a decomposition with rank < 63
    best_rank = rank_schoolbook
    best_result = None
    
    # First, try the schoolbook as fallback
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to improve using numerical search
    np.random.seed(42)
    
    for target_rank in range(40, rank_schoolbook):  # Try ranks from 40 to 62
        if time.time() > deadline - 30:
            break
        
        # Initialize with random values
        U_init = np.random.randn(n * m, target_rank)
        V_init = np.random.randn(m * p, target_rank)
        W_init = np.random.randn(p * n, target_rank)
        
        # Alternating least squares
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for iteration in range(50):
            if time.time() > deadline - 30:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                    # This is: U[i*m+j, :] * (V[j*p:0:p, :] * W[0:n:n, :].T) ... 
                    # Actually, for each k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # Let M_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product)
                    # Then we need: U[i*m+j, :] dot M_k = 1 for all k
                    # This is a linear system: M @ U[i*m+j, :]^T = 1
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M @ x = 1
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Project to half-integers
        U_proj = project_to_half_integers(U)
        V_proj = project_to_half_integers(V)
        W_proj = project_to_half_integers(W)
        
        # Verify
        U_list = U_proj.tolist()
        V_list = V_proj.tolist()
        W_list = W_proj.tolist()
        
        if verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
            best_rank = target_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    return best_result
# EVOLVE-BLOCK-END
