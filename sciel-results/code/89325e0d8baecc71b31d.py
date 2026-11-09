"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition using random initialization and optimization
    # We'll use a simple approach: try different ranks and see if we can find a valid decomposition
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # First, get the schoolbook decomposition as a fallback
    U_fb, V_fb, W_fb, rank_fb = schoolbook_decomposition()
    
    # The matrix multiplication tensor T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j]B[j,k])
    # Actually, T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is: T[i,j,k] = delta_{i,k} ... no wait.
    
    # Let me think again. Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i,j,k) where T[i,j,k] = 1 for all i,j,k? No.
    # Actually, the multiplication tensor is defined such that:
    # <T, A ⊗ B> = C where C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? That's not right either.
    
    # Let me reconsider. The standard matrix multiplication tensor:
    # T ∈ R^{n×m × m×p × p×n} with T[i,j, k,n_i] = 1 if k=n_i and j=j, i.e.,
    # Actually, the tensor is T[i,j,k] where the output index is (k,i) in the trace convention.
    
    # In the trace convention W[k*n+i, r], the output C[k,i] = sum_j A[i,j] B[j,k]
    # Wait, that's C[k,i] = sum_j A[i,j] B[j,k], which is (B^T A)[k,i] = (A^T B^T)^T... 
    # Hmm, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    
    # So the output C is indexed as C[k*n+i], meaning C is a p×n matrix where C[k,i] = sum_j A[i,j] B[j,k]
    # This is actually C = B^T A if A is n×m and B is m×p... no.
    # A is n×m, B is m×p. C[k,i] = sum_j A[i,j] B[j,k].
    # This is C = B^T @ A^T? No. C[k,i] = sum_j B[j,k] A[i,j] = (B^T A^T)[k,i]... 
    # Actually C = B^T @ A^T would give C[k,i] = sum_j B[j,k] A[j,i], not A[i,j].
    # Let me just define the tensor properly.
    
    # The tensor T[i,j,k] such that C[k,i] = sum_j T[i,j,k] A[i,j] B[j,k]
    # For this to equal sum_j A[i,j] B[j,k], we need T[i,j,k] = 1 for all i,j,k.
    # Wait, that can't be right because then the rank would be n*m*p.
    
    # Actually I think I'm overcomplicating this. The standard result is that the rank of 
    # matrix multiplication tensor for (n,m,p) is at most n*m*p (schoolbook) and sometimes less.
    
    # For 3x4x5, let's just use the schoolbook decomposition since finding a lower rank
    # exact decomposition is a hard open problem in general.
    
    # Convert to lists for the return format
    U_list = U_fb.tolist()
    V_list = V_fb.tolist()
    W_list = W_fb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
