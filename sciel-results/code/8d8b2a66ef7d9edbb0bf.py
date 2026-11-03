import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 9
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, the convention is W[k*n+i, r] where i is row of C, k is col of C
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] should be 1 if this is the (i,j,k) term, 0 otherwise
        # Actually, the tensor for matrix multiplication has T[i,j,k] = delta_{i,j,k} in some sense
        # Let me re-read: "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor M_{m,n,p} has entries M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # More precisely, the tensor is in (R^{n*m})^* x (R^{m*p})^* x (R^{p*n})^*
        # M[i,j,k] = 1 for all i,j,k (it's the identity tensor for the bilinear map)
        # No wait. The tensor M satisfies: C = A*B iff for all i,k: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor M is defined by M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: M = sum_r u_r v_r w_r where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        # So M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And we need M[i,j,k] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # First, try to construct the rank-49 decomposition using the mn+mp+np-2 bound
    # The construction: 
    # We have n=2, m=3, p=9
    # mn + mp + np - 2 = 6 + 27 + 18 - 2 = 49
    
    # One standard construction for this bound:
    # The tensor can be written as a sum of:
    # - For each (i,j): a term that captures A[i,j] times all of B[j,:] and projects to row i of C
    # - For each (j,k): a term that captures B[j,k] times all of A[:,j] and projects to col k of C
    # - But we subtract 2 to avoid double-counting at two specific positions
    
    # Let me try a specific construction:
    # Term types:
    # 1. For each i in [0,n), j in [0,m): 
    #    U[i*m+j, r] = 1, V[j*p+k, r] = 1 for all k, W[k*n+i, r] = delta_{k,0}
    #    This gives n*m = 6 terms
    # 2. For each j in [0,m), k in [0,p):
    #    U[i*m+j, r] = delta_{i,0}, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for all i
    #    This gives m*p = 27 terms
    # 3. For each i in [0,n), k in [0,p):
    #    U[i*m+j, r] = delta_{j,0}, V[j*p+k, r] = delta_{j,0}, W[k*n+i, r] = 1
    #    This gives n*p = 18 terms
    # Total: 6 + 27 + 18 = 51
    # We need to remove 2 terms.
    
    # Actually, let me try a different approach. I'll use the fact that:
    # M[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # Let me try to construct it step by step.
    # For the term where i=0, j=0, k=0:
    # We need sum_r U[0, r] V[0, r] W[0, r] = 1
    # For i=0, j=0, k=1:
    # sum_r U[0, r] V[1, r] W[n+0, r] = 1  (W index: k*n+i = 1*2+0 = 2)
    
    # This is complex. Let me try a known algorithm.
    # The algorithm by Landsberg: 
    # The rank of <m,n,p> is at most mn + mp + np - 2.
    # Construction:
    # Let e_i be the standard basis.
    # The tensor = sum_{i=0}^{n-1} sum_{j=0}^{m-1} e_i e_j^T (e_i^T) (sum_{k=0}^{p-1} e_k e_k^T) + ...
    
    # I'll try a direct construction with verification.
    # Start with the naive 54 and try to merge terms.
    
    # For now, let's just return the naive rank-54 decomposition.
    # If we have time, we can try to optimize.
    
    if time.time() > deadline - 5:
        U = [[0] * 54 for _ in range(6)]
        V = [[0] * 54 for _ in range(27)]
        W = [[0] * 54 for _ in range(18)]
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    col = (i * 3 + j) * 9 + k
                    U[i * 3 + j][col] = 1
                    V[j * 9 + k][col] = 1
                    W[k * 2 + i][col] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try rank 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construction for mn+mp+np-2:
    # We use the following pattern:
    # For each i in [0,n), j in [0,m), k in [0,p):
    #   We assign a term index. But we want to share terms.
    #   
    # Alternative: use the "corner" construction.
    # Let me try:
    # For i in [0,n), j in [0,m):
    #   For k in [0,p):
    #     If (i,j,k) is in a "core" set, use a dedicated term.
    #     Otherwise, share with adjacent terms.
    
    # I'll try a specific construction:
    # Terms 0..5: for (i,j) pairs, k=0
    #   U[i*m+j, r] = 1, V[j*p+0, r] = 1, W[0*n+i, r] = 1
    # Terms 6..32: for (j,k) pairs where k>0, i=0
    #   U[0*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+0, r] = 1
    # Terms 33..50: for (i,k) pairs where i>0, j=0
    #   U[i*m+0, r] = 1, V[0*p+k, r] = 1, W[k*n+i, r] = 1
    # This gives 6 + 27 + 18 = 51 terms.
    # But we only have 49 slots.
    
    # Let me try to verify if this 51-term construction works, then see if we can reduce.
    
    # Actually, let me just return the naive 54 for safety.
    U = [[0] * 54 for _ in range(6)]
    V = [[0] * 54 for _ in range(27)]
    W = [[0] * 54 for _ in range(18)]
    
    for i in range(2):
        for j in range(3):
            for k in range(9):
                col = (i * 3 + j) * 9 + k
                U[i * 3 + j][col] = 1
                V[j * 9 + k][col] = 1
                W[k * 2 + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
