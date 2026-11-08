"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n+i, r] convention
    
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook rank
    rank_school = n * m * p  # 64
    
    # Try to find a lower rank decomposition
    # Start with schoolbook and try to reduce
    
    def schoolbook_decomp(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Check: for each (i,k), sum_l A_{i,l} B_{l,k} = sum_c U_{i,l,c} V_{l,k,c} W_{i,k,c}
        # But U, V, W are already indexed correctly
        # The condition is: for all i,l,k, the tensor T[i,l,k] = sum_c U[i*m+l, c] * V[l*p+k, c] * ... 
        # Actually, the decomposition means: C_{i,k} = sum_c (sum_l U[i*m+l,c] * A_{i,l}) * (sum_k' V[l*p+k',c] * B_{l,k'}) * W[k*n+i,c]
        # No, that's not right.
        
        # The correct interpretation: U, V, W define a rank-r decomposition of the multiplication tensor.
        # The tensor T has T[i, l, k] = 1 for the "identity" part, but actually for matrix mult:
        # T is such that <T, A ⊗ B> = sum_{i,k} C_{i,k} where C = A*B
        
        # More precisely, the bilinear form is:
        # sum_{i,k} C_{i,k} * X_{i,k} = sum_{i,k,l} A_{i,l} * B_{l,k} * X_{i,k}
        # = sum_l (sum_i A_{i,l} X_{i,k} for each k) ... 
        
        # Let me think differently. The decomposition U, V, W means:
        # For the multiplication tensor, we have:
        # T = sum_c u_c ⊗ v_c ⊗ w_c
        # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
        # and T[i, l, k] (in appropriate indexing) = sum_c U[i*m+l, c] * V[l*p+k, c] * W[k*n+i, c]
        
        # But T for matrix mult is: T[i, l, k] = 1 if we're summing over l, but actually T is a 3-tensor
        # where T_{(i,l), (l,k), (i,k)} ... this is getting confusing.
        
        # Let me use the standard check: the decomposition is valid if for all A, B:
        # A * B = sum_c (U_c^T * A) * (V_c^T * B) * W_c
        # where U_c is the c-th column of U reshaped, etc.
        
        # Actually, the standard formulation: if we have U ∈ R^{(n*m) × r}, V ∈ R^{(m*p) × r}, W ∈ R^{(p*n) × r},
        # then the decomposition is valid if the tensor T = sum_c u_c ⊗ v_c ⊗ w_c equals the multiplication tensor.
        
        # The multiplication tensor T has T[a, b, c] = 1 if a = (i, l), b = (l, k), c = (k, i) ... no.
        
        # Let me just check numerically: generate random A, B, compute A*B, and check if the decomposition gives the same result.
        
        # Reshape U, V, W
        U_reshaped = U.reshape(n, m, rank)  # U[i, l, c]
        V_reshaped = V.reshape(m, p, rank)  # V[l, k, c]  
        W_reshaped = W.reshape(p, n, rank)  # W[k, i, c]
        
        # For random A (n×m) and B (m×p):
        # C_expected[i, k] = sum_l A[i, l] * B[l, k]
        # C_decomp[i, k] = sum_c (sum_l U_reshaped[i, l, c] * A[i, l]) * (sum_l V_reshaped[l, k, c] * B[l, k]) * W_reshaped[k, i, c]
        # Wait, that's not right either.
        
        # The correct formula: the rank-r decomposition means:
        # C[i, k] = sum_c W[k*n+i, c] * (sum_l U[i*m+l, c] * A[i, l]) * (sum_l V[l*p+k, c] * B[l, k])
        # No, that's also not standard.
        
        # Let me look at the schoolbook: U[i*m+l, c] = 1 when c corresponds to (i,l,k), V[l*p+k, c] = 1, W[k*n+i, c] = 1.
        # Then sum_c U[i*m+l', c] * A[i, l'] * V[l'*p+k', c] * B[l', k'] * W[k'*n+i', c]
        # = sum over c where c=(i,l,k): A[i,l] * B[l,k] * 1
        # = A[i,l] * B[l,k] for the specific (i,l,k)
        # Summing over l: sum_l A[i,l] * B[l,k] = C[i,k]
        
        # So the formula is: C[i,k] = sum_c W[k*n+i, c] * (sum_l U[i*m+l, c] * A[i,l]) * (sum_l V[l*p+k, c] * B[l,k])
        # But wait, in the schoolbook, for a given c=(i0,l0,k0):
        # sum_l U[i*m+l, c] * A[i,l] = A[i, l0] (only non-zero when i=i0, l=l0)
        # sum_l V[l*p+k, c] * B[l,k] = B[l0, k] (only non-zero when l=l0, k=k0)
        # W[k*n+i, c] = 1 (only when k=k0, i=i0)
        # So the product is A[i0, l0] * B[l0, k0] * 1, and it only contributes when i=i0, k=k0.
        # Summing over c: for fixed (i,k), sum over c where c=(i,l,k) for some l: A[i,l] * B[l,k] = C[i,k]. Correct!
        
        # So the verification is:
        # C_decomp[i, k] = sum_c W[k*n+i, c] * (sum_l U[i*m+l, c] * A[i, l]) * (sum_l V[l*p+k, c] * B[l, k])
        
        # Let's test with random matrices
        for _ in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        u_part = sum(U[i*m+l, c] * A[i, l] for l in range(m))
                        v_part = sum(V[l*p+k, c] * B[l, k] for l in range(m))
                        val += W[k*n+i, c] * u_part * v_part
                    C_decomp[i, k] = val
                    if abs(C_decomp[i, k] - C_expected[i, k]) > tol:
                        return False
        return True
    
    # Start with schoolbook
    U, V, W = schoolbook_decomp(rank_school)
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by finding linear dependencies
    # This is complex, so for now, let's try some heuristic reductions
    
    # One simple improvement: if n=2, we can potentially use the fact that we only have 2 rows
    # Try rank = n*p*m/2 or similar heuristics
    
    # For now, return schoolbook as a valid baseline
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
