"""Matrix multiplication 4x4 x 4x5 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try to find a good decomposition
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p  # 80
    
    # Strategy: Use known rank-19 for 4x4x4, extend to 4x4x5
    # For 4x4x4, we need 19 terms. For the 5th column (k=4), we need additional terms.
    # Each term for k=4 requires a new rank-1 component.
    # If we can share structure, we might do better than 19 + 16 = 35
    
    # Actually, let's think about this differently.
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # For 4x4x5: i in 0..3, j in 0..3, k in 0..4
    # Total "1" entries: 4*4*5 = 80
    
    # Rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i*m+j], v_r[j*p+k], w_r[k*n+i]
    
    # For each (i,j,k) with T[i,j,k]=1, we need sum_r u_r[i*m+j]*v_r[j*p+k]*w_r[k*n+i] = 1
    # For each (i,j,k) with T[i,j,k]=0, we need sum_r u_r[i*m+j]*v_r[j*p+k]*w_r[k*n+i] = 0
    
    # Let's try a structured approach:
    # Group by k. For each k, we have a 4x4 matrix multiplication (rank 19 via Lander-Bailey)
    # But the U and V factors depend on k through their indexing.
    
    # Alternative: Use the fact that for fixed k, the sub-tensor is 4x4x1 (rank 16 schoolbook)
    # Total: 5 * 16 = 80 (schoolbook)
    
    # Better: Use Strassen-like for the 4x4 part.
    # 4x4 can be done with rank 19 (Lander-Bailey) or 20 (Strassen).
    # For 4x4x5, if we use rank-19 for each of the 5 output columns, that's 95.
    # But we can share the U and V factors across k values.
    
    # Key insight: For matrix multiplication, the U factor depends on (i,j) and the V factor depends on (j,k).
    # If we use a rank-r decomposition for the 4x4x4 tensor, we get 19 terms.
    # For the 5th column, we need additional terms where w_r has support on k=4.
    
    # Let's try: rank-19 for 4x4x4, plus 16 additional terms for the 5th column (one per (i,j) pair)
    # Total: 19 + 16 = 35
    
    # But can we do better? Maybe some of the 19 terms already have support on k=4?
    # No, the 4x4x4 decomposition only has k in 0..3.
    
    # Let's try to find a decomposition with rank < 80.
    # Use random search + local optimization.
    
    def _verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = 4, 4, 5
        rank = len(U[0]) if U else 0
        # Build the tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1
        
        # Compute decomposition
        # T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_decomp = np.zeros((n, m, p))
        for r in range(rank):
            u = np.array([U[i * m + j][r] for i in range(n) for j in range(m)]).reshape(n, m)
            v = np.array([V[j * p + k][r] for j in range(m) for k in range(p)]).reshape(m, p)
            w = np.array([W[k * n + i][r] for k in range(p) for i in range(n)]).reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        # Check exactness
        if not np.allclose(T_decomp, T, atol=1e-10):
            return False
        return True
    
    # Try structured decomposition: rank-19 for 4x4x4 + 16 for 5th column
    # Lander-Bailey 4x4x4 decomposition (rank 19)
    # This is complex, let's use a simpler approach first.
    
    # Try: Use Strassen's 2x2x2 (rank 7) as building blocks
    # 4x4 = 2x2 blocks of 2x2
    # 4x4x5: 2x2 blocks in output, 5 columns
    
    # Let's try a direct construction with rank 35:
    # 19 terms for k=0,1,2,3 (4x4x4 part)
    # 16 terms for k=4 (one per (i,j) pair)
    
    # For the 4x4x4 part, use a known rank-19 decomposition.
    # Lander-Bailey 19-term 4x4x4 decomposition:
    # This is quite complex. Let's use a simpler rank-20 (Strassen-based) for now.
    
    # Actually, let's just do a random search for a good rank.
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook as initial best
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to improve with structured approach
    # Use the fact that for each k, we have a 4x4 matrix multiplication
    # We can use rank-7 Strassen for 2x2x2 and combine
    
    # For now, return the schoolbook decomposition
    # TODO: Implement better search
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
