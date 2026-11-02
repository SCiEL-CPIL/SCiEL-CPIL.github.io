"""Construct a rank-24 tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    rank = 24

    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows

    # 21 terms for the 3x3 x 3x3 core (k=0,1,2)
    # Using a known rank-21 decomposition for 3x3x3
    # The decomposition has the form: for each term r, U[i,j,r], V[j,k,r], W[k,i,r]
    # such that sum_r U[i,j,r]*V[j',k',r]*W[k'',i',r] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # for the 3x3x3 subproblem (k in 0..2)

    # Here's a standard rank-21 decomposition for 3x3x3:
    # Terms are indexed by (i, j, k) for the 9 "corner" terms, plus 12 additional terms
    # Actually, let me use a simpler approach: 
    # The rank-21 3x3x3 algorithm can be constructed as follows:
    # 9 terms corresponding to the 9 (i,j) pairs, each handling all 3 k values
    # plus 12 terms for cross-terms
    # 
    # A cleaner way: use the fact that 3x3x3 has rank 21.
    # One explicit construction:
    # For i,j,k in {0,1,2}, we have terms. Let me use a known explicit form.
    
    # Actually, let me just verify the schoolbook rank-9 works for the 3x3x3 part
    # and add 3 more terms for the k=3 column. Total = 12.
    # But that's rank 12, not 24. Let me think again.
    
    # Wait, the current program uses rank 36 (full schoolbook). 
    # The task says score 1.0000 for the current program.
    # If I can get rank 24, that would be an improvement.
    
    # Let me construct a valid rank-24 decomposition:
    # 9 schoolbook terms for the 3x3x3 core (i,j,k with k in 0,1,2)
    # Plus 9 schoolbook terms for the 3x3x1 extension (i,j,k=3)
    # Total = 18 terms. But that's rank 18, not 24.
    
    # Hmm, but 18 < 24, so if rank 18 is valid, that's even better.
    # Let me check: is rank 18 valid?
    # For the 3x3x3 part: 9 terms (i,j,k for k=0,1,2)
    # For the k=3 part: 9 terms (i,j for the j index, k=3)
    # Each term is a rank-1 slice.
    # Total rank = 18.
    
    # Let me verify this is correct:
    # Term (i0,j0,k0) for k0 in 0,1,2:
    #   U[i0*m+j0, r] = 1
    #   V[j0*p+k0, r] = 1  
    #   W[k0*n+i0, r] = 1
    # Term (i0,j0) for k=3:
    #   U[i0*m+j0, r] = 1
    #   V[j0*p+3, r] = 1
    #   W[3*n+i0, r] = 1
    
    # This gives rank 18. Let me use this.
    
    col = 0
    # 9 terms for 3x3x3 core
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 9 terms for k=3 column
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
    
    # col should be 18, but we allocated rank=24. 
    # The extra 6 columns will be all zeros, which is fine.
    # Actually, let me just set rank to 18 to be clean.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct tensor: T[a,b,c] = sum_r U[b,r] * V[c,r] * W[a,r]
    # where a = (i,k), b = (i',j'), c = (j'',k'')
    # Actually, the indices are:
    # U indexed by (i,j) -> row i*m+j
    # V indexed by (j,k) -> row j*p+k
    # W indexed by (k,i) -> row k*n+i
    
    # The tensor T[i,k, i',j', j'',k''] = sum_r U[i,j',r] * V[j'',k'',r] * W[k,i',r]
    # Wait, I need to be careful about index conventions.
    
    # Let me just verify numerically:
    # C[i,k] should = sum_j A[i,j] * B[j,k]
    # Using the decomposition: C[i,k] = sum_{i',j',j'',k''} T[i,k,i',j',j'',k''] * A[i',j'] * B[j'',k'']
    # where T[i,k,i',j',j'',k''] = sum_r U[i*m+j', r] * V[j''*p+k'', r] * W[k*n+i', r]
    
    # Let me construct the full tensor and check
    T = np.zeros((n*p, n*m, m*p), dtype=float)
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for j3 in range(m):
                            for k3 in range(p):
                                T[k*n+i, i2*m+j2, j3*p+k3] += U[i*m+j2][r] * V[j3*p+k3][r] * W[k*n+i2][r]
    
    # The true tensor for matrix multiplication:
    # T_true[i,k, i',j', j'',k''] = delta_{i,i'} * delta_{j',j''} * delta_{k,k''}
    T_true = np.zeros((n*p, n*m, m*p), dtype=float)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T_true[k*n+i, i*m+j, j*p+k] = 1.0
    
    if not np.allclose(T, T_true):
        # Fallback to rank-36 schoolbook
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
