"""Direct approach: exploit the structure of 2x3 * 3x6 matrix multiplication."""
import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Convention:
    - U: shape (n*m, rank) = (6, r), U[i*m+j, t]
    - V: shape (m*p, rank) = (18, r), V[j*p+k, t]
    - W: shape (p*n, rank) = (12, r), W[k*n+i, t]
    
    Constraint: for all i in [0,n), j in [0,m), k in [0,p):
      sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    """
    start_time = time.time()
    timeout = 120
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy: Partition B (3x6) into two 3x3 blocks B1 (cols 0-2) and B2 (cols 3-5)
    # Find a rank-9 decomposition for 2x3 * 3x3, then replicate for each block.
    # Total rank = 9 + 9 = 18.
    
    # For 2x3 * 3x3:
    # A (2x3): [[a0, a1, a2], [b0, b1, b2]]
    # B (3x3): [[c0, c1, c2], [d0, d1, d2], [e0, e1, e2]]
    # C (2x3):
    #   C[0,0] = a0*c0 + a1*d0 + a2*e0
    #   C[0,1] = a0*c1 + a1*d1 + a2*e1
    #   C[0,2] = a0*c2 + a1*d2 + a2*e2
    #   C[1,0] = b0*c0 + b1*d0 + b2*e0
    #   C[1,1] = b0*c1 + b1*d1 + b2*e1
    #   C[1,2] = b0*c2 + b1*d2 + b2*e2
    
    # Schoolbook rank for 2x3*3x3 = 18. Lower bound = max(6,9,6) = 9.
    # Let's try to construct rank 9.
    
    # Approach: Use the structure that C[:,k] = A * B[:,k] for k=0,1,2
    # Each column of C is a 2x1 vector = A (2x3) times B_col (3x1)
    
    # For rank 9, we need 9 bilinear terms. Let me try a construction:
    # 
    # Key idea: For each column k of B, we compute A * B[:,k] (2x1 result).
    # This is a 2x3 times 3x1 multiplication. The schoolbook rank for this is 2*3*1=6.
    # But we can share structure across columns.
    
    # Let me try: define 9 terms where each term involves products of linear
    # combinations of A entries and B entries.
    
    # Construction for 2x3 * 3x3 with rank 9:
    # 
    # Let me use the following approach. For each of the 3 "inner" indices j,
    # and for each of the 3 output columns k, the term A[i,j]*B[j,k] contributes
    # to C[i,k]. 
    #
    # To get rank 9, I'll try:
    # Terms 1-3: handle j=0 for all (i,k)
    # Terms 4-6: handle j=1 for all (i,k)  
    # Terms 7-9: handle j=2 for all (i,k)
    #
    # But that's still 18 terms (2 i-values * 3 k-values * 3 j-values = 18).
    #
    # Alternative: For each j, use 3 terms to handle all 6 (i,k) pairs.
    # For fixed j, C[i,k] gets contribution A[i,j]*B[j,k].
    # If I set U[(i,j),t] = A[i,j] coefficient and V[(j,k),t] = B[j,k] coefficient,
    # then for fixed j, I need: sum over t for this j of U[i,j]*V[j,k]*W[k*n+i] = A[i,j]*B[j,k] contribution
    #
    # Actually, let me think differently. 
    #
    # For a rank-1 term t: it contributes U[i,j]*V[j,k]*W[k*n+i] to the (i,j,k) entry.
    # We need this sum over all t to equal 1 for every valid (i,j,k).
    #
    # Let me try a concrete rank-9 construction for 2x3*3x3:
    # 
    # Term t (t=0,...,8): Let t = j*3 + k where j in {0,1,2}, k in {0,1,2}
    # For term (j,k):
    #   U[i*m+j, t] = 1 for all i (i=0,1)  -- wait, this doesn't work
    #
    # Let me try: For each j (0,1,2), use 3 terms indexed by k (0,1,2).
    # Term (j,k):
    #   U[i*3+j, (j,k)] = 1 for i=0,1
    #   V[j*3+k, (j,k)] = 1
    #   W[k*2+i, (j,k)] = 1 for i=0,1
    #
    # Check: sum_t U[i*3+j', t] * V[j'*3+k', t] * W[k'*2+i, t]
    # For t=(j,k): U[i*3+j, (j,k)] = 1 if j'=j, V[j*3+k, (j,k)] = 1 if k'=k
    # So U[i*3+j', t]*V[j'*3+k', t] = 1 iff j'=j and k'=k
    # Then W[k*2+i, t] = 1
    # So the sum = 1 iff j'=j and k'=k, which gives [j'=j, k'=k]
    # But we need the sum to be 1 for ALL (i,j,k), not just when j'=j and k'=k.
    #
    # Wait, I'm confusing myself. The constraint is:
    # For each (i,j,k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    #
    # So for a fixed (i,j,k), we sum over all t. Let me re-examine.
    #
    # With my construction for 2x3*3x3 (n=2, m=3, p=3):
    # U has shape (6, 9), V has shape (9, 9), W has shape (6, 9)
    # 
    # Term t = j*3+k (j=0,1,2; k=0,1,2):
    #   U[i*3+j, t] = 1 for i=0,1
    #   V[j*3+k, t] = 1
    #   W[k*2+i, t] = 1 for i=0,1
    #
    # Now check constraint for (i,j,k):
    # sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*2+i, t]
    # 
    # For t = j'*3+k':
    #   U[i*3+j, t] = 1 iff j = j'
    #   V[j*3+k, t] = 1 iff k = k'
    #   W[k*2+i, t] = 1 (always, since W[k*2+i, t] = 1 for all i)
    #
    # So the product is 1 iff j=j' AND k=k', i.e., iff t = j*3+k.
    # Sum over all t: only t=j*3+k contributes, giving 1. ✓
    #
    # Wait, but W[k*2+i, t] = 1 for all i means W has the same value for all i in each column t.
    # Let me verify: W has shape (p*n, rank) = (3*2, 9) = (6, 9).
    # W[k*2+i, t] where k in {0,1,2}, i in {0,1}.
    # For term t = j*3+k: W[k*2+i, t] = 1 for all i (0,1).
    # So W[0, t] = W[1, t] = 1 for term t corresponding to column k.
    # W[2, t] = W[3, t] = 1 for term t corresponding to column k+1... 
    #
    # Wait, I need to be more careful. For term t = j*3+k:
    #   W[k*2+i, t] = 1 for i=0,1
    #   W[k'*2+i', t] = 0 for k' != k
    #
    # So W[0*2+0, t] = W[0, t] = 1 if k=0, else 0
    # W[0*2+1, t] = W[1, t] = 1 if k=0, else 0
    # W[1*2+0, t] = W[2, t] = 1 if k=1, else 0
    # W[1*2+1, t] = W[3, t] = 1 if k=1, else 0
    # W[2*2+0, t] = W[4, t] = 1 if k=2, else 0
    # W[2*2+1, t] = W[5, t] = 1 if k=2, else 0
    #
    # This is valid! And it gives rank 9 for 2x3*3x3.
    #
    # Now for 2x3*3x6: partition B into B1 (cols 0-2) and B2 (cols 3-5).
    # Use the rank-9 decomp for each block, total rank 18.
    
    n, m, p = 2, 3, 6
    p_block = 3  # each block has 3 columns
    n_blocks = p // p_block  # 2 blocks
    
    rank_per_block = 9
    total_rank = rank_per_block * n_blocks  # 18
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(n_blocks):
        k_offset = block * p_block  # 0 or 3
        for j in range(m):  # j = 0, 1, 2
            for k_local in range(p_block):  # k_local = 0, 1, 2
                k = k_offset + k_local  # actual column index
                t = block * rank_per_block + j * p_block + k_local  # term index
                
                # U[i*m + j, t] = 1 for i = 0, 1
                for i in range(n):
                    U[i * m + j][t] = 1
                
                # V[j*p + k, t] = 1
                V[j * p + k][t] = 1
                
                # W[k*n + i, t] = 1 for i = 0, 1
                for i in range(n):
                    W[k * n + i][t] = 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-10):
        """Check that sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i,j,k"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > tol:
                        return False, (i, j, k, s)
        return True, None
    
    valid, info = verify(U, V, W, n, m, p)
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
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
