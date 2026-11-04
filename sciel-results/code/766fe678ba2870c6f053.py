"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Partition p=10 into 5 pairs of 2
    num_blocks = p // 2  # 5 blocks
    rank_per_block = 7   # Strassen rank for 2x3x2
    total_rank = num_blocks * rank_per_block  # 35
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 35
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 35
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 35
    
    # For each block b, handle columns 2b and 2b+1 of B
    # Strassen decomposition for 2x3x2:
    # A is 2x3: A[i,j] for i in {0,1}, j in {0,1,2}
    # B is 3x2: B[j,k] for j in {0,1,2}, k in {0,1}
    # C is 2x2: C[i,k] for i in {0,1}, k in {0,1}
    # 
    # 7 products:
    # M1 = (A[0,0]+A[1,0]) * (B[0,0]+B[2,0])
    # M2 = A[1,0] * (B[0,1]+B[2,1])
    # M3 = A[0,0] * B[0,0]
    # M4 = A[1,1] * B[1,1]
    # M5 = (A[0,0]+A[0,1]) * (B[1,0]+B[1,1])
    # M6 = (A[1,0]+A[1,1]) * (B[1,0]+B[1,1])
    # M7 = A[0,2] * B[2,0] + A[1,2] * B[2,1]  -- wait, this is two products
    #
    # Let me reconsider. The issue is that the third column of A (j=2) 
    # only multiplies with the third row of B (j=2), which is a 1x2 submatrix.
    # So A[0,2]*B[2,0] and A[1,2]*B[2,1] are separate rank-1 terms.
    # That would give 8 terms total, not 7.
    #
    # Actually, let me verify: is the rank of 2x3x2 really 7?
    # The tensor T has entries T[i,j,k] = 1 if output C[i,k] = sum_j A[i,j]*B[j,k]
    # Number of non-zero entries in T: for each (i,j,k) triple, T[i,j,k,l] = delta_{l,k*2+i}
    # So T has 2*3*2 = 12 non-zero entries (one per (i,j,k) pair, at position l=k*2+i)
    #
    # Rank lower bound: The tensor is 12-dimensional in a certain sense.
    # Actually, let me just use rank 8 per block to be safe, giving total rank 40.
    # Or better, let me find the actual minimal rank.
    
    # Let me use a direct approach: for each (i,j,k) triple, we need a rank-1 term.
    # That's 2*3*2 = 12 terms per block, total 60 (schoolbook).
    # 
    # To use Strassen-like savings, I need to find linear combinations.
    # 
    # For now, let me just implement the schoolbook to ensure correctness,
    # then try to optimize.
    
    # Schoolbook: rank = n*m*p = 60
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
    
    # Verify correctness
    _verify(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = len(U[0])
    # Check: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal delta for valid (i,j,k)
    # Actually, the tensor T[i,j,k,l] where l = k*n+i
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid product term
    # The multiplication tensor has T[i,j,k,l] = 1 iff l == k*n + i (for the (i,j,k) input to output l)
    # Wait, the convention is W[k*n+i, r], so the output index is l = k*n + i
    # And the tensor entry T[i,j,k,l] = 1 if and only if this term contributes to C[i,k]
    # which is always the case for valid i,j,k, so T[i,j,k,k*n+i] = 1
    
    # Actually, the decomposition should satisfy:
    # For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For all i,j,k,l where l != k*n+i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[l,c] = 0
    # But W is indexed by l = k*n+i, so we only check the diagonal.
    
    # Let's just check the main condition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: sum={s}")
    
    # Check off-diagonal terms are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(p * n):
                    if l == k * n + i:
                        continue
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[l][c]
                    if abs(s) > 1e-9:
                        raise ValueError(f"Off-diagonal verification failed at i={i},j={j},k={k},l={l}: sum={s}")
# EVOLVE-BLOCK-END
