"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Use Strassen-inspired decomposition
    # Split m=3 into 2+1
    # For 2x2xp block: Strassen gives 7p terms
    # For 2x1xp block: naive gives 2p terms
    # But we can share some structure
    
    # Actually, let's use a known 55-rank decomposition
    # The key insight: for 2x3xp, we can achieve rank 55
    
    # Let me construct this using the algebraic structure
    # C = A * B where A is 2x3, B is 3xp
    
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00,...,b0,p-1],
    #      [b10,...,b1,p-1],
    #      [b20,...,b2,p-1]]
    # C = [[c00,...,c0,p-1],
    #      [c10,...,c1,p-1]]
    
    # ci,k = a_i0*b0k + a_i1*b1k + a_i2*b2k
    
    # We need to express this as sum of rank-1 tensors
    # Using Strassen-like tricks on the a_i0, a_i1 part (2x2 submatrix)
    
    # Let me build a 55-rank decomposition
    # Strategy: for each column k of B, we have a 2x3 matrix A times 3-vector B[:,k]
    # But we need to share across k values
    
    # Known result: rank(2x3xn) = 5n + 5 for n >= 1? Let's try 55 for n=10
    
    # Construct using Strassen on 2x2 blocks with careful sharing
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # For the 2x2x10 part (using A[:,0:2] and B[0:2,:]):
    # Strassen terms for each k in range(p):
    # M1 = (a_i0 + a_i1)(b0k + b1k) for i in 0,1
    # M2 = (a10 + a11) b0k
    # M3 = a00 (b1k - b0k)
    # M4 = a11 (b1k - b0k)  -- wait, let me use standard Strassen
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Where A11=a00, A12=a01, A21=a10, A22=a11
    # B11=b0k, B12=b0k (same row), B21=b1k, B22=b1k
    
    # Wait, for 2x2x1, B is 2x1, so B11=b0k, B21=b1k, and B12=B22 don't exist in same way
    
    # For 2x2xp, the Strassen decomposition is per column:
    # Actually, the rank is 7p, meaning we need 7p independent terms
    
    # Let me just use the naive approach but with the known 55-rank structure
    
    # For now, use naive 60-rank as fallback, but structure for improvement
    # Actually, let me implement the 55-rank decomposition directly
    
    # The 55-rank decomposition for 2x3x10:
    # Use 5 blocks of 11 terms each? Or some other structure
    
    # Let me try: split into groups and use Strassen on 2x2 subblocks
    
    # For each k, we compute C[:,k] = A * B[:,k]
    # C0,k = a00*b0k + a01*b1k + a02*b2k
    # C1,k = a10*b0k + a11*b1k + a12*b2k
    
    # Group (a00, a01) and (b0k, b1k): use Strassen 2x2
    # Group a02, a12 with b2k separately
    
    # Strassen for the 2x2 part (a00,a01;a10,a11) * (b0k;b1k):
    # M1 = (a00+a11)(b0k+b1k)
    # M2 = (a10+a11)b0k  
    # M3 = a00(b1k-b0k)
    # M4 = a11(b1k-b0k)  -- hmm, this doesn't look right
    
    # Let me use the correct Strassen formula:
    # For C = A*B where A is 2x2, B is 2x1 (column vector)
    # This is just matrix-vector, rank is 4 (no savings from Strassen for single column)
    
    # The savings come from batching across k values!
    
    # For 2x2xp, the tensor rank is 7p, achieved by:
    # M1_j = (a00+a11)(b0j+b1j) for j=0..p-1 -> p terms
    # M2_j = (a10+a11)b0j for j=0..p-1 -> p terms
    # M3_j = a00(b1j-b0j) for j=0..p-1 -> p terms
    # M4_j = a11(b1j-b0j) for j=0..p-1 -> p terms  
    # M5_j = (a00+a01)b1j for j=0..p-1 -> p terms
    # M6_j = (a10-a00)(b0j+b1j) for j=0..p-1 -> p terms
    # M7_j = (a01-a11)(b1j+b0j) for j=0..p-1 -> p terms
    # Total: 7p
    
    # For 2x1xp: rank 2p
    # Total for 2x3xp: 7p + 2p = 9p = 90 for p=10. Still not 55.
    
    # The 55 must come from a different structure. Let me just use naive 60.
    
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
