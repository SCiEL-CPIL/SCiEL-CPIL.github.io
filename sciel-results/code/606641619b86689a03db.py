"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a rank-36 decomposition based on splitting the middle dimension
    # into two 2-dim halves and using rank-18 decompositions for 2x2*2x6.
    # For 2x2*2x6, we use a construction with 18 bilinear terms.
    
    def build_2x2_2x6_decomp():
        """Return (U, V, W) for 2x2 * 2x6 -> 2x6 with rank 18."""
        # A is 2x2, B is 2x6, C = A*B is 2x6
        # U: (2*2=4) x 18, V: (2*6=12) x 18, W: (6*2=12) x 18
        # Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*6+k], C[k,i] -> W[k*2+i]
        
        U = [[0.0]*18 for _ in range(4)]
        V = [[0.0]*18 for _ in range(12)]
        W = [[0.0]*18 for _ in range(12)]
        
        # Terms 0-5: A[0,0]*B[0,k] for k=0..5
        for k in range(6):
            U[0][k] = 1.0
            V[0*6+k][k] = 1.0
            W[k*2+0][k] = 1.0
            W[k*2+1][k] = 0.0
        
        # Terms 6-11: A[1,1]*B[1,k] for k=0..5
        for k in range(6):
            U[3][6+k] = 1.0
            V[1*6+k][6+k] = 1.0
            W[k*2+0][6+k] = 0.0
            W[k*2+1][6+k] = 1.0
        
        # Terms 12-17: off-diagonal using (A[0,1]+A[1,0])*(B[0,k]+B[1,k]) - A[0,0]*B[0,k] - A[1,1]*B[1,k]
        # Actually, use Strassen-style: for each k, compute (A[0,1]+A[1,0])*(B[0,k]+B[1,k])
        # This gives A[0,1]*B[0,k] + A[0,1]*B[1,k] + A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # We need A[0,1]*B[1,k] + A[1,0]*B[0,k] (the cross terms)
        # So: cross_k = (A[0,1]+A[1,0])*(B[0,k]+B[1,k]) - A[0,0]*B[0,k] - A[1,1]*B[1,k] - A[0,1]*B[0,k] - A[1,0]*B[1,k]
        # This is getting complex. Let me use a simpler rank-18 construction.
        
        # Reset and use a different approach: 6 terms for each diagonal, and 6 terms for off-diagonals
        # using (A[0,1]+A[1,0]) and (B[0,k]+B[1,k])
        
        # Clear
        U = [[0.0]*18 for _ in range(4)]
        V = [[0.0]*18 for _ in range(12)]
        W = [[0.0]*18 for _ in range(12)]
        
        # Diagonal terms: 12 terms
        for k in range(6):
            # A[0,0]*B[0,k] -> C[0,k]
            U[0][k] = 1.0
            V[0*6+k][k] = 1.0
            W[k*2+0][k] = 1.0
            
            # A[1,1]*B[1,k] -> C[1,k]
            U[3][k+6] = 1.0
            V[1*6+k][k+6] = 1.0
            W[k*2+1][k+6] = 1.0
        
        # Off-diagonal: use 6 terms with (A[0,1]+A[1,0])*(B[0,k]+B[1,k])
        for k in range(6):
            idx = 12 + k
            U[1][idx] = 1.0  # A[0,1]
            U[2][idx] = 1.0  # A[1,0]
            V[0*6+k][idx] = 1.0  # B[0,k]
            V[1*6+k][idx] = 1.0  # B[1,k]
            # This product gives: A[0,1]*B[0,k] + A[0,1]*B[1,k] + A[1,0]*B[0,k] + A[1,1]*B[1,k]
            # We need to extract A[0,1]*B[1,k] for C[0,k] and A[1,0]*B[0,k] for C[1,k]
            # But we can't separate them in a single rank-1 term.
            # This approach doesn't work directly.
            
            # Instead, let's use the fact that we need 12 more terms for the 12 off-diagonal products
            # A[0,1]*B[1,k] and A[1,0]*B[0,k] for k=0..5
            # That's 12 terms, total 24. Not 18.
            
            pass
        
        # I'll fall back to a known correct rank-24 construction for 2x2*2x6
        # and total rank 48, but that's not an improvement.
        
        # Given the complexity, I'll return a valid rank-36 construction using a different method.
        # Use 36 terms: for each of the 4 middle indices j and 6 output columns k, 
        # but share across the 2 input rows.
        
        # Actually, let me just use the schoolbook for the full problem but with rank 48.
        # The improvement isn't achievable with my current knowledge.
        
        U = [[0.0]*18 for _ in range(4)]
        V = [[0.0]*18 for _ in range(12)]
        W = [[0.0]*18 for _ in range(12)]
        
        # Rank-18 construction for 2x2*2x6:
        # Use 6 terms: A[0,0]*B[0,k] -> C[0,k]
        # Use 6 terms: A[1,1]*B[1,k] -> C[1,k]  
        # Use 6 terms: (A[0,1]+A[1,0])*(B[0,k]+B[1,k]) and then subtract the diagonal parts
        # But subtraction requires additional terms.
        
        # I'll use a verified rank-24 construction and accept rank 48 total.
        raise ValueError("Construction not ready")
    
    # Given the difficulty, I'll stick with a correct rank-48 schoolbook decomposition
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
