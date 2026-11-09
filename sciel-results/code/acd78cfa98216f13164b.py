"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
_start = time.time()
DEADLINE = 120.0

def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-24 decomposition
    # Strategy: group the 6 columns of B into 3 pairs, share computations
    # For each pair of columns (k1, k2), we compute shared intermediates
    
    # Rank 24 approach:
    # We can share the 3 intermediate products per row across column pairs
    # But we need to be careful about the exact decomposition
    
    # Let's construct a valid lower-rank decomposition
    # Use the fact that for 2×3×6, we can achieve rank 24
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 6 × 24
    V = [[0] * rank for _ in range(m * p)]  # 18 × 24
    W = [[0] * rank for _ in range(p * n)]  # 12 × 24
    
    # Decomposition strategy:
    # For each column pair (2k, 2k+1), use 4 terms per pair × 2 rows... 
    # Actually let me use a structured approach.
    
    # Schoolbook for each column would be 3 multiplications × 2 rows × 6 cols = 36
    # To get 24, we share: group columns into 3 groups of 2
    # For each group, compute 3 shared A-row × B-partial-sum products
    
    # For group g (columns 2g, 2g+1):
    #   For each row i (2 rows), compute sum_j A[i,j] * B[j, 2g] and sum_j A[i,j] * B[j, 2g+1]
    #   These share the A row, but we still need 2 multiplications per (i, col)
    #   That's still 36...
    
    # Better: use a Strassen-like approach for the 2×3 part
    # Actually, for 2×3, the bilinear complexity is 6 (schoolbook is optimal for 2×3? No, it's 6)
    # Wait, 2×3 multiplication: C[i,k] = sum_j A[i,j] B[j,k] for i=0,1; k=0..2
    # That's 2*3*3 = 18 for 3 columns, so 6 per column
    # For 6 columns: 36 schoolbook
    
    # To reduce, I can use the structure that A has only 2 rows
    # The product is C = A @ B where A is 2×3, B is 3×6
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # This is like multiplying a 2-vector (row of A) with 6 columns of B
    # For each column k, we compute a dot product of length 3
    # Total: 2 rows × 6 cols × 3 mults = 36
    
    # Can we share across columns? The A row entries are shared across all columns.
    # Precompute: For row 0, we have A[0,0], A[0,1], A[0,2]
    # For each column k, C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    
    # This is a fixed linear combination of B's columns. No sharing helps here
    # since each column k uses different B entries.
    
    # Actually, the tensor is M_{2,3,6} and its rank is the bilinear complexity.
    # Known: M_{2,3,6} rank. Let me just return schoolbook to be safe.
    
    # Actually, I recall that M_{2,n,p} can have rank less than 2*n*p
    # For M_{2,3,6}: we can use 24 by a known construction
    
    # Let me try: use 3 multiplications per "super-column" pair
    # No, let me just verify schoolbook works and return it, then try to improve.
    
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
