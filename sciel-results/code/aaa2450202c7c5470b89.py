"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_t U[:,t] V[:,t] W[:,t] = multiplication tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Compute the tensor: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Expected: T[i,j,k] = 1 if this is the (i,k) output from A[i,j]*B[j,k]
    # Actually, the multiplication tensor T[i,j,k] = 1 for all (i,j,k) since
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Build expected tensor
    expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1.0
    
    # Compute actual tensor
    actual = np.zeros((n, m, p))
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual[i, j, k] += Ua[i*m+j, t] * Va[j*p+k, t] * Wa[k*n+i, t]
    
    return np.allclose(actual, expected, atol=1e-10)

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Try rank-18 algebra-based decomposition
    # Decomposition: 2x3 x 3x9 = three instances of 2x3 x 3x3
    # Use rank-6 for each 2x3x3 block if possible, else fall back
    
    # First, try to construct a rank-18 decomposition using block structure
    # The 3x9 matrix B is split into three 3x3 blocks: B0, B1, B2
    # C = A * B = A*B0 + A*B1 + A*B2 (but these are separate column blocks)
    # So C[:, 0:3] = A*B0, C[:, 3:6] = A*B1, C[:, 6:9] = A*B2
    
    # For each 2x3 x 3x3 multiplication, schoolbook rank is 18.
    # But we can try to find a lower rank decomposition.
    
    # Let's try a direct construction with rank 18:
    # Use 6 terms per block * 3 blocks = 18 terms
    # For 2x3 x 3x3, we need rank-6. This is non-trivial.
    
    # Alternative: just use schoolbook but try to find any valid lower rank via search
    # Given time constraints, let's try a structured approach
    
    # Try rank 27: use some sharing
    # Actually, let's just verify schoolbook and return it as fallback
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    if _verify_decomposition(U, V, W, n, m, p):
        pass  # Schoolbook is valid
    
    # Try to find a lower rank decomposition
    # Use a simple heuristic: try rank from 18 to 53
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # For this problem, let's try a specific algebraic construction
    # Rank 18 construction using 3x3 block decomposition with Strassen-like ideas
    
    # The multiplication 2x3 x 3x9 can be viewed as:
    # Output C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # where k ranges over 0..8
    
    # Let's try to construct rank-18 by exploiting the structure
    # where B is 3x9 = three 3x3 blocks
    
    if time.time() < deadline:
        # Try a structured rank-18 decomposition
        # For 2x3 x 3x9: C[i,k] = sum_j A[i,j] B[j,k]
        # Split k into three groups of 3: k = 3*q + r, q in {0,1,2}, r in {0,1,2}
        # C[i, 3q+r] = sum_j A[i,j] B[j, 3q+r]
        
        # Use rank-6 per (q) block: for each q, we do 2x3 x 3x3 multiplication
        # 2x3 x 3x3 schoolbook is rank 18, but we try rank 6
        
        # Rank-6 for 2x3 x 3x3: 
        # C[i,r] = sum_j A[i,j] B[j,r] for i in {0,1}, r in {0,1,2}
        # 6 output entries, 6 input pairs (i,j) x (j,r) = 3x3 = 9 pairs per i
        # Total 18 scalar products schoolbook
        
        # Try: use 6 terms where each term covers multiple outputs
        # Term t: u_t[i,j] * v_t[j,r] * w_t[r,i]
        
        # Let's try a specific construction:
        # For 2x3 x 3x3, try:
        # t=0: u=(1,0,0), v=(1,0,0), w=(1,0,0) -> covers (i=0,j=0,r=0)
        # This is just schoolbook again...
        
        # The issue is that 2x3 x 3x3 likely has rank > 6
        # Let's just use schoolbook for each block: 18 per block * 3 = 54 (no improvement)
        
        # Alternative: use a shared decomposition
        # Let's try rank 27: use 9 terms that each cover 2 positions
        
        # Actually, let's try a completely different approach:
        # Use the fact that we only have 2 rows in A
        # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
        # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
        
        # For each k, we have 2 outputs that are linear combinations of 3 B-rows
        # This is like a 2x3 matrix times a 3x1 vector, done for each k
        
        # The tensor structure: T[i,j,k] = 1
        # We need sum_t u_t[i,j] v_t[j,k] w_t[k,i] = 1
        
        # Let's try a rank-18 construction with a specific pattern
        # Use 18 terms indexed by (i, j, q) where q in {0,1,2} selects the block
        # But that's just schoolbook rearranged
        
        # Given the difficulty, let's just return schoolbook
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
