"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, trials=50):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rng = np.random.RandomState(42)
    for trial in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _build_rank24():
    """Build a rank-24 decomposition for (2,3,8).
    
    For each j in {0,1,2}, the contribution to C is A[:,j] * B[j,:] (outer product).
    A[:,j] is 2-dim, B[j,:] is 8-dim. The outer product is 2x8 = 16 entries.
    We can decompose this 2x8 outer product using 16 rank-1 terms (schoolbook for that block),
    but we can do better: the outer product of a 2-vector and 8-vector has a special structure.
    
    Actually, for fixed j: C[i,k] += A[i,j] * B[j,k]
    This is 16 bilinear forms (one per (i,k) pair). For 3 values of j, that's 48.
    
    To get rank 24: For fixed j, use the fact that A[:,j] has 2 components.
    We can write: for fixed j, the 16 entries are A[0,j]*B[j,0], A[0,j]*B[j,1], ..., A[1,j]*B[j,7]
    = A[0,j] * [B[j,0],...,B[j,7]] + A[1,j] * [B[j,0],...,B[j,7]]
    
    Hmm, that's still 2*8=16 per j.
    
    Let me try a different grouping. Split the 8 columns into 4 groups of 2.
    For each group of 2 columns, we have a (2,3,2) subproblem.
    If (2,3,2) has rank R, total rank = 4*R.
    
    Let me try to find the rank of (2,3,2) by solving a linear system.
    """
    # For now, use a rank-24 construction:
    # Group columns into 4 pairs. For each pair, use 6 forms (rank 6 for 2x3x2).
    # Total: 4 * 6 = 24.
    
    n, m, p = 2, 3, 8
    rank_per_pair = 6
    rank = 4 * rank_per_pair  # 24
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For (2,3,2): A(2x3) x B(3x2) = C(2x2)
    # C[0,0] = a0*b0 + a1*b1 + a2*b2
    # C[0,1] = a0*c0 + a1*c1 + a2*c2
    # C[1,0] = d0*b0 + d1*b1 + d2*b2
    # C[1,1] = d0*c0 + d1*c1 + d2*c2
    #
    # Rank-6 decomposition (schoolbook for 2x3x2 is 12, but we can do 6):
    # Actually, let me just use 6 forms:
    # f0 = a0*b0 -> C[0,0]
    # f1 = a1*b1 -> C[0,0]
    # f2 = a2*b2 -> C[0,0]
    # f3 = a0*c0 -> C[0,1]
    # f4 = a1*c1 -> C[0,1]
    # f5 = a2*c2 -> C[0,1]
    # But we also need C[1,0] and C[1,1]!
    #
    # So rank-6 isn't enough for (2,3,2). We need at least 8 (4 per output row).
    # Let me use rank 8 per pair: total 4*8 = 32.
    
    # Actually, let me just use the straightforward approach:
    # For each output entry C[i,k], it's a dot product of length 3.
    # 16 output entries * 3 terms = 48 (schoolbook).
    # 
    # To reduce: note that C[0,k] and C[1,k] share the same B[j,k] factors.
    # For fixed k, C[:,k] = A @ B[:,k] (2x3 times 3x1 = 2x1)
    # This is 6 bilinear forms for one column k. For 8 columns: 48.
    # 
    # No savings here with simple grouping.
    
    # Let me just return schoolbook rank 48 but with a clean structure.
    rank = n * m * p  # 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Use schoolbook decomposition (rank 48) which is guaranteed correct.
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
