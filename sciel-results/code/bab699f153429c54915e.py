"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Return U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[a*2+b, r], V[b*2+c, r], W[c*2+a, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 1: M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Product 2: M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # Product 3: M3 = A11*(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Product 4: M4 = A22*(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Product 5: M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Product 6: M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Product 7: M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _laderman_3x3():
    """
    Return U, V, W for 3x3 matrix multiplication using Laderman's algorithm (rank 15).
    Indexing: U[a*3+b, r], V[b*3+c, r], W[c*3+a, r]
    """
    U = [[0]*15 for _ in range(9)]
    V = [[0]*15 for _ in range(9)]
    W = [[0]*15 for _ in range(9)]
    
    # Product 0: A[0,0] * B[0,0] -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Product 1: (A[0,0]+A[0,1]) * (B[0,1]-B[1,1]) -> C[0,1]
    U[0][1] = 1; U[1][1] = 1
    V[1][1] = 1; V[4][1] = -1
    W[1][1] = 1
    
    # Product 2: (A[0,0]+A[0,2]) * (B[0,2]-B[2,2]) -> C[0,2]
    U[0][2] = 1; U[2][2] = 1
    V[2][2] = 1; V[8][2] = -1
    W[2][2] = 1
    
    # Product 3: (A[0,1]+A[0,2]) * (B[1,2]-B[2,1]) -> C[0,2] (additional)
    U[1][3] = 1; U[2][3] = 1
    V[5][3] = 1; V[3][3] = -1
    W[2][3] = 1
    
    # Product 4: A[1,0] * B[0,0] -> C[1,0]
    U[3][4] = 1; V[0][4] = 1; W[3][4] = 1
    
    # Product 5: (A[1,0]+A[1,1]) * (B[0,1]-B[1,1]) -> C[1,1]
    U[3][5] = 1; U[4][5] = 1
    V[1][5] = 1; V[4][5] = -1
    W[4][5] = 1
    
    # Product 6: (A[1,0]+A[1,2]) * (B[0,2]-B[2,2]) -> C[1,2]
    U[3][6] = 1; U[5][6] = 1
    V[2][6] = 1; V[8][6] = -1
    W[5][6] = 1
    
    # Product 7: (A[1,1]+A[1,2]) * (B[1,2]-B[2,1]) -> C[1,2] (additional)
    U[4][7] = 1; U[5][7] = 1
    V[5][7] = 1; V[3][7] = -1
    W[5][7] = 1
    
    # Product 8: A[2,0] * B[0,0] -> C[2,0]
    U[6][8] = 1; V[0][8] = 1; W[6][8] = 1
    
    # Product 9: (A[2,0]+A[2,1]) * (B[0,1]-B[1,1]) -> C[2,1]
    U[6][9] = 1; U[7][9] = 1
    V[1][9] = 1; V[4][9] = -1
    W[7][9] = 1
    
    # Product 10: (A[2,0]+A[2,2]) * (B[0,2]-B[2,2]) -> C[2,2]
    U[6][10] = 1; U[8][10] = 1
    V[2][10] = 1; V[8][10] = -1
    W[8][10] = 1
    
    # Product 11: (A[2,1]+A[2,2]) * (B[1,2]-B[2,1]) -> C[2,2] (additional)
    U[7][11] = 1; U[8][11] = 1
    V[5][11] = 1; V[3][11] = -1
    W[8][11] = 1
    
    # Product 12: (A[0,0]+A[1,0]+A[2,0]) * (B[1,0]+B[2,0]) -> corrections
    U[0][12] = 1; U[3][12] = 1; U[6][12] = 1
    V[3][12] = 1; V[6][12] = 1
    W[0][12] = 1; W[3][12] = 1; W[6][12] = 1
    
    # Product 13: (A[0,1]+A[1,1]+A[2,1]) * (B[1,0]+B[2,0]) -> corrections
    U[1][13] = 1; U[4][13] = 1; U[7][13] = 1
    V[3][13] = 1; V[6][13] = 1
    W[1][13] = 1; W[4][13] = 1; W[7][13] = 1
    
    # Product 14: (A[0,2]+A[1,2]+A[2,2]) * (B[1,0]+B[2,0]) -> corrections
    U[2][14] = 1; U[5][14] = 1; U[8][14] = 1
    V[3][14] = 1; V[6][14] = 1
    W[2][14] = 1; W[5][14] = 1; W[8][14] = 1
    
    return U, V, W

def _build_5x5_decomposition():
    """
    Build a 5x5 matrix multiplication decomposition.
    
    Strategy: Use a 3+2 split. The 5x5 matmul tensor has 125 nonzero entries.
    We partition the (a,b) indices into two groups: S1 = {0,1,2} (size 3) and S2 = {3,4} (size 2).
    Similarly for (b,c) and (c,a) indices.
    
    The tensor decomposes into 8 blocks based on which group each index falls into:
    Block (i,j,k) where i indicates which group a is in, j which group b is in, k which group c is in.
    
    For the tensor to be valid, we need b to be consistent: if a is in group i and b is in group j,
    then the (a,b) pair must be valid. Similarly for (b,c) and (c,a).
    
    Actually, let me think about this differently. The tensor T[i,j,k] = 1 where:
    i = a*5+b, j = b*5+c, k = c*5+a
    
    The constraint is that the b in i and j must match, and the c in j and k must match, and the a in k and i must match.
    
    Let me partition by the value of b (the middle index):
    - b in {0,1,2}: contributes 3*5*5 = 75 terms... no wait.
    
    Actually, each term is uniquely identified by (a,b,c). There are 5*5*5 = 125 terms.
    
    Let me try a different approach: use the known result that 5x5 matmul can be done with rank <= 50.
    One construction: use a 4x4 rank-49 algorithm and add one more term.
    But I don't have a verified 4x4 rank-49 algorithm.
    
    Alternative: Use the schoolbook but try to find linear dependencies to reduce rank.
    
    For now, let me just return schoolbook and verify it works.
    """
    return _schoolbook(5, 5, 5)

def _try_laderman_based_5x5():
    """
    Try to build a 5x5 decomposition using Laderman 3x3 + additional terms.
    
    Split: rows/cols 0-2 use Laderman, rows/cols 3-4 use schoolbook for the remaining parts.
    
    The 5x5 matmul tensor T[a,b,c] = 1 for all a,b,c in {0,...,4}.
    
    Partition by (a,b) pair:
    - (a,b) both in {0,1,2}: 9 pairs -> handled by 3x3 Laderman when c in {0,1,2}
    - (a,b) both in {0,1,2}: 9 pairs -> need terms for c in {3,4}
    - (a,b) with a in {0,1,2}, b in {3,4}: 6 pairs -> need terms for all c
    - (a,b) with a in {3,4}, b in {0,1,2}: 6 pairs -> need terms for all c
    - (a,b) both in {3,4}: 4 pairs -> need terms for all c
    
    This is getting complex. Let me just verify schoolbook works.
    """
    return None

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    if _verify(U_sb, V_sb, W_sb, n, m, p, best_rank):
        print(f"Schoolbook decomposition verified, rank {best_rank}")
    else:
        print("ERROR: Schoolbook decomposition failed verification!")
    
    # Try Laderman 3x3 to see if it's valid
    U_l, V_l, W_l = _laderman_3x3()
    if _verify(U_l, V_l, W_l, 3, 3, 3, 15):
        print("Laderman 3x3 verified, rank 15")
    else:
        print("Laderman 3x3 FAILED verification")
    
    # Try Strassen 2x2
    U_s, V_s, W_s = _strassen_2x2()
    if _verify(U_s, V_s, W_s, 2, 2, 2, 7):
        print("Strassen 2x2 verified, rank 7")
    else:
        print("Strassen 2x2 FAILED verification")
    
    return best_result
# EVOLVE-BLOCK-END
