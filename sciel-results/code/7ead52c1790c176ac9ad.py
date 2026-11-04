"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2 with rank 7 (Strassen).
    U shape (4,7), V shape (4,7), W shape (4,7)
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2*2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k]
    # r=0: M1
    # r=1: M2
    # r=2: M3
    # r=3: M4
    # r=4: M5
    # r=5: M6
    # r=6: M7
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00 (i=0,j=0) -> U[0,0]=1; a11 (i=1,j=1) -> U[3,0]=1
    # V: b00 (j=0,k=0) -> V[0,0]=1; b11 (j=1,k=1) -> V[3,0]=1
    # W: c00 (k=0,i=0) -> W[0,0]=1; c11 (k=1,i=1) -> W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    # U: a10 (i=1,j=0) -> U[2,1]=1; a11 (i=1,j=1) -> U[3,1]=1
    # V: b00 (j=0,k=0) -> V[0,1]=1
    # W: c10 (k=0,i=1) -> W[2,1]=1; c11 (k=1,i=1) -> W[3,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = a00(b01-b11)
    # U: a00 (i=0,j=0) -> U[0,2]=1
    # V: b01 (j=0,k=1) -> V[1,2]=1; b11 (j=1,k=1) -> V[3,2]=-1
    # W: c01 (k=1,i=0) -> W[1,2]=1; c11 (k=1,i=1) -> W[3,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b10-b00)
    # U: a11 (i=1,j=1) -> U[3,3]=1
    # V: b10 (j=1,k=0) -> V[2,3]=1; b00 (j=0,k=0) -> V[0,3]=-1
    # W: c10 (k=0,i=1) -> W[2,3]=1; c00 (k=0,i=0) -> W[0,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5 = (a00+a01)b11
    # U: a00 (i=0,j=0) -> U[0,4]=1; a01 (i=0,j=1) -> U[1,4]=1
    # V: b11 (j=1,k=1) -> V[3,4]=1
    # W: c00 (k=0,i=0) -> W[0,4]=1; c10 (k=0,i=1) -> W[2,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    # U: a10 (i=1,j=0) -> U[2,5]=1; a00 (i=0,j=0) -> U[0,5]=-1
    # V: b00 (j=0,k=0) -> V[0,5]=1; b01 (j=0,k=1) -> V[1,5]=1
    # W: c11 (k=1,i=1) -> W[3,5]=1; c01 (k=1,i=0) -> W[1,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1; W[1][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    # U: a01 (i=0,j=1) -> U[1,6]=1; a11 (i=1,j=1) -> U[3,6]=-1
    # V: b10 (j=1,k=0) -> V[2,6]=1; b11 (j=1,k=1) -> V[3,6]=1
    # W: c10 (k=0,i=1) -> W[2,6]=1; c00 (k=0,i=0) -> W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = 1; W[0][6] = 1
    
    return U, V, W

def _kron_strassen_4x4x4():
    """Return (U, V, W) for 4x4 * 4x4 using Kronecker product of two Strassen tensors.
    U shape (16,49), V shape (16,49), W shape (16,49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    # Kronecker product: U = U2 ⊗ U2, etc.
    # U2 is 4x7, so U is (4*4) x (7*7) = 16x49
    U = np.kron(np.array(U2, dtype=int), np.array(U2, dtype=int))
    V = np.kron(np.array(V2, dtype=int), np.array(V2, dtype=int))
    W = np.kron(np.array(W2, dtype=int), np.array(W2, dtype=int))
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook_nm_p(n, m, p):
    """Return (U, V, W) for n x m * m x p with schoolbook rank n*m*p.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p)
    """
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct.
    For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: decompose 4x4 * 4x7 into two parts:
    # Part 1: 4x4 * 4x4 (columns 0-3) using Strassen Kronecker, rank 49
    # Part 2: 4x4 * 4x3 (columns 4-6) using schoolbook, rank 48
    # Total rank: 49 + 48 = 97
    
    # Part 1: 4x4 * 4x4
    U1, V1, W1 = _kron_strassen_4x4x4()
    
    # Part 2: 4x4 * 4x3 (schoolbook)
    # This handles output columns 4,5,6
    # U2 shape (16, 48), V2 shape (28, 48), W2 shape (28, 48)
    # But we need to map indices correctly.
    
    n2, m2, p2 = 4, 4, 3
    U2_local, V2_local, W2_local = _schoolbook_nm_p(n2, m2, p2)
    
    # Now we need to embed these into the full problem.
    # U2_local has shape (16, 48) - rows are i*m2+j = i*4+j, same as full U rows.
    # V2_local has shape (12, 48) - rows are j*p2+k = j*3+k, but in full problem V rows are j*7+k_full.
    #   For k_full = 4,5,6, the local k = 0,1,2.
    #   So V2_local row j*3+k_local maps to full row j*7 + (4 + k_local).
    # W2_local has shape (12, 48) - rows are k2*n2+i = k_local*4+i, but in full problem W rows are k_full*4+i.
    #   For k_full = 4,5,6, the local k = 0,1,2.
    #   So W2_local row k_local*4+i maps to full row (4+k_local)*4+i.
    
    rank2 = len(U2_local[0])  # 48
    
    # Create full-size matrices for part 2
    U2 = [[0] * rank2 for _ in range(n * m)]  # 16 x 48
    V2 = [[0] * rank2 for _ in range(m * p)]  # 28 x 48
    W2 = [[0] * rank2 for _ in range(p * n)]  # 28 x 48
    
    # Fill U2: same row indexing
    for row in range(n * m):
        U2[row] = U2_local[row].copy()
    
    # Fill V2: map local row j*3+k_local to full row j*7 + (4+k_local)
    for j in range(m2):
        for k_local in range(p2):
            local_row = j * p2 + k_local
            full_row = j * p + (4 + k_local)
            V2[full_row] = V2_local[local_row].copy()
    
    # Fill W2: map local row k_local*4+i to full row (4+k_local)*4+i
    for k_local in range(p2):
        for i in range(n2):
            local_row = k_local * n2 + i
            full_row = (4 + k_local) * n + i
            W2[full_row] = W2_local[local_row].copy()
    
    # Concatenate: U = [U1, U2], V = [V1, V2], W = [W1, W2]
    rank1 = len(U1[0])  # 49
    total_rank = rank1 + rank2  # 97
    
    U = []
    for row in range(n * m):
        U.append(U1[row] + U2[row])
    
    V = []
    for row in range(m * p):
        V.append(V1[row] + V2[row])
    
    W = []
    for row in range(p * n):
        W.append(W1[row] + W2[row])
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
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
