"""Program for matrix multiplication 3x4x7 tensor decomposition using Strassen blocks."""
DIMENSIONS = [3, 4, 7]

import time

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix mult with Strassen rank 7.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Index convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for n x m times m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x4x2():
    """2x4 * 4x2 using two Strassen 2x2x2 blocks (split m=4 into 2+2).
    U: (8, 14), V: (8, 14), W: (4, 14)
    U index: i*4+j, i in [0,2), j in [0,4)
    V index: j*2+k, j in [0,4), k in [0,2)
    W index: k*2+i, k in [0,2), i in [0,2)
    """
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _strassen_2x2x2()
    rank = 14
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(4)]
    
    # First 2x2x2 block: A rows 0-1, B cols 0-1 (B rows 0-1), output cols 0-1
    # U: i*4+j, for i in [0,2), j in [0,2) -> rows 0,1,4,5
    # V: j*2+k, for j in [0,2), k in [0,2) -> rows 0,1,2,3
    # W: k*2+i, for k in [0,2), i in [0,2) -> rows 0,1,2,3
    
    for i in range(2):
        for j in range(2):
            u_idx = i * 4 + j
            for c in range(7):
                U[u_idx][c] = U1[i*2+j][c]
    
    for j in range(2):
        for k in range(2):
            v_idx = j * 2 + k
            for c in range(7):
                V[v_idx][c] = V1[j*2+k][c]
    
    for k in range(2):
        for i in range(2):
            w_idx = k * 2 + i
            for c in range(7):
                W[w_idx][c] = W1[k*2+i][c]
    
    # Second 2x2x2 block: A rows 0-1, B cols 2-3 (B rows 2-3), output cols 0-1
    # U: i*4+j, for i in [0,2), j in [2,4) -> rows 2,3,6,7
    # V: j*2+k, for j in [2,4), k in [0,2) -> rows 4,5,6,7
    # W: same as first block (output cols 0-1)
    
    for i in range(2):
        for j in range(2):
            u_idx = i * 4 + (j + 2)
            for c in range(7):
                U[u_idx][7+c] = U2[i*2+j][c]
    
    for j in range(2):
        for k in range(2):
            v_idx = (j + 2) * 2 + k
            for c in range(7):
                V[v_idx][7+c] = V2[j*2+k][c]
    
    for k in range(2):
        for i in range(2):
            w_idx = k * 2 + i
            for c in range(7):
                W[w_idx][7+c] = W2[k*2+i][c]
    
    return U, V, W


def construct():
    """Construct rank-78 decomposition for 3x4 * 4x7.
    
    Split n=3 into (2,1), keep m=4, split p=7 into (2,2,2,1).
    Blocks:
      (2,4,2) x3: each rank 14 (two Strassen 2x2x2) -> 42
      (2,4,1) x1: rank 8 (schoolbook) -> 8
      (1,4,2) x3: rank 8 (schoolbook) -> 24
      (1,4,1) x1: rank 4 (schoolbook) -> 4
    Total rank: 78
    """
    n, m, p = 3, 4, 7
    n_splits = [2, 1]
    p_splits = [2, 2, 2, 1]
    
    # Compute total rank
    total_rank = 0
    for ni in n_splits:
        for pi in p_splits:
            if ni == 2 and pi == 2:
                total_rank += 14
            elif ni == 2 and pi == 1:
                total_rank += 8
            elif ni == 1 and pi == 2:
                total_rank += 8
            elif ni == 1 and pi == 1:
                total_rank += 4
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    n_offset = 0
    p_offset = 0
    
    for ni in n_splits:
        p_offset = 0
        for pi in p_splits:
            if ni == 2 and pi == 2:
                Ub, Vb, Wb = _strassen_2x4x2()
            elif ni == 2 and pi == 1:
                Ub, Vb, Wb = _schoolbook_nm_p(2, 4, 1)
            elif ni == 1 and pi == 2:
                Ub, Vb, Wb = _schoolbook_nm_p(1, 4, 2)
            elif ni == 1 and pi == 1:
                Ub, Vb, Wb = _schoolbook_nm_p(1, 4, 1)
            
            block_rank = len(Ub[0])
            
            # U: rows n_offset*m + j, j in [0, ni*m)
            # Ub is indexed as i*m + j where i in [0,ni), j in [0,m)
            for i in range(ni):
                for j in range(m):
                    u_idx = n_offset * m + i * m + j
                    for c in range(block_rank):
                        U[u_idx][offset + c] = Ub[i * m + j][c]
            
            # V: rows j*p + p_offset + k, j in [0,m), k in [0,pi)
            # Vb is indexed as j*pi + k where j in [0,m), k in [0,pi)
            for j in range(m):
                for k in range(pi):
                    v_idx = j * p + p_offset + k
                    for c in range(block_rank):
                        V[v_idx][offset + c] = Vb[j * pi + k][c]
            
            # W: rows (p_offset + k)*n + n_offset + i, k in [0,pi), i in [0,ni)
            # Wb is indexed as k*ni + i where k in [0,pi), i in [0,ni)
            for k in range(pi):
                for i in range(ni):
                    w_idx = (p_offset + k) * n + n_offset + i
                    for c in range(block_rank):
                        W[w_idx][offset + c] = Wb[k * ni + i][c]
            
            offset += block_rank
            p_offset += pi
        n_offset += ni
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
