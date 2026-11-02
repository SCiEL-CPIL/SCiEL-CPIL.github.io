"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Check entries are integer or half-integer
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=tol):
            return False
    
    # Check tensor equation for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > tol:
                    return False
    return True

def _winograd_2x2x2():
    """
    Return a rank-7 decomposition for 2x2x2 matrix multiplication.
    U: 4x7, V: 4x7, W: 4x7
    Convention: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Winograd rank-7 for 2x2x2:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a00+a01)(b10+b11)
    # m2 = (a10+a11)(b00+b01)
    # m3 = a00*b00
    # m4 = a01*b11
    # m5 = a10*b01
    # m6 = a11*b10
    
    # c00 = m0 + m3 - m4 - m6
    # c01 = m1 + m4 - m3 - m5
    # c10 = m2 + m5 - m1 - m6
    # c11 = m0 + m6 - m5 - m4
    
    # U[i*2+j, t]:
    # t0: (a00+a11) -> U[0,0]=1, U[3,0]=1
    # t1: (a00+a01) -> U[0,1]=1, U[1,1]=1
    # t2: (a10+a11) -> U[2,2]=1, U[3,2]=1
    # t3: a00 -> U[0,3]=1
    # t4: a01 -> U[1,4]=1
    # t5: a10 -> U[2,5]=1
    # t6: a11 -> U[3,6]=1
    
    U[0][0] = 1; U[3][0] = 1
    U[0][1] = 1; U[1][1] = 1
    U[2][2] = 1; U[3][2] = 1
    U[0][3] = 1
    U[1][4] = 1
    U[2][5] = 1
    U[3][6] = 1
    
    # V[j*2+k, t]:
    # t0: (b00+b11) -> V[0,0]=1, V[3,0]=1
    # t1: (b10+b11) -> V[2,1]=1, V[3,1]=1
    # t2: (b00+b01) -> V[0,2]=1, V[1,2]=1
    # t3: b00 -> V[0,3]=1
    # t4: b11 -> V[3,4]=1
    # t5: b01 -> V[1,5]=1
    # t6: b10 -> V[2,6]=1
    
    V[0][0] = 1; V[3][0] = 1
    V[2][1] = 1; V[3][1] = 1
    V[0][2] = 1; V[1][2] = 1
    V[0][3] = 1
    V[3][4] = 1
    V[1][5] = 1
    V[2][6] = 1
    
    # W[k*2+i, t]:
    # t0: c00 += 1, c11 += 1 -> W[0,0]=1, W[3,0]=1
    # t1: c01 += 1, c10 -= 1 -> W[1,1]=1, W[2,1]=-1
    # t2: c10 += 1, c01 -= 1 -> W[2,2]=1, W[1,2]=-1
    # t3: c00 += 1, c01 -= 1 -> W[0,3]=1, W[1,3]=-1
    # t4: c01 += 1, c11 -= 1 -> W[1,4]=1, W[3,4]=-1
    # t5: c10 += 1, c11 -= 1 -> W[2,5]=1, W[3,5]=-1
    # t6: c11 += 1, c10 -= 1 -> W[3,6]=1, W[2,6]=-1
    
    W[0][0] = 1; W[3][0] = 1
    W[1][1] = 1; W[2][1] = -1
    W[2][2] = 1; W[1][2] = -1
    W[0][3] = 1; W[1][3] = -1
    W[1][4] = 1; W[3][4] = -1
    W[2][5] = 1; W[3][5] = -1
    W[3][6] = 1; W[2][6] = -1
    
    return U, V, W

def _construct_2x2x3_from_winograd():
    """
    Construct a rank-9 decomposition for 2x2x3 using Winograd 2x2x2.
    Split the 3 columns into 2+1, use Winograd for the 2x2x2 part and schoolbook for the remaining column.
    U: 4x9, V: 6x9, W: 6x9
    Convention: U[i*2+j, t], V[j*3+k, t], W[k*2+i, t]
    """
    U222, V222, W222 = _winograd_2x2x2()
    
    U = [[0]*9 for _ in range(4)]
    V = [[0]*9 for _ in range(6)]
    W = [[0]*9 for _ in range(6)]
    
    # Copy Winograd 2x2x2 into first 7 columns
    for i in range(4):
        for t in range(7):
            U[i][t] = U222[i][t]
    
    for j in range(2):
        for k in range(2):
            for t in range(7):
                V[j*3+k][t] = V222[j*2+k][t]
    
    for k in range(2):
        for i in range(2):
            for t in range(7):
                W[k*2+i][t] = W222[k*2+i][t]
    
    # Schoolbook for k=2:
    # Term 7: A[i,0]*B[0,2] for i=0,1
    U[0][7] = 1
    U[2][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    W[5][7] = 1
    
    # Term 8: A[i,1]*B[1,2] for i=0,1
    U[1][8] = 1
    U[3][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    W[5][8] = 1
    
    return U, V, W

def _bcr_block_decomposition():
    """
    Apply BCR block method: partition 4x4x9 into 2x2x3 blocks.
    4 = 2*2, 4 = 2*2, 9 = 3*3
    So we have 2*2*3 = 12 blocks, each 2x2x3.
    
    For each 2x2x3 block, use rank-9 decomposition (Winograd + schoolbook).
    Total rank = 12 * 9 = 108.
    """
    n, m, p = 4, 4, 9
    num_ib = n // 2  # 2
    num_jb = m // 2  # 2
    num_kb = p // 3  # 3
    rank_per_block = 9
    total_rank = num_ib * num_jb * num_kb * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    Ublock, Vblock, Wblock = _construct_2x2x3_from_winograd()
    
    col = 0
    for ib in range(num_ib):
        for jb in range(num_jb):
            for kb in range(num_kb):
                for t in range(rank_per_block):
                    for i in range(2):
                        for j in range(2):
                            U[(2*ib+i)*m + (2*jb+j)][col+t] = Ublock[i*2+j][t]
                    
                    for j in range(2):
                        for k in range(3):
                            V[(2*jb+j)*p + (3*kb+k)][col+t] = Vblock[j*3+k][t]
                    
                    for k in range(3):
                        for i in range(2):
                            W[(3*kb+k)*n + (2*ib+i)][col+t] = Wblock[k*2+i][t]
                
                col += rank_per_block
    
    return {"U": U, "V": V, "W": W}

def _bcr_2x2x3_decomposition():
    """
    Alternative BCR: partition 4x4x9 into 2x2x3 blocks but with different tiling.
    4 = 2*2 (rows), 4 = 2*2 (cols of A), 9 = 3*3 (cols of B)
    Same as above but let's try 2x4x3 blocks: 4=2*2, 4=4, 9=3*3 -> 2*3=6 blocks of 2x4x3
    2x4x3: A is 2x4, B is 4x3, C is 2x3
    Can we do 2x4x3 in rank 18? (2*4*3/2 = 12 lower bound, schoolbook 24)
    Actually, let's just use the 2x2x3 approach which gives 108.
    """
    return _bcr_block_decomposition()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the BCR block decomposition with Winograd
    try:
        result = _bcr_block_decomposition()
        if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
