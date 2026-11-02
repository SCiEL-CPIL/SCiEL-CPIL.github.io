"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # The tensor decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: indices (i*2+j, r) for i,j in {0,1}
    # U[0, r] = A00, U[1, r] = A01, U[2, r] = A10, U[3, r] = A11
    U = [[0]*7 for _ in range(4)]
    # M1: A00 + A11 -> U[0,0]=1, U[3,0]=1
    # M2: A10 + A11 -> U[2,1]=1, U[3,1]=1
    # M3: A00 -> U[0,2]=1
    # M4: A11 -> U[3,3]=1
    # M5: A00 + A01 -> U[0,4]=1, U[1,4]=1
    # M6: A10 - A00 -> U[2,5]=1, U[0,5]=-1
    # M7: A01 - A11 -> U[1,6]=1, U[3,6]=-1
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    # V: indices (j*2+k, r) for j,k in {0,1}
    # V[0, r] = B00, V[1, r] = B01, V[2, r] = B10, V[3, r] = B11
    V = [[0]*7 for _ in range(4)]
    # M1: B00 + B11 -> V[0,0]=1, V[3,0]=1
    # M2: B00 -> V[0,1]=1
    # M3: B01 - B11 -> V[1,2]=1, V[3,2]=-1
    # M4: B10 - B00 -> V[2,3]=1, V[0,3]=-1
    # M5: B11 -> V[3,4]=1
    # M6: B00 + B01 -> V[0,5]=1, V[1,5]=1
    # M7: B10 + B11 -> V[2,6]=1, V[3,6]=1
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    # W: indices (k*2+i, r) for k,i in {0,1}
    # W[0, r] = C00, W[1, r] = C01, W[2, r] = C10, W[3, r] = C11
    W = [[0]*7 for _ in range(4)]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _strassen_4x4x8():
    """
    Construct a decomposition for 4x4 x 4x8 matrix multiplication.
    View as two independent 4x4 x 4x4 multiplications.
    Use Strassen's algorithm on 2x2 blocks within each 4x4.
    
    For 4x4 x 4x4:
    - Split into four 2x2 blocks
    - Each 2x2x2 multiplication uses rank 7
    - Total rank: 4 blocks * 7 = 28 per 4x4x4
    
    For 4x4 x 4x8:
    - Split 4x8 into two 4x4 blocks
    - Each 4x4 x 4x4 uses rank 28
    - Total rank: 2 * 28 = 56
    
    This gives rank 56, better than schoolbook rank 128.
    """
    n, m, p = 4, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (16, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (32, 56)
    
    # Get Strassen 2x2x2 decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # For each of the two 4x4 output blocks (k in {0,1} for the 4x8 split)
    for k_block in range(2):
        # For each 2x2 block in the 4x4x4 multiplication
        for i_block in range(2):  # row block of A (and C)
            for j_block in range(2):  # col block of A / row block of B
                # For each Strassen product
                for r_s in range(7):
                    col = k_block * 28 + i_block * 14 + j_block * 7 + r_s
                    
                    # U: A is 4x4, indexed by (i, j)
                    # i = i_block * 2 + i_local, j = j_block * 2 + j_local
                    for i_local in range(2):
                        for j_local in range(2):
                            i = i_block * 2 + i_local
                            j = j_block * 2 + j_local
                            U[i * m + j][col] = U_s[i_local * 2 + j_local][r_s]
                    
                    # V: B is 4x8, indexed by (j, k)
                    # j = j_block * 2 + j_local, k = k_block * 4 + k_local (since 4x8 split into two 4x4)
                    # Wait, 4x8: j in [0,4), k in [0,8)
                    # k_block splits k: k = k_block * 4 + k_local where k_local in [0,4)
                    # But we're doing 4x4 x 4x4, so for each k_block, B[:, k_block*4:(k_block+1)*4]
                    for j_local in range(2):
                        for k_local in range(2):
                            j = j_block * 2 + j_local
                            k = k_block * 4 + k_local
                            V[j * p + k][col] = V_s[j_local * 2 + k_local][r_s]
                    
                    # W: C is 4x8, indexed by (k, i)
                    # k = k_block * 4 + k_local, i = i_block * 2 + i_local
                    for k_local in range(2):
                        for i_local in range(2):
                            k = k_block * 4 + k_local
                            i = i_block * 2 + i_local
                            W[k * n + i][col] = W_s[k_local * 2 + i_local][r_s]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try Strassen-based decomposition
    best = _strassen_4x4x8()
    best_rank = len(best["U"][0])
    
    # Verify the decomposition
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # If verification fails, fall back to schoolbook
    best = _schoolbook(n, m, p)
    return best
# EVOLVE-BLOCK-END
