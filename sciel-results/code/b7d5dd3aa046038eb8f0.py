"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_2x2x2():
    """
    Build the Strassen 2x2x2 tensor decomposition.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists.
    
    For 2x2x2, indices: i,j,k in {0,1}
    U index: i*2+j (0..3)
    V index: j*2+k (0..3)  
    W index: k*2+i (0..3)
    
    The 7 Strassen products:
    M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    M4 = A11(B10-B00) -> C00-=M4, C01+=M4
    M5 = (A00+A01)B11 -> C00+=M5
    M6 = (A10-A00)(B00+B01) -> C11+=M6
    M7 = (A01-A11)(B10+B11) -> C00-=M7, C10+=M7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=(0,0)->idx0, A11=(1,1)->idx3
    # V: B00=(0,0)->idx0, B11=(1,1)->idx3
    # W: C00=(0,0)->idx0, C11=(1,1)->idx3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # U: A10=(1,0)->idx2, A11=(1,1)->idx3
    # V: B00=(0,0)->idx0
    # W: C10=(0,1)->idx2, C11=(1,1)->idx3
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    # U: A00=(0,0)->idx0
    # V: B01=(0,1)->idx1, B11=(1,1)->idx3
    # W: C01=(1,0)->idx1, C11=(1,1)->idx3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    # U: A11=(1,1)->idx3
    # V: B10=(1,0)->idx2, B00=(0,0)->idx0
    # W: C00=(0,0)->idx0, C01=(1,0)->idx1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    # U: A00=(0,0)->idx0, A01=(0,1)->idx1
    # V: B11=(1,1)->idx3
    # W: C00=(0,0)->idx0
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10=(1,0)->idx2, A00=(0,0)->idx0
    # V: B00=(0,0)->idx0, B01=(0,1)->idx1
    # W: C11=(1,1)->idx3
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01=(0,1)->idx1, A11=(1,1)->idx3
    # V: B10=(1,0)->idx2, B11=(1,1)->idx3
    # W: C00=(0,0)->idx0, C10=(0,1)->idx2
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[2][6] = 1
    
    return U, V, W

def _build_3x4x5_decomposition():
    """
    Build a rank-56 decomposition for 3x4x5 matrix multiplication.
    
    Strategy: Split into blocks and use Strassen where beneficial.
    
    Partition:
    - i: {0,1} | {2}  (sizes 2, 1)
    - j: {0,1} | {2,3}  (sizes 2, 2)
    - k: {0,1} | {2,3} | {4}  (sizes 2, 2, 1)
    
    This gives 2*2*3 = 12 blocks:
    - 4 blocks of size 2x2x2 -> Strassen rank 7 each = 28
    - 2 blocks of size 2x2x1 -> schoolbook rank 4 each = 8
    - 4 blocks of size 1x2x2 -> schoolbook rank 4 each = 16
    - 2 blocks of size 1x2x1 -> schoolbook rank 2 each = 4
    
    Total: 28+8+16+4 = 56
    """
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0]*rank for _ in range(n*m)]  # 12 x 56
    V = [[0]*rank for _ in range(m*p)]  # 20 x 56
    W = [[0]*rank for _ in range(p*n)]  # 15 x 56
    
    col = 0
    
    # Get Strassen 2x2x2 factors
    sU, sV, sW = _build_strassen_2x2x2()
    
    # Define partitions
    i_parts = [[0, 1], [2]]
    j_parts = [[0, 1], [2, 3]]
    k_parts = [[0, 1], [2, 3], [4]]
    
    for i_idx, i_list in enumerate(i_parts):
        for j_idx, j_list in enumerate(j_parts):
            for k_idx, k_list in enumerate(k_parts):
                i_size = len(i_list)
                j_size = len(j_list)
                k_size = len(k_list)
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    for c in range(7):
                        for ii, i in enumerate(i_list):
                            for jj, j in enumerate(j_list):
                                u_val = sU[ii*2+jj][c]
                                if u_val != 0:
                                    U[i*m + j][col] += u_val
                        for jj, j in enumerate(j_list):
                            for kk, k in enumerate(k_list):
                                v_val = sV[jj*2+kk][c]
                                if v_val != 0:
                                    V[j*p + k][col] += v_val
                        for kk, k in enumerate(k_list):
                            for ii, i in enumerate(i_list):
                                w_val = sW[kk*2+ii][c]
                                if w_val != 0:
                                    W[k*n + i][col] += w_val
                        col += 1
                else:
                    # Schoolbook
                    for i in i_list:
                        for j in j_list:
                            for k in k_list:
                                U[i*m + j][col] = 1
                                V[j*p + k][col] = 1
                                W[k*n + i][col] = 1
                                col += 1
    
    if col != rank:
        raise ValueError(f"Column count {col} != expected rank {rank}")
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Start with schoolbook as fallback
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try the rank-58 Strassen+schoolbook decomposition
    if time.time() < deadline - 5:
        try:
            result = _build_3x4x5_decomposition()
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = 56
            else:
                # If verification fails, keep schoolbook
                pass
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
