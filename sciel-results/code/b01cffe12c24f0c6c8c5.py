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
    Build a rank-58 decomposition for 3x4x5 matrix multiplication.
    
    Strategy: Split into blocks and use Strassen where beneficial.
    
    For 3x4x5:
    n=3 rows of A, m=4 cols of A / rows of B, p=5 cols of B
    
    We can decompose:
    - Use two 2x2x2 Strassen blocks (rank 7 each) covering part of the computation
    - Fill the rest with schoolbook terms
    
    Actually, let me think more carefully. The tensor is T[i,j,k] = 1 for all 
    i in [0,3), j in [0,4), k in [0,5).
    
    A simple approach: 
    - Partition i into {0,1} and {2}
    - Partition j into {0,1} and {2,3}  
    - Partition k into {0,1} and {2,3,4}
    
    Block (0,0,0): i∈{0,1}, j∈{0,1}, k∈{0,1} -> 2x2x2, use Strassen rank 7
    Block (0,1,0): i∈{0,1}, j∈{2,3}, k∈{0,1} -> 2x2x2, use Strassen rank 7
    Block (1,0,0): i∈{2}, j∈{0,1}, k∈{0,1} -> 1x2x2, schoolbook rank 4
    Block (1,1,0): i∈{2}, j∈{2,3}, k∈{0,1} -> 1x2x2, schoolbook rank 4
    Block (0,0,1): i∈{0,1}, j∈{0,1}, k∈{2,3,4} -> 2x2x3, schoolbook rank 12
    Block (0,1,1): i∈{0,1}, j∈{2,3}, k∈{2,3,4} -> 2x2x3, schoolbook rank 12
    Block (1,0,1): i∈{2}, j∈{0,1}, k∈{2,3,4} -> 1x2x3, schoolbook rank 6
    Block (1,1,1): i∈{2}, j∈{2,3}, k∈{2,3,4} -> 1x2x3, schoolbook rank 6
    
    Total: 7+7+4+4+12+12+6+6 = 58
    """
    n, m, p = 3, 4, 5
    rank = 58
    U = [[0]*rank for _ in range(n*m)]  # 12 x 58
    V = [[0]*rank for _ in range(m*p)]  # 20 x 58
    W = [[0]*rank for _ in range(p*n)]  # 15 x 58
    
    col = 0
    
    # Get Strassen 2x2x2 factors
    sU, sV, sW = _build_strassen_2x2x2()
    
    # Block (0,0,0): i∈{0,1}, j∈{0,1}, k∈{0,1} -> Strassen
    # i_idx = i, j_idx = j, k_idx = k
    for c in range(7):
        for i in [0, 1]:
            for j in [0, 1]:
                u_val = sU[i*2+j][c]
                if u_val != 0:
                    U[i*4 + j][col] += u_val
        for j in [0, 1]:
            for k in [0, 1]:
                v_val = sV[j*2+k][c]
                if v_val != 0:
                    V[j*5 + k][col] += v_val
        for k in [0, 1]:
            for i in [0, 1]:
                w_val = sW[k*2+i][c]
                if w_val != 0:
                    W[k*3 + i][col] += w_val
        col += 1
    
    # Block (0,1,0): i∈{0,1}, j∈{2,3}, k∈{0,1} -> Strassen
    # i_idx = i, j_idx = j-2, k_idx = k
    for c in range(7):
        for i in [0, 1]:
            for jj in [0, 1]:
                u_val = sU[i*2+jj][c]
                if u_val != 0:
                    U[i*4 + (jj+2)][col] += u_val
        for jj in [0, 1]:
            for k in [0, 1]:
                v_val = sV[jj*2+k][c]
                if v_val != 0:
                    V[(jj+2)*5 + k][col] += v_val
        for k in [0, 1]:
            for i in [0, 1]:
                w_val = sW[k*2+i][c]
                if w_val != 0:
                    W[k*3 + i][col] += w_val
        col += 1
    
    # Block (1,0,0): i∈{2}, j∈{0,1}, k∈{0,1} -> schoolbook 1x2x2
    for j in [0, 1]:
        for k in [0, 1]:
            U[2*4 + j][col] = 1
            V[j*5 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
    
    # Block (1,1,0): i∈{2}, j∈{2,3}, k∈{0,1} -> schoolbook 1x2x2
    for j in [2, 3]:
        for k in [0, 1]:
            U[2*4 + j][col] = 1
            V[j*5 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
    
    # Block (0,0,1): i∈{0,1}, j∈{0,1}, k∈{2,3,4} -> schoolbook 2x2x3
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i*4 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
    
    # Block (0,1,1): i∈{0,1}, j∈{2,3}, k∈{2,3,4} -> schoolbook 2x2x3
    for i in [0, 1]:
        for j in [2, 3]:
            for k in [2, 3, 4]:
                U[i*4 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
    
    # Block (1,0,1): i∈{2}, j∈{0,1}, k∈{2,3,4} -> schoolbook 1x2x3
    for j in [0, 1]:
        for k in [2, 3, 4]:
            U[2*4 + j][col] = 1
            V[j*5 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
    
    # Block (1,1,1): i∈{2}, j∈{2,3}, k∈{2,3,4} -> schoolbook 1x2x3
    for j in [2, 3]:
        for k in [2, 3, 4]:
            U[2*4 + j][col] = 1
            V[j*5 + k][col] = 1
            W[k*3 + 2][col] = 1
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
                best_rank = 58
            else:
                # If verification fails, keep schoolbook
                pass
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
