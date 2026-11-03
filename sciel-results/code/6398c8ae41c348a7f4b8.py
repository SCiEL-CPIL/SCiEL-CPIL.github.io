"""Matrix multiplication 3x4x6: search for low-rank exact decomposition."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = delta
    # For each rank term c: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 (it's the identity in the middle dimension)
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is W[k*n+i, r], so W index is (k, i) -> k*n+i
    # U index is (i, j) -> i*m+j, shape (n*m, rank)
    # V index is (j, k) -> j*p+k, shape (m*p, rank)
    # W index is (k, i) -> k*n+i, shape (p*n, rank)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
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
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2.
    Rank 7. Returns U, V, W as lists of lists.
    Convention: U[i*m+j, c] for m=2, V[j*p+k, c] for p=2, W[k*n+i, c] for n=2.
    
    Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # n=2, m=2, p=2
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # a00 = U[0], a01 = U[1], a10 = U[2], a11 = U[3]
    # b00 = V[0], b01 = V[1], b10 = V[2], b11 = V[3]
    # C00 = W[0], C01 = W[1], C10 = W[2], C11 = W[3]
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00->1, a11->1; V: b00->1, b11->1; W: C00->1, C11->1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _combine_blocks(block_list, n, m, p):
    """
    Combine block decompositions. Each block is a dict with 'U','V','W' 
    and 'offset' info for where it applies.
    This is simplified - we'll handle specific cases.
    """
    pass

def _try_strassen_blocks(n, m, p):
    """
    Try to use Strassen for 2x2 subblocks and schoolbook for the rest.
    For 3x4*4x6:
    - Split rows into 2+1, cols into 2+2
    - Block (0:2, 0:2): 2x2*2x2 -> Strassen rank 7
    - Block (0:2, 2:4): 2x2*2x2 -> Strassen rank 7  
    - Block (2:3, 0:2): 1x2*2x2 -> rank 4 (schoolbook for 1x2*2x2 is 1*2*2=4)
    - Block (2:3, 2:4): 1x2*2x2 -> rank 4
    Total: 7+7+4+4 = 22
    
    But we need to be careful about how the blocks interact with the shared middle dimension.
    Actually, for block decomposition of matrix multiplication:
    If we partition A into row blocks and B into col blocks, the middle dimension (columns of A = rows of B) 
    must be consistent. The middle dimension is shared, so we can't independently decompose blocks
    unless we're careful.
    
    Actually, the standard block Strassen approach works when the middle dimension is also partitioned.
    For A (n x m) * B (m x p), if we partition m into m1+m2, then:
    C = A1*B1 + A2*B2
    where A1 is n x m1, B1 is m1 x p, A2 is n x m2, B2 is m2 x p.
    
    This gives rank = rank(n x m1 * m1 x p) + rank(n x m2 * m2 x p)
    
    For 3x4*4x6, partition m=4 into 2+2:
    C = A1(3x2)*B1(2x6) + A2(3x2)*B2(2x6)
    Each term is 3x2*2x6.
    
    For 3x2*2x6: partition n=3 into 2+1:
    (2x2*2x6) + (1x2*2x6)
    
    2x2*2x6: partition p=6 into 2+2+2:
    (2x2*2x2) + (2x2*2x2) + (2x2*2x2) -> 7+7+7 = 21? No wait.
    
    Actually for 2x2*2x6, we can partition p into three 2s:
    C1 = A*B1 where B1 is 2x2, C2 = A*B2, C3 = A*B3
    Each is 2x2*2x2 with rank 7 (Strassen).
    So 2x2*2x6 has rank 3*7 = 21? That's worse than schoolbook (2*2*6=24).
    Hmm, 21 < 24, so it's better.
    
    Wait, but we can also do better. Let me think about 2x2*2x6 differently.
    Actually, the rank of 2x2*2x6 is at most 2*2*6 = 24 (schoolbook).
    With Strassen on the 2x2 part: for each of the 6 columns of B, we do a 2x2*2x1 multiplication.
    2x2*2x1: this is just 4 multiplications (schoolbook), or can we do better?
    Rank of 2x2*2x1: the tensor is 2x2x1, which has rank... well it's just 4 independent entries, so rank 4.
    So 2x2*2x6 with column-wise Strassen doesn't help since the last dimension is 1.
    
    Let me reconsider. The key insight from the knowledge is:
    "Partition B columns as 2+2+2 instead of 3+3, enabling pure 2x2 Strassen blocks (rank 7) and 1x2*2x2 blocks (rank 4) with total rank 66 vs 68"
    
    Wait, that's for a different problem perhaps. Let me re-read: the knowledge says total rank 66 vs 68.
    But our schoolbook is 3*4*6 = 72. So 66 is better.
    
    Hmm, but how do we get 66? Let me think about this differently.
    
    Actually, I think the partitioning is on the middle dimension m=4.
    If we split m=4 into 2+2:
    C = A[:,0:2]*B[0:2,:] + A[:,2:4]*B[2:4,:]
    Each term is 3x2 * 2x6.
    
    For 3x2*2x6, we can split n=3 into 2+1:
    Term1 = A[0:2,:]*B (2x2*2x6)
    Term2 = A[2:3,:]*B (1x2*2x6)
    
    For 2x2*2x6: split p=6 into 2+2+2:
    Three blocks of 2x2*2x2, each with Strassen rank 7.
    Total: 3*7 = 21
    
    For 1x2*2x6: split p=6 into 2+2+2:
    Three blocks of 1x2*2x2.
    1x2*2x2: rank is 1*2*2 = 4 (schoolbook). Can we do better?
    The tensor is 1x2x2, which has 4 independent entries, so minimum rank is 4.
    Total: 3*4 = 12
    
    So one half (m=0:2) has rank 21+12 = 33.
    The other half (m=2:4) also has rank 33.
    Total: 66.
    
    This matches the knowledge! So the strategy is:
    - Split middle dimension m=4 into two groups of 2
    - For each group, split rows n=3 into 2+1
    - For the 2x2*2x6 part, split cols p=6 into three groups of 2, use Strassen (rank 7 each)
    - For the 1x2*2x6 part, split cols p=6 into three groups of 2, use schoolbook (rank 4 each)
    - Total: 2*(3*7 + 3*4) = 2*(21+12) = 66
    """
    n, m, p = 3, 4, 6
    
    # We'll build the full decomposition
    rank_total = 66
    U = [[0]*rank_total for _ in range(n*m)]  # 12 x 66
    V = [[0]*rank_total for _ in range(m*p)]  # 24 x 66
    W = [[0]*rank_total for _ in range(p*n)]  # 18 x 66
    
    col_offset = 0
    
    for mid_block in range(2):  # m split into [0:2] and [2:4]
        m_start = mid_block * 2
        m_end = m_start + 2
        
        for row_block in range(2):  # n split into [0:2] and [2:3]
            if row_block == 0:
                n_start, n_end = 0, 2
                is_strassen = True
            else:
                n_start, n_end = 2, 3
                is_strassen = False
            
            for col_block in range(3):  # p split into three groups of 2
                p_start = col_block * 2
                p_end = p_start + 2
                
                if is_strassen:
                    # 2x2*2x2 Strassen block
                    sU, sV, sW = _strassen_2x2()
                    for c in range(7):
                        for i in range(2):
                            for j in range(2):
                                U[(n_start+i)*m + (m_start+j)][col_offset+c] = sU[i*2+j][c]
                        for j in range(2):
                            for k in range(2):
                                V[(m_start+j)*p + (p_start+k)][col_offset+c] = sV[j*2+k][c]
                        for k in range(2):
                            for i in range(2):
                                W[(p_start+k)*n + (n_start+i)][col_offset+c] = sW[k*2+i][c]
                    col_offset += 7
                else:
                    # 1x2*2x2 schoolbook block
                    # n_block=1, m_block=2, p_block=2
                    for i in range(1):
                        for j in range(2):
                            for k in range(2):
                                U[(n_start+i)*m + (m_start+j)][col_offset] = 1
                                V[(m_start+j)*p + (p_start+k)][col_offset] = 1
                                W[(p_start+k)*n + (n_start+i)][col_offset] = 1
                                col_offset += 1
    
    if col_offset != rank_total:
        # Adjust
        U = [row[:col_offset] for row in U]
        V = [row[:col_offset] for row in V]
        W = [row[:col_offset] for row in W]
        rank_total = col_offset
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # First, try the Strassen-block decomposition
    try:
        result = _try_strassen_blocks(n, m, p)
        # Verify
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            # Exact check
            if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
