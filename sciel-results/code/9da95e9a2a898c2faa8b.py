"""Strassen-based rank-99 decomposition for 3x5 by 5x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction as F

DIMENSIONS = [3, 5, 7]


def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """Check that the factor matrices define the matrix multiplication tensor."""
    n_local, m_local, p_local = n, m, p
    rank = len(U[0])

    # Check: for all i, a, b, k: sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] == (a==b)
    for i in range(n_local):
        for a in range(m_local):
            for b in range(m_local):
                for k in range(p_local):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n_local + i][c] * U[i * m_local + a][c] * V[b * p_local + k][c]
                    expected = 1.0 if a == b else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is a dict with keys 'U', 'V', 'W' where each is a length-4 list.
    U indexed by (i, j) in {0,1}x{0,1} -> index i*2+j
    V indexed by (j, k) in {0,1}x{0,1} -> index j*2+k
    W indexed by (k, i) in {0,1}x{0,1} -> index k*2+i
    """
    # Strassen products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M6

    terms = []

    # M0 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U = [1, 0, 0, 1] (indices 00,01,10,11)
    # V: B00=1, B11=1 -> V = [1, 0, 0, 1] (indices 00,01,10,11)
    # W: C00 += 1, C11 += 1 -> W = [1, 0, 0, 1] (indices k*2+i: 00,01,10,11)
    terms.append({'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]})

    # M1 = (A10+A11)B00
    # U: A10=1, A11=1 -> U = [0, 0, 1, 1]
    # V: B00=1 -> V = [1, 0, 0, 0]
    # W: C10 += 1 -> W index k*2+i = 1*2+0 = 2 -> W = [0, 0, 1, 0]
    terms.append({'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 0, 1, 0]})

    # M2 = A00(B01-B11)
    # U: A00=1 -> U = [1, 0, 0, 0]
    # V: B01=1, B11=-1 -> V = [0, 1, 0, -1]
    # W: C01 += 1, C11 += -1 -> W index 01=1, 11=3 -> W = [0, 1, 0, -1]
    terms.append({'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 1, 0, -1]})

    # M3 = A11(B10-B00)
    # U: A11=1 -> U = [0, 0, 0, 1]
    # V: B10=1, B00=-1 -> V = [-1, 0, 1, 0]
    # W: C10 += 1, C00 += -1 -> W index 10=2, 00=0 -> W = [-1, 0, 1, 0]
    terms.append({'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [-1, 0, 1, 0]})

    # M4 = (A00+A01)B11
    # U: A00=1, A01=1 -> U = [1, 1, 0, 0]
    # V: B11=1 -> V = [0, 0, 0, 1]
    # W: C01 += 1 -> W index 01=1 -> W = [0, 1, 0, 0]
    terms.append({'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [0, 1, 0, 0]})

    # M5 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> U = [-1, 0, 1, 0]
    # V: B00=1, B01=1 -> V = [1, 1, 0, 0]
    # W: C00 += 1 -> W index 00=0 -> W = [1, 0, 0, 0]
    terms.append({'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [1, 0, 0, 0]})

    # M6 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 -> U = [0, 1, 0, -1]
    # V: B10=1, B11=1 -> V = [0, 0, 1, 1]
    # W: C11 += 1 -> W index 11=3 -> W = [0, 0, 0, 1]
    terms.append({'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [0, 0, 0, 1]})

    return terms


def _block_decomposition(n, m, p):
    """
    Construct a decomposition using block Strassen for 2x2x2 sub-blocks.
    
    Strategy: Partition the dimensions into blocks, and for each 2x2x2 block of
    (I_block, J_block, K_block) use Strassen (rank 7) instead of schoolbook (rank 8).
    For remaining 1-wide slices, use schoolbook (rank 1 each).
    """
    
    # Define block partitions
    I_blocks = [(0, 2), (2, 1)]  # I0: indices 0,1 (size 2); I1: index 2 (size 1)
    J_blocks = [(0, 2), (2, 2), (4, 1)]  # J0: 0,1; J1: 2,3; J2: 4
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]  # K0: 0,1; K1: 2,3; K2: 4,5; K3: 6
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _strassen_2x2x2_terms()
    
    for (i_start, i_size) in I_blocks:
        for (j_start, j_size) in J_blocks:
            for (k_start, k_size) in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen for this 2x2x2 block
                    for term in strassen_terms:
                        for li in range(2):
                            for lj in range(2):
                                u_val = term['U'][li * 2 + lj]
                                if u_val != 0:
                                    i_global = i_start + li
                                    j_global = j_start + lj
                                    U[i_global * m + j_global][col] = u_val
                        
                        for lj in range(2):
                            for lk in range(2):
                                v_val = term['V'][lj * 2 + lk]
                                if v_val != 0:
                                    j_global = j_start + lj
                                    k_global = k_start + lk
                                    V[j_global * p + k_global][col] = v_val
                        
                        for lk in range(2):
                            for li in range(2):
                                w_val = term['W'][lk * 2 + li]
                                if w_val != 0:
                                    k_global = k_start + lk
                                    i_global = i_start + li
                                    W[k_global * n + i_global][col] = w_val
                        
                        col += 1
                else:
                    # Use schoolbook for this block
                    for li in range(i_size):
                        for lj in range(j_size):
                            for lk in range(k_size):
                                i_global = i_start + li
                                j_global = j_start + lj
                                k_global = k_start + lk
                                U[i_global * m + j_global][col] = 1
                                V[j_global * p + k_global][col] = 1
                                W[k_global * n + i_global][col] = 1
                                col += 1
    
    # Trim to actual rank used
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
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


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, leaving 20s margin
    
    n, m, p = 3, 5, 7
    
    # Try block decomposition with Strassen for 2x2x2 blocks
    result = _block_decomposition(n, m, p)
    
    # Validate the decomposition
    if _validate_decomposition(result['U'], result['V'], result['W'], n, m, p):
        rank = len(result['U'][0])
        return result
    
    # Fallback to schoolbook if validation fails
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
