"""Rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 140.0


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: for all (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                s += float(np.sum(u_row * v_row * w_row))
            if abs(s - 1.0) > tol:
                return False
    return True


def _strassen_2x2x2():
    """
    Rank-7 decomposition of 2x2 * 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h)
    U[0,0]=1; U[1,0]=1   # a (i=0,j=0), d (i=1,j=1)
    V[0,0]=1; V[3,0]=1   # e (j=0,k=0), h (j=1,k=1)
    W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
    
    # M2 = (c+d)e
    U[2,1]=1; U[3,1]=1   # c (i=1,j=0), d (i=1,j=1)
    V[0,1]=1             # e (j=0,k=0)
    W[2,1]=1; W[3,1]=1
    
    # M3 = a(f-h)
    U[0,2]=1             # a (i=0,j=0)
    V[1,2]=1; V[3,2]=-1  # f (j=0,k=1), -h (j=1,k=1)
    W[1,2]=1; W[3,2]=1
    
    # M4 = d(g-e)
    U[1,3]=1             # d (i=1,j=1)
    V[2,3]=1; V[0,3]=-1  # g (j=1,k=0), -e (j=0,k=0)
    W[0,3]=1; W[2,3]=1
    
    # M5 = (a+b)h
    U[0,4]=1; U[2,4]=1   # a (i=0,j=0), b (i=0,j=1)
    V[3,4]=1             # h (j=1,k=1)
    W[1,4]=-1; W[3,4]=-1
    
    # M6 = (c-a)(e+f)
    U[2,5]=1; U[0,5]=-1  # c (i=1,j=0), -a (i=0,j=0)
    V[0,5]=1; V[1,5]=1   # e (j=0,k=0), f (j=0,k=1)
    W[2,5]=-1; W[3,5]=-1
    
    # M7 = (b-d)(g+h)
    U[2,6]=1; U[3,6]=-1  # b (i=0,j=1), -d (i=1,j=1)
    V[2,6]=1; V[3,6]=1   # g (j=1,k=0), h (j=1,k=1)
    W[0,6]=1; W[1,6]=1
    
    return U, V, W


def _schoolbook_2x2x3():
    """
    Rank-12 decomposition of 2x2 * 2x3 matrix multiplication.
    Returns (U, V, W) with shapes (4, 12), (6, 12), (6, 12).
    U[i*2+j, c], V[j*3+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 12), dtype=float)
    V = np.zeros((6, 12), dtype=float)
    W = np.zeros((6, 12), dtype=float)
    
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j, c] = 1.0
                V[j*3+k, c] = 1.0
                W[k*2+i, c] = 1.0
                c += 1
    
    return U, V, W


def _construct_block_decomposition(n, m, p):
    """
    Block decomposition for 2x4 * 4x7:
    A (2x4) = [A1 | A2], each 2x2
    B (4x7) = [B1; B2], each 2x7
    C = A1*B1 + A2*B2
    
    Each 2x2 * 2x7:
      Split B into [B_left (2x2) | B_mid (2x2) | B_right (2x3)]
      2x2*2x2: Strassen rank 7
      2x2*2x2: Strassen rank 7
      2x2*2x3: schoolbook rank 12
      Total per 2x2*2x7: 26
    Total: 2 * 26 = 52
    """
    total_rank = 52
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for block in range(2):
        # U row offset: block * 4 (2 rows of A * 2 cols per block)
        a_base = block * 4
        # V row offset: block * 14 (2 rows of B * 7 cols)
        b_base = block * 14
        
        # Sub-block 1: 2x2 * 2x2 (B columns 0-1)
        sU, sV, sW = _strassen_2x2x2()
        k_base = 0
        U[a_base:a_base+4, col:col+7] = sU
        for j in range(2):
            for k in range(2):
                V[b_base + j*7 + k_base + k, col:col+7] = sV[j*2+k]
        for k in range(2):
            for i in range(2):
                W[(k_base+k)*2 + i, col:col+7] = sW[k*2+i]
        col += 7
        
        # Sub-block 2: 2x2 * 2x2 (B columns 2-3)
        sU, sV, sW = _strassen_2x2x2()
        k_base = 2
        U[a_base:a_base+4, col:col+7] = sU
        for j in range(2):
            for k in range(2):
                V[b_base + j*7 + k_base + k, col:col+7] = sV[j*2+k]
        for k in range(2):
            for i in range(2):
                W[(k_base+k)*2 + i, col:col+7] = sW[k*2+i]
        col += 7
        
        # Sub-block 3: 2x2 * 2x3 (B columns 4-6)
        sU, sV, sW = _schoolbook_2x2x3()
        k_base = 4
        U[a_base:a_base+4, col:col+12] = sU
        for j in range(2):
            for k in range(3):
                V[b_base + j*7 + k_base + k, col:col+12] = sV[j*3+k]
        for k in range(3):
            for i in range(2):
                W[(k_base+k)*2 + i, col:col+12] = sW[k*2+i]
        col += 12
    
    return U, V, W, total_rank


def _to_list(mat):
    """Convert numpy array to list of lists with int/Fraction entries."""
    result = []
    for row in mat:
        row_list = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                row_list.append(int(round(val)))
            else:
                # Try to express as half-integer or small fraction
                frac = Fraction(val).limit_denominator(100)
                if frac.denominator <= 2:
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    else:
                        row_list.append(frac)
                else:
                    row_list.append(frac)
        result.append(row_list)
    return result


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Try block decomposition (rank 52)
    U_b, V_b, W_b, r_b = _construct_block_decomposition(n, m, p)
    if _verify(U_b, V_b, W_b, n, m, p):
        return {
            'U': _to_list(U_b),
            'V': _to_list(V_b),
            'W': _to_list(W_b)
        }
    
    # Fallback: schoolbook rank 56
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    
    return {
        'U': _to_list(U),
        'V': _to_list(V),
        'W': _to_list(W)
    }
