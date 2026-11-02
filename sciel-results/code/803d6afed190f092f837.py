"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """
    Returns the 7 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is a tuple (U_dict, V_dict, W_dict) where keys are (i,j), (j,k), (k,i).
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A21+A22)B11
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3 = A11(B21-B22)
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4 = A22(B12-B11)
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A11+A12)B22
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6 = (A21-A11)(B11+B12)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7 = (A12-A22)(B21+B22)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    return terms

def _build_4x4x4_strassen():
    """
    Builds the rank 49 decomposition for 4x4 @ 4x4 -> 4x4.
    Returns U, V, W as lists of columns (vectors).
    """
    n, m, p = 4, 4, 4
    U_cols = []
    V_cols = []
    W_cols = []
    
    s2_terms = _strassen_2x2_terms()
    
    def add_term_global(u_blocks, v_blocks, c_blocks):
        """
        u_blocks: list of (bi, bj)
        v_blocks: list of (bj, bk)
        c_blocks: list of (bi, bk, coeff)
        """
        for term in s2_terms:
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            for (li, lj), cu in term['U'].items():
                for (bi, bj) in u_blocks:
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    u_vec[gi*m + gj] += cu
                    
            for (lj, lk), cv in term['V'].items():
                for (bj, bk) in v_blocks:
                    gj = bj * 2 + lj
                    gk = bk * 2 + lk
                    v_vec[gj*p + gk] += cv
                    
            for (lk, li), cw in term['W'].items():
                for (bi, bk, coeff) in c_blocks:
                    gi = bi * 2 + li
                    gk = bk * 2 + lk
                    w_vec[gk*n + gi] += cw * coeff
                    
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    # M1 = (A00+A11)(B00+B11)
    add_term_global([(0,0), (1,1)], [(0,0), (1,1)], [(0,0,1), (1,1,1)])
    # M2 = (A10+A11)B00
    add_term_global([(1,0), (1,1)], [(0,0)], [(1,0,1), (1,1,-1)])
    # M3 = A00(B10-B11)
    add_term_global([(0,0)], [(1,0), (1,1)], [(0,1,1), (1,1,1)])
    # M4 = A11(B01-B00)
    add_term_global([(1,1)], [(0,0), (0,1)], [(0,0,1), (1,0,1)])
    # M5 = (A00+A01)B11
    add_term_global([(0,0), (0,1)], [(1,1)], [(0,0,-1), (0,1,1)])
    # M6 = (A10-A00)(B00+B01)
    add_term_global([(1,0), (0,0)], [(0,0), (0,1)], [(1,1,1)])
    # M7 = (A01-A11)(B10+B11)
    add_term_global([(0,1), (1,1)], [(1,0), (1,1)], [(0,0,1)])
    
    return U_cols, V_cols, W_cols

def _build_4x4x5():
    n, m, p = DIMENSIONS
    U_cols, V_cols, W_cols = _build_4x4x4_strassen()
    
    # The 5th column (k=4) of B: C[i,4] = sum_j A[i,j] * B[j,4]
    # Use 4 rank-1 terms (one per j) for the 4x4 @ 4x1 part.
    for j in range(m):
        u_vec = np.zeros(n*m)
        v_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        
        for i in range(n):
            u_vec[i*m + j] = 1
        v_vec[j*p + 4] = 1
        for i in range(n):
            w_vec[4*n + i] = 1
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    
    try:
        result = _build_4x4x5()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception as e:
        print(f"Error in new decomposition: {e}")
        
    # Fallback to schoolbook (rank 80)
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
# EVOLVE-BLOCK-END
