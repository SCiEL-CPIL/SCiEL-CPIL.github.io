"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
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

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 Strassen terms as (u_dict, v_dict, w_dict) tuples.
    
    U indexed by (i,j): A[i,j]
    V indexed by (j,k): B[j,k]
    W indexed by (k,i): contributes to C[i,k]
    
    M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    M2 = (A10+A11)B00        -> C10+=M2, C11-=M2
    M3 = A00(B10-B11)        -> C01+=M3, C11+=M3
    M4 = A11(B01-B00)        -> C00+=M4, C10+=M4
    M5 = (A00+A01)B11        -> C00-=M5, C01+=M5
    M6 = (A10-A00)(B00+B01)  -> C11+=M6
    M7 = (A01-A11)(B10+B11)  -> C00+=M7
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2 = (A10+A11)B00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1}
    ))
    # M3 = A00(B10-B11)
    terms.append((
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1}
    ))
    # M4 = A11(B01-B00)
    terms.append((
        {(1,1): 1},
        {(0,1): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1}
    ))
    # M5 = (A00+A01)B11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (1,0): 1}
    ))
    # M6 = (A10-A00)(B00+B01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7 = (A01-A11)(B10+B11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _build_kronecker_strassen_4x4x4():
    """Build 4x4 * 4x4 -> 4x4 using Kronecker product of Strassen (rank 49).
    
    The 4x4x4 multiplication tensor is the Kronecker product of two 2x2x2 tensors.
    If each 2x2x2 has rank 7, the Kronecker product has rank 7*7 = 49.
    
    A (4x4) indexed by (i1,i2,j1,j2) where i1,j1 in {0,1} (block row/col) and i2,j2 in {0,1} (within-block)
    B (4x4) indexed by (j1,j2,k1,k2)
    C (4x4) indexed by (i1,i2,k1,k2)
    
    C[i1,i2,k1,k2] = sum_{j1,j2} A[i1,i2,j1,j2] * B[j1,j2,k1,k2]
    
    Using Kronecker: for each pair of Strassen terms (a,b), we get one rank-1 term
    where:
    U[(i1,i2,j1,j2)] = U_a[i1,j1] * U_b[i2,j2]
    V[(j1,j2,k1,k2)] = V_a[j1,k1] * V_b[j2,k2]
    W[(k1,k2,i1,i2)] = W_a[k1,i1] * W_b[k2,i2]
    
    Global indices:
    A row = i1*2 + i2, col = j1*2 + j2 -> U index = (i1*2+i2)*4 + (j1*2+j2)
    B row = j1*2 + j2, col = k1*2 + k2 -> V index = (j1*2+j2)*4 + (k1*2+k2)
    C row = i1*2 + i2, col = k1*2 + k2 -> W index = (k1*2+k2)*4 + (i1*2+i2)
    """
    n, m, p = 4, 4, 4
    s_terms = _strassen_2x2_terms()
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    col = 0
    for ua, va, wa in s_terms:
        for ub, vb, wb in s_terms:
            # U: for each (i1,j1) in ua and (i2,j2) in ub
            for (i1, j1), ca in ua.items():
                for (i2, j2), cb in ub.items():
                    u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    U[u_idx][col] += ca * cb
            # V: for each (j1,k1) in va and (j2,k2) in vb
            for (j1, k1), ca in va.items():
                for (j2, k2), cb in vb.items():
                    v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                    V[v_idx][col] += ca * cb
            # W: for each (k1,i1) in wa and (k2,i2) in wb
            for (k1, i1), ca in wa.items():
                for (k2, i2), cb in wb.items():
                    w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                    W[w_idx][col] += ca * cb
            col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    return {"U": U, "V": V, "W": W}

def _build_4x4x5_kronecker():
    """Build 4x4 * 4x5 -> 4x5 using Kronecker Strassen for 4x4x4 part + rank-1 for last column.
    
    Split B (4x5) into B_left (4x4) and B_right (4x1, the 5th column).
    C = A*B = [A*B_left (4x4) | A*B_right (4x1)]
    
    A*B_left: rank 49 via Kronecker Strassen
    A*B_right: rank 1 (since T[i,j,4] = 1 for all i,j, which is rank 1:
               U[i*4+j] = 1, V[j*5+4] = 1, W[4*4+i] = 1 for all i,j)
    
    Total rank: 49 + 1 = 50
    """
    n, m, p = 4, 4, 5
    
    # Build the 4x4x4 part (rank 49)
    kronecker = _build_kronecker_strassen_4x4x4()
    rank = 49 + 1  # 50
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 50
    V = [[0] * rank for _ in range(m * p)]  # 20 x 50
    W = [[0] * rank for _ in range(p * n)]  # 20 x 50
    
    # Copy the Kronecker Strassen terms (first 49 columns)
    for r in range(49):
        for i in range(16):
            U[i][r] = kronecker["U"][i][r]
        for j in range(16):
            V[j][r] = kronecker["V"][j][r]
        for k in range(16):
            W[k][r] = kronecker["W"][k][r]
    
    # Add the rank-1 term for the 5th column (index 4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # T[i,j,4] = 1 for all i,j
    # U[i*4+j] = 1, V[j*5+4] = 1, W[4*4+i] = 1
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][49] = 1
            V[j * 5 + 4][49] = 1
            W[4 * 4 + i][49] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary approach: Kronecker Strassen + rank-1 last column (rank 50)
    try:
        result = _build_4x4x5_kronecker()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 80)
    return _build_schoolbook()
# EVOLVE-BLOCK-END
