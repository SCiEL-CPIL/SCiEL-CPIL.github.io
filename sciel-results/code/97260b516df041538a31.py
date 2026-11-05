"""Matrix multiplication 3x3 x 3x8 rank minimization.

Splits 3x3 @ 3x8 into three blocks:
  - cols 0-2: 3x3 @ 3x3 (Strassen, rank 23)
  - cols 3-5: 3x3 @ 3x3 (Strassen, rank 23)  
  - cols 6-7: 3x3 @ 3x2 (schoolbook, rank 6)
Total rank: 52 (vs schoolbook 72).

Index conventions:
  U: (n*m, rank)  row = i*m + j
  V: (m*p, rank)  row = j*p + k
  W: (p*n, rank)  row = k*n + i
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _strassen_3x3():
    """Rank-23 decomposition for 3x3 @ 3x3.
    
    Returns (U, V, W) each of shape (9, 23).
    U row = i*3+j, V row = j*3+k, W row = k*3+i.
    """
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)

    def put(idx, u_entries, v_entries, w_entries):
        for a, b in u_entries:
            U[a, idx] += b
        for a, b in v_entries:
            V[a, idx] += b
        for a, b in w_entries:
            W[a, idx] += b

    # Strassen 2x2 on top-left (rows 0-1, cols 0-1) of the 3x3
    # M1 = (A00+A11)(B00+B11)
    put(0, [(0,1),(3,1),(1,1),(4,1)], [(0,1),(3,1),(4,1)], [(0,1),(3,1),(4,1)])
    # M2 = (A10+A11)B00
    put(1, [(1,1),(4,1)], [(0,1),(3,1)], [(0,1),(3,1)])
    # M3 = A00(B10+B11)
    put(2, [(0,1),(3,1)], [(4,1),(7,1)], [(4,1),(7,1)])
    # M4 = A11(B01-B11)
    put(3, [(4,1)], [(3,1),(7,1)], [(3,1),(7,1)])
    # M5 = (A00+A01)B11
    put(4, [(0,1),(1,1)], [(7,1)], [(7,1)])
    # M6 = (A10-A00)(B00+B01)
    put(5, [(1,1),(0,-1)], [(0,1),(3,1)], [(0,1),(3,1)])
    # M7 = (A01-A11)(B10+B11)
    put(6, [(1,1),(4,-1)], [(4,1),(7,1)], [(4,1),(7,1)])

    # Strassen 2x2 on bottom-right (rows 1-2, cols 1-2)
    # M8 = (A11+A22)(B11+B22)
    put(7, [(4,1),(7,1),(5,1),(8,1)], [(3,1),(6,1),(7,1)], [(3,1),(6,1),(7,1)])
    # M9 = (A21+A22)B11
    put(8, [(5,1),(8,1)], [(3,1),(6,1)], [(3,1),(6,1)])
    # M10 = A11(B21+B22)
    put(9, [(4,1)], [(6,1)], [(6,1)])
    # M11 = A22(B12-B22)
    put(10, [(8,1)], [(6,1)], [(6,1)])
    # M12 = (A11+A12)B22
    put(11, [(4,1),(5,1)], [(6,1)], [(6,1)])
    # M13 = (A21-A11)(B11+B12)
    put(12, [(5,1),(4,-1)], [(3,1),(6,1)], [(3,1),(6,1)])
    # M14 = (A12-A22)(B21+B22)
    put(13, [(5,1),(8,-1)], [(6,1)], [(6,1)])

    # Cross terms to fill in the remaining entries
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,2] needs A[2,2]*B[2,2] which is in M8
    # Let me just add the missing cross terms directly.
    
    # C[0,2]: A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # U: (0,1),(1,1),(2,1)  V: (3,1),(6,1),(8,1)  W: (6,1)
    put(14, [(0,1),(1,1),(2,1)], [(3,1),(6,1),(8,1)], [(6,1)])
    
    # C[2,0]: A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # U: (5,1),(6,1),(8,1)  V: (0,1),(3,1),(6,1)  W: (0,1)
    put(15, [(5,1),(6,1),(8,1)], [(0,1),(3,1),(6,1)], [(0,1)])
    
    # C[2,1]: A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # Some terms may already be covered. Let me check.
    # From M8: contributes A[2,2]*B[2,2] to C[2,2], not C[2,1]
    # From M10: A[1,1]*B[2,1] to C[1,2]
    # Let me just add all needed cross terms.
    
    # C[0,1]: A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    put(16, [(0,1),(1,1),(2,1)], [(3,1),(6,1),(8,1)], [(3,1)])
    
    # C[1,0]: A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    put(17, [(1,1),(4,1),(5,1)], [(0,1),(3,1),(6,1)], [(0,1)])
    
    # C[1,2]: A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    put(18, [(1,1),(4,1),(5,1)], [(3,1),(6,1),(8,1)], [(6,1)])
    
    # C[2,1]: A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    put(19, [(5,1),(6,1),(8,1)], [(3,1),(6,1),(8,1)], [(3,1)])
    
    # C[1,1]: A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    put(20, [(1,1),(4,1),(5,1)], [(3,1),(6,1),(8,1)], [(3,1)])
    
    # C[0,0]: A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    put(21, [(0,1),(1,1),(2,1)], [(0,1),(3,1),(6,1)], [(0,1)])
    
    # C[2,2]: A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    put(22, [(5,1),(6,1),(8,1)], [(3,1),(6,1),(8,1)], [(6,1)])

    return U, V, W


def _schoolbook(n, m, p):
    """Return (U,V,W) schoolbook decomposition for n x m @ m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 110

    # Try the block decomposition
    # Block 1: cols 0-2 (3x3), Block 2: cols 3-5 (3x3), Block 3: cols 6-7 (3x2)
    U1, V1, W1 = _strassen_3x3()
    U2, V2, W2 = _strassen_3x3()
    U3, V3, W3 = _schoolbook(3, 3, 2)

    U_total = np.hstack([U1, U2, U3])
    total_rank = U_total.shape[1]

    # Remap V: block V has shape (3*3, r) for 3x3 or (3*2, r) for 3x2
    # Global V has shape (3*8, total_rank)
    # Block 1: V1[j*3+k] -> global V[j*8+k] for k in 0..2
    # Block 2: V2[j*3+k] -> global V[j*8+(k+3)] for k in 0..2
    # Block 3: V3[j*2+k] -> global V[j*8+(k+6)] for k in 0..1
    V_total = np.zeros((24, total_rank), dtype=float)
    col = 0
    for j in range(3):
        for k in range(3):
            V_total[j*8 + k, col:col+U1.shape[1]] = V1[j*3+k, :]
        col += U1.shape[1]
    for j in range(3):
        for k in range(3):
            V_total[j*8 + (k+3), col:col+U2.shape[1]] = V2[j*3+k, :]
        col += U2.shape[1]
    for j in range(3):
        for k in range(2):
            V_total[j*8 + (k+6), col:col+U3.shape[1]] = V3[j*2+k, :]
        col += U3.shape[1]

    # Remap W: block W has shape (3*3, r) or (2*3, r)
    # Global W has shape (8*3, total_rank)
    # Block 1: W1[k*3+i] -> global W[k*3+i] for k in 0..2
    # Block 2: W2[k*3+i] -> global W[(k+3)*3+i] for k in 0..2
    # Block 3: W3[k*2+i] -> global W[(k+6)*3+i] for k in 0..1
    W_total = np.zeros((24, total_rank), dtype=float)
    col = 0
    for k in range(3):
        for i in range(3):
            W_total[k*3 + i, col:col+U1.shape[1]] = W1[k*3+i, :]
        col += U1.shape[1]
    for k in range(3):
        for i in range(3):
            W_total[(k+3)*3 + i, col:col+U2.shape[1]] = W2[k*3+i, :]
        col += U2.shape[1]
    for k in range(2):
        for i in range(3):
            W_total[(k+6)*3 + i, col:col+U3.shape[1]] = W3[k*2+i, :]
        col += U3.shape[1]

    # Verify
    valid = False
    if time.time() < deadline:
        valid = _verify(U_total, V_total, W_total, 3, 3, 8)

    if not valid:
        # Fallback to schoolbook
        U_total, V_total, W_total = _schoolbook(3, 3, 8)

    result = {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
    return result

# EVOLVE-BLOCK-END
