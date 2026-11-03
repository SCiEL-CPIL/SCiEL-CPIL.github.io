"""Strassen-based rank-98 decomposition for 4x4 * 4x8 -> 4x8.

Uses Strassen's 7-multiplication algorithm on 2x2 blocks recursively.
4x4x4: Rank 49 (7 Strassen multiplications of 2x2 blocks, each rank 7).
4x4x8: Two 4x4x4 blocks -> rank 98.
"""

# EVOLVE-BLOCK-START

import time
import numpy as np


def _strassen_2x2():
    """Return (U, V, W) for 2x2 x 2x2 -> 2x2 Strassen decomposition.
    
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1
    
    # M3 = A00(B10-B11)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] -= 1
    
    # M4 = A11(B01-B10)
    U[3, 3] += 1
    V[1, 3] += 1; V[2, 3] -= 1
    W[0, 3] -= 1; W[1, 3] += 1
    
    # M5 = (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[2, 5] += 1
    W[3, 5] += 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[1, 6] += 1; V[3, 6] += 1
    W[0, 6] -= 1; W[2, 6] += 1; W[3, 6] -= 1
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4 * 4x4 -> 4x4 using Strassen on 2x2 blocks."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_2x2()
    
    # Strassen bilinear forms for 4x4 blocks of 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Coefficients for A blocks in each M
    # A blocks: A00 (top-left), A01 (top-right), A10 (bottom-left), A11 (bottom-right)
    # Indices in U (16x49): row = i*4+j. 
    # A00: i in 0..1, j in 0..1 -> rows 0,1,4,5
    # A01: i in 0..1, j in 2..3 -> rows 2,3,6,7
    # A10: i in 2..3, j in 0..1 -> rows 8,9,12,13
    # A11: i in 2..3, j in 2..3 -> rows 10,11,14,15
    
    # Coefficients for B blocks in each M
    # B blocks: B00 (top-left), B01 (top-right), B10 (bottom-left), B11 (bottom-right)
    # Indices in V (16x49): row = j*4+k.
    # B00: j in 0..1, k in 0..1 -> rows 0,1,4,5
    # B01: j in 0..1, k in 2..3 -> rows 2,3,6,7
    # B10: j in 2..3, k in 0..1 -> rows 8,9,12,13
    # B11: j in 2..3, k in 2..3 -> rows 10,11,14,15
    
    # Coefficients for C blocks in each M
    # C blocks: C00, C01, C10, C11
    # Indices in W (16x49): row = k*4+i.
    # C00: k in 0..1, i in 0..1 -> rows 0,1,4,5
    # C01: k in 0..1, i in 2..3 -> rows 2,3,6,7
    # C10: k in 2..3, i in 0..1 -> rows 8,9,12,13
    # C11: k in 2..3, i in 2..3 -> rows 10,11,14,15
    
    # Define the 7 Strassen steps
    # Each step has: A_coeffs (dict block->val), B_coeffs (dict block->val), C_coeffs (dict block->val)
    
    steps = [
        # M1
        {'A': {'00': 1, '11': 1}, 'B': {'00': 1, '11': 1}, 'C': {'00': 1, '11': 1}},
        # M2
        {'A': {'10': 1, '11': 1}, 'B': {'00': 1}, 'C': {'10': 1}},
        # M3
        {'A': {'00': 1}, 'B': {'10': 1, '11': -1}, 'C': {'01': 1, '11': -1}},
        # M4
        {'A': {'11': 1}, 'B': {'01': 1, '10': -1}, 'C': {'00': -1, '01': 1}},
        # M5
        {'A': {'00': 1, '01': 1}, 'B': {'11': 1}, 'C': {'00': 1}},
        # M6
        {'A': {'10': 1, '00': -1}, 'B': {'00': 1, '10': 1}, 'C': {'11': 1}},
        # M7
        {'A': {'01': 1, '11': -1}, 'B': {'01': 1, '11': 1}, 'C': {'00': -1, '10': 1, '11': -1}},
    ]
    
    # Block row ranges in the 4x4 matrices
    # A[i,j] block 'xy' where x is row block (0,1), y is col block (0,1)
    # i in 2x..2x+1, j in 2y..2y+1
    # U row index: i*4 + j
    
    for s_idx, step in enumerate(steps):
        base_col = s_idx * 7
        for c in range(7):
            col = base_col + c
            
            # Fill U
            for block, val in step['A'].items():
                if val == 0: continue
                x, y = int(block[0]), int(block[1])
                for di in range(2):
                    for dj in range(2):
                        i = 2*x + di
                        j = 2*y + dj
                        u_row = i*4 + j
                        # U2 index: di*2+dj
                        u_val = U2[di*2+dj, c]
                        if u_val != 0:
                            U[u_row, col] += val * u_val
            
            # Fill V
            for block, val in step['B'].items():
                if val == 0: continue
                x, y = int(block[0]), int(block[1])
                for dj in range(2):
                    for dk in range(2):
                        j = 2*x + dj
                        k = 2*y + dk
                        v_row = j*4 + k
                        v_val = V2[dj*2+dk, c]
                        if v_val != 0:
                            V[v_row, col] += val * v_val
            
            # Fill W
            for block, val in step['C'].items():
                if val == 0: continue
                x, y = int(block[0]), int(block[1])
                for dk in range(2):
                    for di in range(2):
                        k = 2*x + dk
                        i = 2*y + di
                        w_row = k*4 + i
                        w_val = W2[dk*2+di, c]
                        if w_val != 0:
                            W[w_row, col] += val * w_val
    
    return U, V, W


def _build_4x4x8_strassen():
    """Build rank-98 decomposition for 4x4 * 4x8 -> 4x8."""
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # First half: B cols 0-3, C rows 0-3
    # U is same
    U[:, 0:49] = U4
    # V: B rows j, cols 0-3. V4 rows are j*4+k. Global V rows are j*8+k.
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
    # W: C rows 0-3, cols 0-3. W4 rows are k*4+i. Global W rows are k*4+i.
    # W shape (32, 98). Rows k*4+i for k in 0..3, i in 0..3 are 0..15.
    W[0:16, 0:49] = W4
    
    # Second half: B cols 4-7, C rows 4-7
    U[:, 49:98] = U4
    # V: B rows j, cols 4-7. Global V rows are j*8 + (4+k).
    for j in range(4):
        for k in range(4):
            V[j*8+4+k, 49:98] = V4[j*4+k, :]
    # W: C rows 4-7, cols 0-3. Global W rows are (4+k)*4+i.
    for k in range(4):
        for i in range(4):
            W[(4+k)*4+i, 49:98] = W4[k*4+i, :]
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False, (i, j, k, s)
    return True, None


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 8
    
    # Try Strassen rank-98
    try:
        U, V, W = _build_4x4x8_strassen()
        ok, err = _verify(U, V, W, n, m, p)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            print(f"Strassen verification failed: {err}")
    except Exception as e:
        print(f"Strassen construction failed: {e}")
    
    # Fallback to schoolbook
    try:
        U, V, W = _schoolbook(n, m, p)
        ok, err = _verify(U, V, W, n, m, p)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        print(f"Schoolbook failed: {e}")
    
    # Last resort: return schoolbook without verification
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
