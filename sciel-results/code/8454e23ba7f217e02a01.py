"""Strassen-based rank-56 decomposition for 4x4 * 4x8 -> 4x8.

Strategy:
- 4x4 * 4x4 via Strassen (7 multiplications of 2x2 blocks), each 2x2*2x2 done
  with schoolbook (rank 4) -> rank 28 per 4x4x4.
- 4x4 * 4x8 split into two 4x4 * 4x4 sub-problems (B split into two 4x4 halves).
- Total rank: 2 * 28 = 56.

Convention: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
"""

import time
import numpy as np


def _strassen_2x2_schoolbook():
    """Return (U, V, W) for 2x2 x 2x2 -> 2x2 schoolbook decomposition.
    
    U shape: (4, 4), V shape: (4, 4), W shape: (4, 4)
    Convention: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k
    """
    n, m, p = 2, 2, 2
    rank = n * m * p  # 4
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


def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 * 4x4 -> 4x4 using Strassen with schoolbook inner.
    
    Outer: 7 Strassen multiplications of 2x2 blocks.
    Inner: each 2x2*2x2 uses schoolbook rank 4.
    Total: 7 * 4 = 28.
    """
    n, m, p = 4, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_2x2_schoolbook()
    
    # Strassen bilinear forms for 4x4 blocks of 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C reconstruction:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M7
    
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
    
    # Block indices in 4x4 matrices:
    # A block 'xy': x=row block (0,1), y=col block (0,1)
    # A[i,j] where i in 2x..2x+1, j in 2y..2y+1
    # U row index: i*4 + j
    # B block 'xy': x=row block, y=col block
    # V row index: j*4 + k
    # C block 'xy': x=row block, y=col block
    # W row index: k*4 + i
    
    for s_idx, step in enumerate(steps):
        base_col = s_idx * 4
        for c in range(4):
            col = base_col + c
            
            # Fill U
            for block, val in step['A'].items():
                if val == 0:
                    continue
                x, y = int(block[0]), int(block[1])
                for di in range(2):
                    for dj in range(2):
                        i = 2 * x + di
                        j = 2 * y + dj
                        u_row = i * 4 + j
                        u_val = U2[di * 2 + dj, c]
                        if u_val != 0:
                            U[u_row, col] += val * u_val
            
            # Fill V
            for block, val in step['B'].items():
                if val == 0:
                    continue
                x, y = int(block[0]), int(block[1])
                for dj in range(2):
                    for dk in range(2):
                        j = 2 * x + dj
                        k = 2 * y + dk
                        v_row = j * 4 + k
                        v_val = V2[dj * 2 + dk, c]
                        if v_val != 0:
                            V[v_row, col] += val * v_val
            
            # Fill W
            for block, val in step['C'].items():
                if val == 0:
                    continue
                x, y = int(block[0]), int(block[1])
                for dk in range(2):
                    for di in range(2):
                        k = 2 * x + dk
                        i = 2 * y + di
                        w_row = k * 4 + i
                        w_val = W2[dk * 2 + di, c]
                        if w_val != 0:
                            W[w_row, col] += val * w_val
    
    return U, V, W


def _build_4x4x8_strassen():
    """Build rank-56 decomposition for 4x4 * 4x8 -> 4x8.
    
    Split B into two 4x4 halves: B_left (cols 0-3) and B_right (cols 4-7).
    C = A * B = [A * B_left | A * B_right]
    Each A * B_half is a 4x4 * 4x4 multiplication with rank 28.
    Total rank: 56.
    """
    n, m, p = 4, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # First half: B cols 0-3, C rows 0-3
    U[:, 0:28] = U4
    # V: B rows j, cols 0-3. V4 rows are j*4+k. Global V rows are j*8+k.
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:28] = V4[j * 4 + k, :]
    # W: C rows 0-3, cols 0-3. W4 rows are k*4+i. Global W rows are k*4+i.
    W[0:16, 0:28] = W4
    
    # Second half: B cols 4-7, C rows 4-7
    U[:, 28:56] = U4
    # V: B rows j, cols 4-7. Global V rows are j*8 + (4+k).
    for j in range(4):
        for k in range(4):
            V[j * 8 + 4 + k, 28:56] = V4[j * 4 + k, :]
    # W: C rows 4-7, cols 0-3. Global W rows are (4+k)*4+i.
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 28:56] = W4[k * 4 + i, :]
    
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
    
    # Try Strassen rank-56
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
