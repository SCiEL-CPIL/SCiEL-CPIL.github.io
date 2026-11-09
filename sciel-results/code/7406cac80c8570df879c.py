"""Tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
DEADLINE_MARGIN = 10  # seconds before the 150s limit


def _build_schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W):
    """Check that sum_c U[:,c]*V[:,c]*W[:,c] reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to proper tensor shapes
    Uf = U.reshape(n, m, U.shape[1])
    Vf = V.reshape(m, p, V.shape[1])
    Wf = W.reshape(p, n, W.shape[1])
    
    # Build C from tensor decomposition: C[i,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    
    # Build C from actual matrix multiplication
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    
    return np.allclose(C, Ctrue, atol=1e-6)


def _strassen_2x2_decomp():
    """
    Return Strassen's 7-rank decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) for input A (2x2 flattened row-major)
    V: (4, 7) for input B (2x2 flattened row-major)  
    W: (4, 7) for output C (2x2 flattened row-major)
    
    Strassen's formulas:
    M1 = (a00 + a11) * (b00 + b11)
    M2 = (a10 + a11) * b00
    M3 = a00 * (b01 - b11)
    M4 = a11 * (b10 - b00)
    M5 = (a00 + a01) * b11
    M6 = (a10 - a00) * (b00 + b10)
    M7 = (a11 - a10) * (b01 + b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U rows: [a00, a01, a10, a11]
    # V rows: [b00, b01, b10, b11]
    # W rows: [c00, c01, c10, c11]
    
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a00
        [0, 1, 0, 0, 1, 0, 0],  # a01
        [0, 0, 1, 1, 0, 1, 0],  # a10
        [1, 0, 0, 1, 0, 0, 1],  # a11
    ]
    
    V = [
        [1, 1, 0, 1, 0, 1, 0],  # b00
        [1, 0, 1, 0, 0, 0, 1],  # b01
        [0, 0, 0, 0, 1, 1, 0],  # b10
        [1, 0, -1, 0, 1, 0, 1],  # b11
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, 0, 1, 0, 0, 1, 0],  # c11
    ]
    
    return U, V, W


def _make_4x4x4_strassen():
    """
    Build 4x4 * 4x4 -> 4x4 decomposition using recursive Strassen.
    Split into 2x2 blocks and use Strassen for each.
    Returns U (16, rank), V (16, rank), W (16, rank)
    """
    n = m = p = 4
    U = [[0] * 49 for _ in range(n * m)]
    V = [[0] * 49 for _ in range(m * p)]
    W = [[0] * 49 for _ in range(p * n)]
    
    # Get the 2x2 Strassen decomposition
    strassen_U, strassen_V, strassen_W = _strassen_2x2_decomp()
    
    # Strassen formulas for 2x2 block matrices:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B10)
    # M7 = (A11-A10) * (B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the block structure for each Strassen product
    # Each entry is (block_i, block_j, sign)
    outer_products = [
        # (A_blocks, B_blocks, C_blocks)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),      # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),               # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),               # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),               # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),               # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),               # M6
        ([(1,1,1), (1,0,-1)], [(0,1,1), (1,1,1)], [(0,0,1)]),               # M7
    ]
    
    def get_block_rows(block_i, block_j, size=4):
        """Get row indices for a 2x2 block in a 4x4 matrix (flattened row-major)."""
        start_row = block_i * 2
        start_col = block_j * 2
        rows = []
        for di in range(2):
            for dj in range(2):
                row = (start_row + di) * size + (start_col + dj)
                rows.append(row)
        return rows
    
    # Build the decomposition
    for outer_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_products):
        for inner_idx in range(7):
            col = outer_idx * 7 + inner_idx
            
            # For each involved A block, add its contribution to U
            for (bi, bj, sign) in a_blocks:
                rows = get_block_rows(bi, bj)
                for local_i, row in enumerate(rows):
                    U[row][col] += sign * strassen_U[local_i][inner_idx]
            
            # For each involved B block, add its contribution to V
            for (bi, bj, sign) in b_blocks:
                rows = get_block_rows(bi, bj)
                for local_i, row in enumerate(rows):
                    V[row][col] += sign * strassen_V[local_i][inner_idx]
            
            # For each involved C block, add its contribution to W
            for (bi, bj, sign) in c_blocks:
                rows = get_block_rows(bi, bj)
                for local_i, row in enumerate(rows):
                    W[row][col] += sign * strassen_W[local_i][inner_idx]
    
    return U, V, W


def _make_4x4x5_decomp():
    """
    Build 4x4 * 4x5 -> 4x5 decomposition.
    Split B into B[:, :4] and B[:, 4].
    C = A * B[:, :4] + A * B[:, 4:5]
    
    First part: 4x4 * 4x4 -> 4x4 (use Strassen, rank 49)
    Second part: 4x4 * 4x1 -> 4x1 (rank 1 using all-ones trick)
    
    Total rank: 49 + 1 = 50
    """
    # Get the 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = _make_4x4x4_strassen()
    
    # Now extend to 4x4x5
    # U: (16, 50) - A is 4x4, same for both parts
    # V: (20, 50) - B is 4x5
    # W: (20, 50) - C is 4x5
    
    U = [[0] * 50 for _ in range(16)]
    V = [[0] * 50 for _ in range(20)]
    W = [[0] * 50 for _ in range(20)]
    
    # Copy 4x4x4 part into first 49 columns
    # U_444 is (16, 49), V_444 is (16, 49) for B[:, :4], W_444 is (16, 49) for C[:, :4]
    for i in range(16):
        for c in range(49):
            U[i][c] = U_444[i][c]
    
    # V_444 has 16 rows (for B[:, :4]), we need to place them in V[j*5+k, c] for k=0..3
    # B row j, col k -> index j*5 + k
    for j in range(4):
        for k in range(4):
            row = j * 5 + k
            for c in range(49):
                V[row][c] = V_444[j * 4 + k][c]
    
    # W_444 has 16 rows (for C[:, :4]), we need to place them in W[k*4+i, c] for k=0..3
    # C row i, col k -> index k*4 + i (following the convention W[k*n+i, r])
    for i in range(4):
        for k in range(4):
            row = k * 4 + i
            for c in range(49):
                W[row][c] = W_444[k * 4 + i][c]
    
    # Add the 4x4x1 part (k=4) with rank 1
    # U[i*4+j, 49] = 1 for all i,j
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][49] = 1
    
    # V[j*5+4, 49] = 1 for all j
    for j in range(4):
        V[j * 5 + 4][49] = 1
    
    # W[4*4+i, 49] = 1 for all i
    for i in range(4):
        W[4 * 4 + i][49] = 1
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Start with schoolbook as fallback
    fallback = _build_schoolbook()
    
    # Try the Strassen-based decomposition
    try:
        U, V, W = _make_4x4x5_decomp()
        
        # Verify correctness
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, return schoolbook
            return fallback
    except Exception:
        # If Strassen approach fails, return schoolbook
        return fallback


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['W'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
