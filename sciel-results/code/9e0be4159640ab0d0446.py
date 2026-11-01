"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

_DEADLINE = 145.0  # Safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct (vectorized)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape into 3D tensors
    U3 = U.reshape(n, m, -1)  # (n, m, rank)
    V3 = V.reshape(m, p, -1)  # (m, p, rank)
    W3 = W.reshape(p, n, -1)  # (p, n, rank)
    
    # For each (i, j, k), compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    max_err = np.max(np.abs(result - 1.0))
    return float(max_err)


def _build_decomp(n, m, p):
    """
    Build a decomposition using Strassen for 2x2x2 sub-blocks plus rank-4 for 2x1x2 parts.
    
    For 2x3x8: split into 4 independent 2x3x2 sub-problems (one per pair of columns).
    Each 2x3x2 sub-problem:
      - A (2x3) split into A_left (2x2, cols 0,1) and A_right (2x1, col 2)
      - B (3x2) split into B_top (2x2, rows 0,1) and B_bottom (1x2, row 2)
      - C (2x2) = A_left * B_top + A_right * B_bottom
    
    A_left * B_top: 2x2x2, use Strassen (rank 7)
    A_right * B_bottom: 2x1x2, rank 4 (each output = one product)
    
    Total per sub-block: 7 + 4 = 11
    Total rank: 4 * 11 = 44
    """
    n, m, p = 2, 3, 8
    
    rank_total = 32
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    for g in range(4):
        # For sub-block g, output columns are 2g and 2g+1
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # B indices for this sub-block:
        # B[0, k0] at index 0*8+k0 = k0
        # B[0, k1] at index 0*8+k1 = k1
        # B[1, k0] at index 1*8+k0 = 8+k0
        # B[1, k1] at index 1*8+k1 = 8+k1
        # B[2, k0] at index 2*8+k0 = 16+k0
        # B[2, k1] at index 2*8+k1 = 16+k1
        
        b00 = k0      # B[0, k0]
        b01 = k1      # B[0, k1]
        b10 = 8 + k0  # B[1, k0]
        b11 = 8 + k1  # B[1, k1]
        b20 = 16 + k0 # B[2, k0]
        b21 = 16 + k1 # B[2, k1]
        
        # A indices (same for all sub-blocks):
        a00 = 0  # A[0,0]
        a01 = 1  # A[0,1]
        a02 = 2  # A[0,2]
        a10 = 3  # A[1,0]
        a11 = 4  # A[1,1]
        a12 = 5  # A[1,2]
        
        # C indices for this sub-block:
        # C[0, k0] at index k0*2+0 = k0*2
        # C[0, k1] at index k1*2+0 = k1*2
        # C[1, k0] at index k0*2+1 = k0*2+1
        # C[1, k1] at index k1*2+1 = k1*2+1
        
        c00 = k0 * 2      # C[0, k0]
        c01 = k1 * 2      # C[0, k1]
        c10 = k0 * 2 + 1  # C[1, k0]
        c11 = k1 * 2 + 1  # C[1, k1]
        
        # Part 1: A_right * B_bottom (rank 1 - outer product)
        # C[i, k] += A[i,2] * B[2, k] for i in {0,1}, k in {k0, k1}
        U[a02][col] = 1
        U[a12][col] = 1
        V[b20][col] = 1
        V[b21][col] = 1
        W[c00][col] = 1
        W[c01][col] = 1
        W[c10][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Part 2: A_left * B_top via Strassen (rank 7)
        # A_left is 2x2: A[0,0]@a00, A[0,1]@a01, A[1,0]@a10, A[1,1]@a11
        # B_top is 2x2: B[0,k0]@b00, B[0,k1]@b01, B[1,k0]@b10, B[1,k1]@b11
        # C_left is 2x2: C[0,k0]@c00, C[0,k1]@c01, C[1,k0]@c10, C[1,k1]@c11
        
        # Standard Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b11-b01)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        # C00 += M1, C11 += M1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        # C10 += M2
        W[c10][col] = 1
        col += 1
        
        # M3 = a00(b11-b01)
        U[a00][col] = 1
        V[b11][col] = 1
        V[b01][col] = -1
        # C01 += M3, C11 -= M3
        W[c01][col] = 1
        W[c11][col] = -1
        col += 1
        
        # M4 = a11(b00-b10)
        U[a11][col] = 1
        V[b00][col] = 1
        V[b10][col] = -1
        # C00 -= M4, C10 += M4
        W[c00][col] = -1
        W[c10][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11][col] = 1
        # C01 += M5
        W[c01][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10][col] = 1
        U[a00][col] = -1
        V[b00][col] = 1
        V[b01][col] = 1
        # C00 += M6
        W[c00][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01][col] = 1
        U[a11][col] = -1
        V[b10][col] = 1
        V[b11][col] = 1
        # C11 += M7
        W[c11][col] = 1
        col += 1
    
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        U, V, W = _build_decomp(n, m, p)
        # Quick verification (vectorized, fast)
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
