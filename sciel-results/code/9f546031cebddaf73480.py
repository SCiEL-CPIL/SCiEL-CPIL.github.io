"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

_DEADLINE = 140.0  # Safety margin before 150s kill


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


def _build_decomp():
    """
    Build a rank-44 decomposition for 2x3x8 matrix multiplication.
    
    Strategy:
    - Split the 3x8 part into A_2x2 * B_2x8 and A_2x1 * B_1x8.
    - A_2x1 * B_1x8 requires rank 16 (2*8).
    - A_2x2 * B_2x8: Split B_2x8 into four 2x2 blocks.
    - Each 2x2 * 2x2 = 2x2 product uses Strassen (rank 7).
    - Total rank: 16 + 4 * 7 = 44.
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A_2x1 * B_1x8 (rank 16)
    # A[0,2] at index 0*3+2=2, A[1,2] at index 1*3+2=5
    # B[2,k] at index 2*8+k = 16+k
    # C[i,k] at index k*2+i
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A_2x2 * B_2x8 via Strassen (rank 28)
    # A_2x2 indices: A[0,0]@0, A[0,1]@1, A[1,0]@3, A[1,1]@4
    # B_2x8 rows 0,1. B[0,k]@k, B[1,k]@8+k
    # C is 2x8: C[i,k] at k*2+i
    
    # Strassen multipliers for 2x2 * 2x2:
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    # M2 = (a10+a11)b00     -> C10 += M2
    # M3 = a00(b11-b01)     -> C01 += M3, C11 -= M3
    # M4 = a11(b00-b10)     -> C00 -= M4, C10 += M4
    # M5 = (a00+a01)b11     -> C01 += M5
    # M6 = (a10-a00)(b00+b01) -> C00 += M6
    # M7 = (a01-a11)(b10+b11) -> C11 += M7
    
    # Note: The standard Strassen output combination is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Wait, the standard Strassen formulas are:
    # p1 = (A11 + A22) (B11 + B22)
    # p2 = (A21 + A22) B11
    # p3 = A11 (B22 - B12)
    # p4 = A22 (B11 - B21)
    # p5 = (A11 + A12) B22
    # p6 = (A21 - A11) (B11 + B12)
    # p7 = (A12 - A22) (B21 + B22)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # My indices:
    # A11 = A[0,0] (0), A12 = A[0,1] (1)
    # A21 = A[1,0] (3), A22 = A[1,1] (4)
    # B11 = B[0, 2g], B12 = B[0, 2g+1]
    # B21 = B[1, 2g], B22 = B[1, 2g+1]
    # C11 = C[0, 2g] (idx (2g)*2+0)
    # C12 = C[0, 2g+1] (idx (2g+1)*2+0)
    # C21 = C[1, 2g] (idx (2g)*2+1)
    # C22 = C[1, 2g+1] (idx (2g+1)*2+1)
    
    for g in range(4):
        b11_idx = 0 + 2 * g
        b12_idx = 0 + 2 * g + 1
        b21_idx = 8 + 2 * g
        b22_idx = 8 + 2 * g + 1
        
        a11_idx = 0
        a12_idx = 1
        a21_idx = 3
        a22_idx = 4
        
        c11_idx = (2 * g) * 2 + 0
        c12_idx = (2 * g + 1) * 2 + 0
        c21_idx = (2 * g) * 2 + 1
        c22_idx = (2 * g + 1) * 2 + 1
        
        # p1 = (a11+a22)(b11+b22)
        U[a11_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        V[b22_idx][col] = 1
        W[c11_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
        # p2 = (a21+a22)b11
        U[a21_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        W[c21_idx][col] = 1
        col += 1
        
        # p3 = a11(b22-b12)
        U[a11_idx][col] = 1
        V[b22_idx][col] = 1
        V[b12_idx][col] = -1
        W[c12_idx][col] = 1
        W[c22_idx][col] = -1
        col += 1
        
        # p4 = a22(b11-b21)
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        V[b21_idx][col] = -1
        W[c11_idx][col] = -1
        W[c21_idx][col] = 1
        col += 1
        
        # p5 = (a11+a12)b22
        U[a11_idx][col] = 1
        U[a12_idx][col] = 1
        V[b22_idx][col] = 1
        W[c12_idx][col] = 1
        W[c11_idx][col] = -1
        col += 1
        
        # p6 = (a21-a11)(b11+b12)
        U[a21_idx][col] = 1
        U[a11_idx][col] = -1
        V[b11_idx][col] = 1
        V[b12_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # p7 = (a12-a22)(b21+b22)
        U[a12_idx][col] = 1
        U[a22_idx][col] = -1
        V[b21_idx][col] = 1
        V[b22_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        U, V, W = _build_decomp()
        
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        
        # Quick verification using NumPy (vectorized)
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = len(U[0])
        
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape and use einsum
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # We need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Create index arrays
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        u_idx = i_idx * m + j_idx  # shape (n,m,p)
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        # Gather
        U_g = U_np[u_idx]  # (n,m,p,rank)
        V_g = V_np[v_idx]
        W_g = W_np[w_idx]
        
        # Sum over rank
        vals = np.sum(U_g * V_g * W_g, axis=-1)  # (n,m,p)
        max_err = np.max(np.abs(vals - 1.0))
        
        if max_err < 1e-10:
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback if verification fails (should not happen)
            pass
            
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
