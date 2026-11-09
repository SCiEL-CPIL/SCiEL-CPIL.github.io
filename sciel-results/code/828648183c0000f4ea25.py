"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    
    Convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    For 2x2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
    """
    n_s, m_s, p_s = 2, 2, 2
    rank_s = 7
    U = np.zeros((n_s * m_s, rank_s), dtype=float)
    V = np.zeros((m_s * p_s, rank_s), dtype=float)
    W = np.zeros((p_s * n_s, rank_s), dtype=float)
    
    # Strassen decomposition for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a10)b11
    # M6 = (a11-a10)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, r]:
    U[0, 0] = 1; U[3, 0] = 1       # M1: a00, a11
    U[2, 1] = 1; U[3, 1] = 1       # M2: a10, a11
    U[0, 2] = 1                     # M3: a00
    U[3, 3] = 1                     # M4: a11
    U[0, 4] = 1; U[2, 4] = 1       # M5: a00, a10
    U[2, 5] = -1; U[3, 5] = 1      # M6: a10, a11
    U[1, 6] = 1; U[3, 6] = -1      # M7: a01, a11
    
    # V[j*2+k, r]:
    V[0, 0] = 1; V[3, 0] = 1       # M1: b00, b11
    V[0, 1] = 1                     # M2: b00
    V[1, 2] = 1; V[3, 2] = -1      # M3: b01, b11
    V[0, 3] = -1; V[2, 3] = 1      # M4: b00, b10
    V[3, 4] = 1                     # M5: b11
    V[0, 5] = 1; V[2, 5] = 1       # M6: b00, b10
    V[1, 6] = 1; V[3, 6] = 1       # M7: b01, b11
    
    # W[k*2+i, r]:
    W[0, 0] = 1; W[3, 0] = 1       # M1: c00, c11
    W[1, 1] = 1; W[3, 1] = -1      # M2: c10, c11
    W[2, 2] = 1; W[3, 2] = 1       # M3: c01, c11
    W[0, 3] = 1; W[1, 3] = 1       # M4: c00, c10
    W[0, 4] = -1; W[2, 4] = 1      # M5: c00, c01
    W[3, 5] = 1                     # M6: c11
    W[0, 6] = 1                     # M7: c00
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(s - 1) > 1e-10:
                    return False
    return True


def _build_2x4x2_from_strassen(U_s, V_s, W_s):
    """Build 2x4x2 decomposition (rank 14) from two 2x2x2 Strassen blocks.
    
    Split j into j in {0,1} (block 1) and j in {2,3} (block 2).
    Both blocks have k in {0,1}.
    """
    n, m, p = 2, 4, 2
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block 1: j_local in {0,1} -> global j = j_local
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                U[i * 4 + j_local, r] = U_s[i * 2 + j_local, r]
        for j_local in range(2):
            for k in range(2):
                V[j_local * 2 + k, r] = V_s[j_local * 2 + k, r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, r] = W_s[k * 2 + i, r]
    
    # Block 2: j_local in {0,1} -> global j = j_local + 2
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                U[i * 4 + (j_local + 2), r + 7] = U_s[i * 2 + j_local, r]
        for j_local in range(2):
            for k in range(2):
                V[(j_local + 2) * 2 + k, r + 7] = V_s[j_local * 2 + k, r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, r + 7] = W_s[k * 2 + i, r]
    
    return U, V, W


def _build_2x4x6_from_2x4x2(U_2x4x2, V_2x4x2, W_2x4x2):
    """Build 2x4x6 decomposition (rank 42) from three 2x4x2 blocks.
    
    Split k into k in {0,1}, {2,3}, {4,5}.
    """
    n, m, p = 2, 4, 6
    rank_sub = 14
    total_rank = 3 * rank_sub  # 42
    
    U_total = np.zeros((n * m, total_rank), dtype=float)
    V_total = np.zeros((m * p, total_rank), dtype=float)
    W_total = np.zeros((p * n, total_rank), dtype=float)
    
    for block in range(3):
        r_offset = block * rank_sub
        k_offset = block * 2
        
        for r in range(rank_sub):
            for i in range(2):
                for j in range(4):
                    U_total[i * 4 + j, r_offset + r] = U_2x4x2[i * 4 + j, r]
            for j in range(4):
                for k_local in range(2):
                    V_total[j * 6 + (k_local + k_offset), r_offset + r] = V_2x4x2[j * 2 + k_local, r]
            for k_local in range(2):
                for i in range(2):
                    W_total[(k_local + k_offset) * 2 + i, r_offset + r] = W_2x4x2[k_local * 2 + i, r]
    
    return U_total, V_total, W_total


def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build Strassen 2x2x2 decomposition (rank 7)
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Verify Strassen decomposition
    if not _verify_decomposition(U_s, V_s, W_s, 2, 2, 2):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 2x4x2 decomposition (rank 14) from two Strassen blocks
    U_2x4x2, V_2x4x2, W_2x4x2 = _build_2x4x2_from_strassen(U_s, V_s, W_s)
    
    # Verify 2x4x2 decomposition
    if not _verify_decomposition(U_2x4x2, V_2x4x2, W_2x4x2, 2, 4, 2):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check time
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build final 2x4x6 decomposition (rank 42) from three 2x4x2 blocks
    U_total, V_total, W_total = _build_2x4x6_from_2x4x2(U_2x4x2, V_2x4x2, W_2x4x2)
    
    # Verify full decomposition
    if not _verify_decomposition(U_total, V_total, W_total, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
