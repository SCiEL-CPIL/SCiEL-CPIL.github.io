import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half(U, V, W):
    """Round entries to nearest 0.5."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Standard Strassen rank-7 decomposition hardcoded.
    # This is a known valid decomposition.
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, -1]
    ], dtype=float)
    
    # Verify and fallback if incorrect (should not happen)
    if not _check_exact(U, V, W, 2, 2, 2, tol=1e-9):
        U = np.zeros((4, 8), dtype=float)
        V = np.zeros((4, 8), dtype=float)
        W = np.zeros((4, 8), dtype=float)
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j, c] = 1
                    V[j*2+k, c] = 1
                    W[k*2+i, c] = 1
                    c += 1
    return U, V, W

def _build_3x2x6():
    """
    Construct 3x2x6 decomposition using 2x2x2 Strassen blocks.
    n=3, m=2, p=6.
    Split p into three 2s.
    For each p-block (size 2), we have a 3x2x2 problem.
    We can decompose 3x2x2 into a 2x2x2 block (for i=0,1) and a 1x2x2 block (for i=2).
    Rank of 2x2x2 is 7. Rank of 1x2x2 is 2 (schoolbook).
    Total rank for 3x2x2 is 7+2 = 9.
    Three such blocks for p=6 -> rank 27.
    """
    n = 3
    m = 2
    p = 6
    
    U222, V222, W222 = _strassen_2x2x2()
    r222 = U222.shape[1] # 7
    r122 = 2
    
    r_block = r222 + r122 # 9
    total_rank = 3 * r_block # 27
    
    U_full = np.zeros((n*m, total_rank), dtype=float)
    V_full = np.zeros((m*p, total_rank), dtype=float)
    W_full = np.zeros((p*n, total_rank), dtype=float)
    
    for block_idx in range(3):
        p_offset = block_idx * 2
        offset = block_idx * r_block
        
        # Embed 2x2x2 part (i=0,1)
        for i in range(2):
            for j in range(2):
                for c in range(r222):
                    U_full[i*m + j, offset + c] = U222[i*2 + j, c]
        for j in range(2):
            for k in range(2):
                for c in range(r222):
                    V_full[j*p + (p_offset + k), offset + c] = V222[j*2 + k, c]
        for k in range(2):
            for i in range(2):
                for c in range(r222):
                    W_full[(p_offset + k)*n + i, offset + c] = W222[k*2 + i, c]
                    
        # Embed 1x2x2 part (i=2)
        for j in range(2):
            for k in range(2):
                c_local = j*2+k
                c_global = offset + r222 + c_local
                U_full[2*m + j, c_global] = 1
                V_full[j*p + (p_offset + k), c_global] = 1
                W_full[(p_offset + k)*n + 2, c_global] = 1
                
    return U_full, V_full, W_full, total_rank

def _build_3x3x6():
    """
    Construct 3x3x6 decomposition.
    Split p into three 2s.
    For each p-block (size 2), we have a 3x3x2 problem.
    Split m into 2 and 1.
    Block A: 3x2x2. Rank 9.
    Block B: 3x1x2. Rank 3 (schoolbook).
    Total rank for 3x3x2 is 9+3 = 12.
    Three such blocks for p=6 -> rank 36.
    """
    n = 3
    m = 3
    p = 6
    
    U222, V222, W222 = _strassen_2x2x2()
    r222 = 7
    r122 = 2
    r312 = 3
    
    r_block = r222 + r122 + r312 # 12
    total_rank = 3 * r_block # 36
    
    U_full = np.zeros((n*m, total_rank), dtype=float)
    V_full = np.zeros((m*p, total_rank), dtype=float)
    W_full = np.zeros((p*n, total_rank), dtype=float)
    
    for block_idx in range(3):
        p_offset = block_idx * 2
        offset = block_idx * r_block
        
        # Part 1: 3x2x2 (j=0,1)
        # i=0,1 part (2x2x2)
        for i in range(2):
            for j in range(2):
                for c in range(r222):
                    U_full[i*m + j, offset + c] = U222[i*2 + j, c]
        for j in range(2):
            for k in range(2):
                for c in range(r222):
                    V_full[j*p + (p_offset + k), offset + c] = V222[j*2 + k, c]
        for k in range(2):
            for i in range(2):
                for c in range(r222):
                    W_full[(p_offset + k)*n + i, offset + c] = W222[k*2 + i, c]
                    
        # i=2 part (1x2x2)
        for j in range(2):
            for k in range(2):
                c_local = j*2+k
                c_global = offset + r222 + c_local
                U_full[2*m + j, c_global] = 1
                V_full[j*p + (p_offset + k), c_global] = 1
                W_full[(p_offset + k)*n + 2, c_global] = 1
        
        # Part 2: 3x1x2 (j=2)
        for i in range(3):
            for k in range(2):
                c_local = i*2+k
                c_global = offset + r222 + r122 + c_local
                U_full[i*m + 2, c_global] = 1
                V_full[2*p + (p_offset + k), c_global] = 1
                W_full[(p_offset + k)*n + i, c_global] = 1
                
    return U_full, V_full, W_full, total_rank

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank

def _build_decomposition(n, m, p):
    # Split m into 2 and 3
    m1, m2 = 2, 3
    
    # Use the specific constructions
    U1, V1, W1, r1 = _build_3x2x6()
    U2, V2, W2, r2 = _build_3x3x6()
    
    total_rank = r1 + r2
    
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    # Embed U1 (3x2x6) into j=0,1
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U_full[i * m + j, c] = U1[i * m1 + j, c]
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V_full[j * p + k, c] = V1[j * p + k, c]
    for k in range(p):
        for i in range(n):
            for c in range(r1):
                W_full[k * n + i, c] = W1[k * n + i, c]
                
    # Embed U2 (3x3x6) into j=2,3,4
    offset = r1
    for i in range(n):
        for j in range(m2):
            for c in range(r2):
                U_full[i * m + (m1 + j), offset + c] = U2[i * m2 + j, c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V_full[(m1 + j) * p + k, offset + c] = V2[j * p + k, c]
    for k in range(p):
        for i in range(n):
            for c in range(r2):
                W_full[k * n + i, offset + c] = W2[k * n + i, c]
                
    return U_full, V_full, W_full, total_rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill

    try:
        U, V, W, rank = _build_decomposition(n, m, p)
        if time.time() < deadline:
            if _check_exact(U, V, W, n, m, p, tol=1e-6):
                # Convert to lists of fractions or floats as required (interface says integer/half-integer)
                # The current program returns .tolist() which gives floats.
                # The task says "integer/half-integer entries".
                # We should ensure they are exact if possible, but floats 0.0, 0.5, 1.0, -1.0 are fine.
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback schoolbook
    U, V, W, rank = _schoolbook_block(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
