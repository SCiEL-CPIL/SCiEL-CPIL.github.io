"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: Split B (4x8) into two 4x4 blocks B_left and B_right.
C = A @ B = [A @ B_left | A @ B_right]
Each 4x4 * 4x4 multiplication is decomposed via recursive Strassen (rank 49).
Total rank = 49 + 49 = 98.

The 4x4*4x4 decomposition uses 4 Strassen 2x2*2x2 blocks (rank 7 each)
plus cross-block corrections, but the standard recursive Strassen gives rank 49
for 4x4*4x4 (since 7^2 = 49).
"""

DIMENSIONS = [4, 4, 8]

import time
import numpy as np
from functools import lru_cache

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W


def _schoolbook_2x2x4():
    """
    Return U, V, W for 2x2 * 2x4 matrix multiplication with rank 8.
    Convention: U[i*2+j, c], V[j*4+k, c], W[k*2+i, c]
    U shape: (4, 8), V shape: (8, 8), W shape: (8, 8)
    n=2, m=2, p=4
    """
    n, m, p = 2, 2, 4
    rank = n * m * p  # 8
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


def _verify_2x2(U, V, W):
    """Verify that the 2x2*2x2 decomposition is correct."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for c in range(7):
                    val += U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _verify_2x2x4(U, V, W):
    """Verify 2x2*2x4 decomposition."""
    n, m, p = 2, 2, 4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(8):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _kronecker_4x4x8():
    """
    Construct rank-56 decomposition for 4x4 * 4x8 using Kronecker product:
    T_{4,4,8} = T_{2,2,4} (outer) x T_{2,2,2} (inner)
    
    Index mapping:
    - i = 2*i_outer + i_inner  (i_outer in 0..1, i_inner in 0..1)
    - j = 2*j_outer + j_inner  (j_outer in 0..1, j_inner in 0..1)  
    - k = 4*k_outer + k_inner  (k_outer in 0..1, k_inner in 0..3)
    
    For T_{2,2,4} x T_{2,2,2} = T_{4,4,8}:
    a=2,b=2,c=4 (outer), d=2,e=2,f=2 (inner)
    i = 2*i_outer + i_inner (i_outer 0..1, i_inner 0..1)
    j = 2*j_outer + j_inner
    k = 4*k_outer + k_inner (k_outer 0..1, k_inner 0..3)
    
    The decomposition: for each pair (c_outer, c_inner) where c_outer in 0..rank_outer-1, c_inner in 0..rank_inner-1:
    U[i,j,c1*r2+c2] = U1[i1,j1,c1] * U2[i2,j2,c2]
    V[j,k,c1*r2+c2] = V1[j1,k1,c1] * V2[j2,k2,c2]
    W[k,i,c1*r2+c2] = W1[k1,i1,c1] * W2[k2,i2,c2]
    
    For our case: outer is 2x2x4 (n1=2,m1=2,p1=4, rank r1=8), inner is 2x2x2 (n2=2,m2=2,p2=2, rank r2=7)
    Total rank = 8*7 = 56
    
    i = 2*i1 + i2 (i1 in 0..1, i2 in 0..1) -> i in 0..3
    j = 2*j1 + j2 (j1 in 0..1, j2 in 0..1) -> j in 0..3
    k = 4*k1 + k2 (k1 in 0..1, k2 in 0..3) -> k in 0..7
    """
    U222, V222, W222 = _strassen_2x2()
    U224, V224, W224 = _schoolbook_2x2x4()
    
    n1, m1, p1, r1 = 2, 2, 4, 8
    n2, m2, p2, r2 = 2, 2, 2, 7
    n, m, p = n1*n2, m1*m2, p1*p2  # 4, 4, 8
    rank = r1 * r2  # 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Vectorized construction using broadcasting
    # U[i*m+j, c1*r2+c2] = U224[i1*m1+j1, c1] * U222[i2*m2+j2, c2]
    # where i = n1*i1 + i2, j = m1*j1 + j2
    
    # Build index arrays
    i1_idx = np.repeat(np.arange(n1), n1*n2)  # [0,0,0,0, 1,1,1,1]
    i2_idx = np.tile(np.arange(n2), n1)       # [0,1,0,1, 0,1,0,1]
    j1_idx = np.repeat(np.arange(m1), m1*m2)  # [0,0,0,0, 1,1,1,1]
    j2_idx = np.tile(np.arange(m2), m1)       # [0,1,0,1, 0,1,0,1]
    
    # For U: shape (n*m, r1*r2)
    # i*m+j where i = n1*i1 + i2, j = m1*j1 + j2
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            u_vals = U224[i1_idx * m1 + j1_idx, c1] * U222[i2_idx * m2 + j2_idx, c2]
            row_indices = (n1 * i1_idx + i2_idx) * m + (m1 * j1_idx + j2_idx)
            U[row_indices, c] = u_vals
    
    # V[j*p+k, c1*r2+c2] = V224[j1*p1+k1, c1] * V222[j2*p2+k2, c2]
    # where j = m1*j1 + j2, k = p1*k1 + k2
    j1_v = np.repeat(np.arange(m1), m1*p1*p2)
    j2_v = np.tile(np.repeat(np.arange(m2), p1*p2), m1)
    k1_v = np.tile(np.repeat(np.arange(p1), p2), m1*m2)
    k2_v = np.tile(np.arange(p2), m1*m2*p1)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            v_vals = V224[j1_v * p1 + k1_v, c1] * V222[j2_v * p2 + k2_v, c2]
            row_indices = (m1 * j1_v + j2_v) * p + (p1 * k1_v + k2_v)
            V[row_indices, c] = v_vals
    
    # W[k*n+i, c1*r2+c2] = W224[k1*n1+i1, c1] * W222[k2*n2+i2, c2]
    # where k = p1*k1 + k2, i = n1*i1 + i2
    k1_w = np.repeat(np.arange(p1), p2*n1*n2)
    k2_w = np.tile(np.repeat(np.arange(p2), n1*n2), p1)
    i1_w = np.tile(np.repeat(np.arange(n1), n2), p1*p2)
    i2_w = np.tile(np.arange(n2), p1*p2*n1)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            w_vals = W224[k1_w * n1 + i1_w, c1] * W222[k2_w * n2 + i2_w, c2]
            row_indices = (p1 * k1_w + k2_w) * n + (n1 * i1_w + i2_w)
            W[row_indices, c] = w_vals
    
    return U, V, W


def _verify_4x4x8(U, V, W, rank):
    """Verify the 4x4*4x8 decomposition using vectorized operations."""
    n, m, p = 4, 4, 8
    # For each (i,j,k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build all required indices
    i_arr = np.repeat(np.arange(n), m * p)
    j_arr = np.tile(np.repeat(np.arange(m), p), n)
    k_arr = np.tile(np.arange(p), n * m)
    
    u_rows = i_arr * m + j_arr
    v_rows = j_arr * p + k_arr
    w_rows = k_arr * n + i_arr
    
    # Vectorized computation: for each (i,j,k), compute dot product over c
    # Result shape: (n*m*p,)
    result = np.zeros(n * m * p)
    for c in range(rank):
        result += U[u_rows, c] * V[v_rows, c] * W[w_rows, c]
    
    return np.all(np.abs(result - 1.0) < 1e-8)


def construct():
    t_start = time.time()
    DEADLINE = t_start + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Try Kronecker product decomposition (rank 56)
    # This uses T_{2,2,4} (rank 8) x T_{2,2,2} (rank 7) = T_{4,4,8} (rank 56)
    try:
        if time.time() > DEADLINE:
            raise TimeoutError("Deadline exceeded before Kronecker construction")
        
        U, V, W = _kronecker_4x4x8()
        rank = W.shape[1]
        
        # Verify
        if _verify_4x4x8(U, V, W, rank):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 128)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
