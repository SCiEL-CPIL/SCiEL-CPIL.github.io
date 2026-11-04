import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _construct_block_2x2x2(offset_u, offset_v, offset_w, U, V, W, J, K, I):
    """
    Adds Strassen decomposition for a 2x2 by 2x2 block.
    J: list of 2 global j indices
    K: list of 2 global k indices
    I: list of 2 global i indices (should be [0, 1])
    offset_u, offset_v, offset_w: current column offsets in U, V, W
    Returns new offsets.
    """
    # Strassen 7 products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We define the terms c=0..6
    # U entries for each term (i, j) where i in {0,1}, j in {0,1} local
    # V entries for each term (j, k)
    # W entries for each term (k, i)
    
    # Term 0 (P1):
    # U: A00(1), A11(1) -> (0,0)=1, (1,1)=1
    # V: B00(1), B11(1) -> (0,0)=1, (1,1)=1
    # W: C00(1), C11(1) -> (0,0)=1, (1,1)=1
    # Note: W index is k*n+i. Local k, i.
    
    terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict {(i,j): val}
        # V_coeffs: dict {(j,k): val}
        # W_coeffs: dict {(k,i): val}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # P1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # P2 -> C10
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1}), # P3 -> C01
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(1,0): 1}), # P4 -> C10, C00
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}), # P5 -> C01
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # P6 -> C11
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}), # P7 -> C00
    ]
    
    # Mapping local to global
    # I = [i0, i1], J = [j0, j1], K = [k0, k1]
    
    for t_idx, (u_c, v_c, w_c) in enumerate(terms):
        c = offset_u + t_idx
        
        # Fill U
        for (li, lj), val in u_c.items():
            gi = I[li]
            gj = J[lj]
            U[gi * 5 + gj][c] = val # m=5
            
        # Fill V
        for (lj, lk), val in v_c.items():
            gj = J[lj]
            gk = K[lk]
            V[gj * 5 + gk][c] = val # p=5
            
        # Fill W
        for (lk, li), val in w_c.items():
            gk = K[lk]
            gi = I[li]
            W[gk * 2 + gi][c] = val # n=2
            
    return offset_u + 7, offset_v + 7, offset_w + 7

def _construct_block_2x1x2(offset_u, offset_v, offset_w, U, V, W, J, K, I):
    """
    Adds decomposition for 2x1 by 1x2 block (outer product).
    J: [j0]
    K: [k0, k1]
    I: [i0, i1]
    """
    # C_i,k = A_i,j0 * B_j0,k
    # 4 terms
    # (i, j, k) -> (i, j0, k)
    # U: (i, j0)
    # V: (j0, k)
    # W: (k, i)
    
    terms = [
        ({(0, 0): 1}, {(0, 0): 1}, {(0, 0): 1}), # i=0, k=0
        ({(0, 0): 1}, {(0, 1): 1}, {(1, 0): 1}), # i=0, k=1
        ({(1, 0): 1}, {(0, 0): 1}, {(0, 1): 1}), # i=1, k=0
        ({(1, 0): 1}, {(0, 1): 1}, {(1, 1): 1}), # i=1, k=1
    ]
    
    j0 = J[0]
    k0, k1 = K[0], K[1]
    i0, i1 = I[0], I[1]
    
    for t_idx, (u_c, v_c, w_c) in enumerate(terms):
        c = offset_u + t_idx
        
        # U
        for (li, lj), val in u_c.items():
            gi = I[li]
            gj = J[lj]
            U[gi * 5 + gj][c] = val
            
        # V
        for (lj, lk), val in v_c.items():
            gj = J[lj]
            gk = K[lk]
            V[gj * 5 + gk][c] = val
            
        # W
        for (lk, li), val in w_c.items():
            gk = K[lk]
            gi = I[li]
            W[gk * 2 + gi][c] = val
            
    return offset_u + 4, offset_v + 4, offset_w + 4

def _construct_block_2x2x1(offset_u, offset_v, offset_w, U, V, W, J, K, I):
    """
    Adds decomposition for 2x2 by 2x1 block.
    J: [j0, j1]
    K: [k0]
    I: [i0, i1]
    """
    # C_i = sum_j A_i,j B_j,k0
    # 4 terms
    terms = [
        ({(0, 0): 1}, {(0, 0): 1}, {(0, 0): 1}), # i=0, j=0
        ({(0, 1): 1}, {(1, 0): 1}, {(0, 0): 1}), # i=0, j=1
        ({(1, 0): 1}, {(0, 0): 1}, {(0, 1): 1}), # i=1, j=0
        ({(1, 1): 1}, {(1, 0): 1}, {(0, 1): 1}), # i=1, j=1
    ]
    
    for t_idx, (u_c, v_c, w_c) in enumerate(terms):
        c = offset_u + t_idx
        
        # U
        for (li, lj), val in u_c.items():
            gi = I[li]
            gj = J[lj]
            U[gi * 5 + gj][c] = val
            
        # V
        for (lj, lk), val in v_c.items():
            gj = J[lj]
            gk = K[lk]
            V[gj * 5 + gk][c] = val
            
        # W
        for (lk, li), val in w_c.items():
            gk = K[lk]
            gi = I[li]
            W[gk * 2 + gi][c] = val
            
    return offset_u + 4, offset_v + 4, offset_w + 4

def _construct_block_2x1x1(offset_u, offset_v, offset_w, U, V, W, J, K, I):
    """
    Adds decomposition for 2x1 by 1x1 block.
    J: [j0]
    K: [k0]
    I: [i0, i1]
    """
    # C_i = A_i,j0 B_j0,k0
    # 2 terms
    terms = [
        ({(0, 0): 1}, {(0, 0): 1}, {(0, 0): 1}), # i=0
        ({(1, 0): 1}, {(0, 0): 1}, {(0, 1): 1}), # i=1
    ]
    
    for t_idx, (u_c, v_c, w_c) in enumerate(terms):
        c = offset_u + t_idx
        
        # U
        for (li, lj), val in u_c.items():
            gi = I[li]
            gj = J[lj]
            U[gi * 5 + gj][c] = val
            
        # V
        for (lj, lk), val in v_c.items():
            gj = J[lj]
            gk = K[lk]
            V[gj * 5 + gk][c] = val
            
        # W
        for (lk, li), val in w_c.items():
            gk = K[lk]
            gi = I[li]
            W[gk * 2 + gi][c] = val
            
    return offset_u + 2, offset_v + 2, offset_w + 2

def _construct():
    n, m, p = DIMENSIONS
    # Fallback
    best = _schoolbook(n, m, p)
    
    # Construct rank 46 solution
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    I = [0, 1]
    J0 = [0, 1]
    J1 = [2, 3]
    J2 = [4]
    K0 = [0, 1]
    K1 = [2, 3]
    K2 = [4]
    
    off_u = 0
    off_v = 0
    off_w = 0
    
    # Block (0,0): 2x2 x 2x2
    off_u, off_v, off_w = _construct_block_2x2x2(off_u, off_v, off_w, U, V, W, J0, K0, I)
    # Block (1,0): 2x2 x 2x2
    off_u, off_v, off_w = _construct_block_2x2x2(off_u, off_v, off_w, U, V, W, J1, K0, I)
    # Block (2,0): 2x1 x 1x2
    off_u, off_v, off_w = _construct_block_2x1x2(off_u, off_v, off_w, U, V, W, J2, K0, I)
    
    # Block (0,1): 2x2 x 2x2
    off_u, off_v, off_w = _construct_block_2x2x2(off_u, off_v, off_w, U, V, W, J0, K1, I)
    # Block (1,1): 2x2 x 2x2
    off_u, off_v, off_w = _construct_block_2x2x2(off_u, off_v, off_w, U, V, W, J1, K1, I)
    # Block (2,1): 2x1 x 1x2
    off_u, off_v, off_w = _construct_block_2x1x2(off_u, off_v, off_w, U, V, W, J2, K1, I)
    
    # Block (0,2): 2x2 x 2x1
    off_u, off_v, off_w = _construct_block_2x2x1(off_u, off_v, off_w, U, V, W, J0, K2, I)
    # Block (1,2): 2x2 x 2x1
    off_u, off_v, off_w = _construct_block_2x2x1(off_u, off_v, off_w, U, V, W, J1, K2, I)
    # Block (2,2): 2x1 x 1x1
    off_u, off_v, off_w = _construct_block_2x1x1(off_u, off_v, off_w, U, V, W, J2, K2, I)
    
    # Verify
    if _verify(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
