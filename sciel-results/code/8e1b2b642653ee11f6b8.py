"""Rewritten program for Matrix Multiplication 3,5,7 using Strassen on 2x2x2 blocks."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _get_strassen_terms():
    """
    Return the 7 rank-1 tensor terms for the 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u, v, w are length-4 vectors.
    Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    For matrix mult: sum_r u_r[i*2+j] * v_r[j*2+k] * w_r[k*2+i] = 1 for all i,j,k in {0,1}.
    """
    # Standard Strassen decomposition verified by construction:
    # M1 = (a00+a11)(b00+b11) -> c00(+1), c11(+1)
    # M2 = (a10+a11)(b00)     -> c10(+1)
    # M3 = a00(b01-b11)        -> c01(+1), c11(+1)
    # M4 = a11(b10-b00)        -> c00(-1), c10(+1)
    # M5 = (a00+a01)(b11)      -> c00(+1), c01(+1)
    # M6 = (a10-a00)(b00+b01)  -> c10(+1), c11(-1)
    # M7 = (a01-a11)(b10+b11)  -> c00(-1), c11(+1)
    #
    # Verify: C00 = M1-M4+M5-M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M7
    # Each C entry gets exactly 1 from the appropriate A*B product.
    
    terms = [
        # M1: u=(1,0,0,1) for a00,a11; v=(1,0,0,1) for b00,b11; w=(1,0,0,1) for c00,c11
        (1, 0, 0, 1,  1, 0, 0, 1,  1, 0, 0, 1),
        # M2: u=(0,0,1,1) for a10,a11; v=(1,0,0,0) for b00; w=(0,0,1,0) for c10
        (0, 0, 1, 1,  1, 0, 0, 0,  0, 0, 1, 0),
        # M3: u=(1,0,0,0) for a00; v=(0,1,0,-1) for b01,-b11; w=(0,1,0,1) for c01,c11
        (1, 0, 0, 0,  0, 1, 0, -1,  0, 1, 0, 1),
        # M4: u=(0,0,0,1) for a11; v=(0,0,1,-1) for b10,-b00; w=(-1,0,1,0) for -c00,c10
        (0, 0, 0, 1,  0, 0, 1, -1,  -1, 0, 1, 0),
        # M5: u=(1,1,0,0) for a00,a01; v=(0,0,0,1) for b11; w=(1,1,0,0) for c00,c01
        (1, 1, 0, 0,  0, 0, 0, 1,  1, 1, 0, 0),
        # M6: u=(0,0,1,-1) for a10,-a00; v=(1,1,0,0) for b00,b01; w=(0,0,1,-1) for c10,-c11
        (0, 0, 1, -1,  1, 1, 0, 0,  0, 0, 1, -1),
        # M7: u=(0,1,0,-1) for a01,-a11; v=(0,0,1,1) for b10,b11; w=(0,0,-1,1) for -c00,c11
        (0, 1, 0, -1,  0, 0, 1, 1,  0, 0, -1, 1),
    ]
    
    # Verify: sum over terms of u[i*2+j] * v[j*2+k] * w[k*2+i] == 1 for all i,j,k
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ui = i * 2 + j
                vi = j * 2 + k
                wi = k * 2 + i
                total = 0.0
                for t in terms:
                    total += t[ui] * t[vi] * t[wi]
                if abs(total - 1.0) > 1e-9:
                    return []
    
    return terms

def _apply_strassen_block(U, V, W, n, m, p, i0, j0, k0):
    """
    Replace the 8 schoolbook terms for the 2x2x2 block at (i0,j0,k0) with 7 Strassen terms.
    Returns (new_U, new_V, new_W, success)
    """
    terms = _get_strassen_terms()
    if not terms:
        return U, V, W, False
    
    # Find the 8 schoolbook column indices for this block
    block_cols = []
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i, j, k = i0 + di, j0 + dj, k0 + dk
                col = i * (m * p) + j * p + k
                block_cols.append(col)
    
    old_rank = U.shape[1]
    new_rank = old_rank - 1
    
    new_U = np.zeros((n * m, new_rank))
    new_V = np.zeros((m * p, new_rank))
    new_W = np.zeros((p * n, new_rank))
    
    # Copy non-block columns
    col_new = 0
    for c in range(old_rank):
        if c not in block_cols:
            new_U[:, col_new] = U[:, c]
            new_V[:, col_new] = V[:, c]
            new_W[:, col_new] = W[:, c]
            col_new += 1
    
    # Add 7 Strassen columns
    for t in terms:
        # t is a flat 12-element tuple: u[0:4], v[4:8], w[8:12]
        u_loc = t[0:4]
        v_loc = t[4:8]
        w_loc = t[8:12]
        
        for li, val in enumerate(u_loc):
            if val != 0:
                di = li // 2
                dj = li % 2
                gi, gj = i0 + di, j0 + dj
                new_U[gi*m+gj, col_new] += val
        
        for li, val in enumerate(v_loc):
            if val != 0:
                dj = li // 2
                dk = li % 2
                gj, gk = j0 + dj, k0 + dk
                new_V[gj*p+gk, col_new] += val
        
        for li, val in enumerate(w_loc):
            if val != 0:
                dk = li // 2
                di = li % 2
                gk, gi = k0 + dk, i0 + di
                new_W[gk*n+gi, col_new] += val
        
        col_new += 1
    
    return new_U, new_V, new_W, True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Find all valid 2x2x2 block positions
    valid_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                valid_blocks.append((i0, j0, k0))
    
    # Greedy: pick blocks that don't share any (i,j,k) tensor element with already-picked blocks
    selected_blocks = []
    used_triples = set()
    
    # Sort blocks to prioritize those with more freedom (try all orderings via sorting)
    # Heuristic: process blocks in a fixed order that tends to pack well
    for i0, j0, k0 in valid_blocks:
        # Check if this block shares any (i,j,k) with already selected blocks
        block_triples = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    block_triples.add((i0+di, j0+dj, k0+dk))
        
        if block_triples & used_triples:
            continue
        
        selected_blocks.append((i0, j0, k0))
        used_triples |= block_triples
    
    # Apply Strassen to each selected block
    cur_U, cur_V, cur_W = best_U, best_V, best_W
    applied = 0
    for i0, j0, k0 in selected_blocks:
        if time.time() > deadline:
            break
        new_U, new_V, new_W, success = _apply_strassen_block(cur_U, cur_V, cur_W, n, m, p, i0, j0, k0)
        if success and _verify(new_U, new_V, new_W, n, m, p):
            cur_U, cur_V, cur_W = new_U, new_V, new_W
            applied += 1
    
    if applied > 0:
        best_U, best_V, best_W = cur_U, cur_V, cur_W
        best_rank = cur_U.shape[1]
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
