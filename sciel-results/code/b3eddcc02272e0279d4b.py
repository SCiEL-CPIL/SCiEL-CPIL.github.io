"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _build_block(n, m, p, row_off_u, row_off_v, row_off_w, col_off, U, V, W, deadline):
    """
    Build factor matrices for an n x m x p sub-block at given row offsets.
    row_off_u: starting row in U (for A indices)
    row_off_v: starting row in V (for B indices)  
    row_off_w: starting row in W (for C indices)
    col_off: starting column in U, V, W
    Returns the number of columns used.
    """
    if time.time() > deadline:
        return 0
    
    # Schoolbook for any block
    if n == 1 or m == 1 or p == 1 or (n != 2 or m != 2 or p != 2):
        r = n * m * p
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[row_off_u + i * m + j, col_off] = 1
                    V[row_off_v + j * p + k, col_off] = 1
                    W[row_off_w + k * n + i, col_off] = 1
                    col_off += 1
        return r
    
    # Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        # Strassen rank-1 tensors with proper row offsets
        # U rows: A[i,j] at row_off_u + i*2 + j
        # V rows: B[j,k] at row_off_v + j*2 + k  
        # W rows: C[i,k] at row_off_w + i*2 + k (trace convention: W[k*n+i] but here local)
        # Wait: the task says W[k*n+i, r] for C[i,k]. So C[i,k] -> row k*n+i
        # For 2x2: C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[2], C[1,1]->W[3]
        
        u00 = row_off_u + 0  # A[0,0]
        u01 = row_off_u + 1  # A[0,1]
        u10 = row_off_u + 2  # A[1,0]
        u11 = row_off_u + 3  # A[1,1]
        
        v00 = row_off_v + 0  # B[0,0]
        v01 = row_off_v + 1  # B[0,1]
        v10 = row_off_v + 2  # B[1,0]
        v11 = row_off_v + 3  # B[1,1]
        
        w00 = row_off_w + 0  # C[0,0] -> W[0*2+0]
        w01 = row_off_w + 1  # C[0,1] -> W[0*2+1]
        w10 = row_off_w + 2  # C[1,0] -> W[1*2+0]
        w11 = row_off_w + 3  # C[1,1] -> W[1*2+1]
        
        strassen_terms = [
            # T1: (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
            ({u00: 1, u11: 1}, {v00: 1, v11: 1}, {w00: 1, w11: 1}),
            # T2: (A10+A11)*B00 -> C10(+1)
            ({u10: 1, u11: 1}, {v00: 1}, {w10: 1}),
            # T3: A00*(B01-B11) -> C01(+1)
            ({u00: 1}, {v01: 1, v11: -1}, {w01: 1}),
            # T4: A11*(B10-B00) -> C10(+1)
            ({u11: 1}, {v10: 1, v00: -1}, {w10: 1}),
            # T5: (A00+A01)*B11 -> C01(+1), C00(-1)
            ({u00: 1, u01: 1}, {v11: 1}, {w01: 1, w00: -1}),
            # T6: (A10-A00)*(B00+B01) -> C11(+1)
            ({u10: 1, u00: -1}, {v00: 1, v01: 1}, {w11: 1}),
            # T7: (A01-A11)*(B10+B11) -> C00(+1)
            ({u01: 1, u11: -1}, {v10: 1, v11: 1}, {w00: 1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            if time.time() > deadline:
                break
            for row, val in u_coeffs.items():
                U[row, col_off] += val
            for row, val in v_coeffs.items():
                V[row, col_off] += val
            for row, val in w_coeffs.items():
                W[row, col_off] += val
            col_off += 1
        
        return 7
    
    # Recursive block decomposition for larger blocks
    # Split into 2+3 blocks
    n1, n2 = 2, n - 2
    m1, m2 = 2, m - 2
    p1, p2 = 2, p - 2
    
    total = 0
    # 8 sub-blocks: (n1,m1,p1), (n1,m1,p2), (n1,m2,p1), (n1,m2,p2),
    #               (n2,m1,p1), (n2,m1,p2), (n2,m2,p1), (n2,m2,p2)
    
    # (n1, m1, p1)
    r = _build_block(n1, m1, p1, row_off_u, row_off_v, row_off_w, col_off, U, V, W, deadline)
    total += r
    col_off += r
    if time.time() > deadline:
        return total
    
    # (n1, m1, p2)
    r = _build_block(n1, m1, p2, row_off_u, row_off_v + m1 * p2, row_off_w + p2 * n1, col_off, U, V, W, deadline)
    total += r
    col_off += r
    if time.time() > deadline:
        return total
    
    # (n1, m2, p1)
    r = _build_block(n1, m2, p1, row_off_u + m1 * n1, row_off_v + m1 * p1, row_off_w, col_off, U, V, W, deadline)
    total += r
    col_off += r
    if time.time() > deadline:
        return total
    
    # (n1, m2, p2)
    r = _build_block(n1, m2, p2, row_off_u + m1 * n1, row_off_v + m1 * p2, row_off_w + p2 * n1, col_off, U, V, W, deadline)
    total += r
    col_off += r
    if time.time() > deadline:
        return total
    
    # (n2, m1, p1)
    r = _build_block(n2, m1, p1, row_off_u + n1 * m1, row_off_v, row_off_w + n1 * p1, col_off, U, V, W, deadline)
    total += r
    col_off += r
    if time.time() > deadline:
        return total
    
    # (n2, m1, p2)
    r = _build_block(n2, m1, p2, row_off_u + n1 * m1, row_off_v + m1 * p2, row_off_w + n1 * p2, col_off, U, V, W, deadline)
    total += r
    col_off += r
    if time.time() > deadline:
        return total
    
    # (n2, m2, p1)
    r = _build_block(n2, m2, p1, row_off_u + n1 * m1 + m2 * n1, row_off_v + m1 * p1, row_off_w + n1 * p1, col_off, U, V, W, deadline)
    total += r
    col_off += r
    if time.time() > deadline:
        return total
    
    # (n2, m2, p2)
    r = _build_block(n2, m2, p2, row_off_u + n1 * m1 + m2 * n1, row_off_v + m1 * p2, row_off_w + n1 * p2, col_off, U, V, W, deadline)
    total += r
    col_off += r
    
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin
    
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    try:
        U = np.zeros((n * m, max_rank), dtype=float)
        V = np.zeros((m * p, max_rank), dtype=float)
        W = np.zeros((p * n, max_rank), dtype=float)
        
        rank_used = _build_block(n, m, p, 0, 0, 0, 0, U, V, W, deadline)
        
        if rank_used > 0 and rank_used < max_rank:
            U = U[:, :rank_used]
            V = V[:, :rank_used]
            W = W[:, :rank_used]
            
            U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in U]
            V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in V]
            W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in W]
            
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
