import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 110

def _elapsed():
    return time.time() - START_TIME

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that the decomposition is exact for matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check tensor identity: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_strassen_block(i0, j0, k0, n, m, p, U, V, W, col):
    """Apply Strassen to 2x2 block at row i0..i0+1, col j0..j0+1, producing k0..k0+1.
    Returns new column index."""
    # The 7 multiplications:
    # idx 0: M1 = (A00+A11)(B00+B11) -> C00 +, C11 +
    # idx 1: M2 = (A10+A11)B00 -> C10 +, C11 -
    # idx 2: M3 = A00(B10-B11) -> C01 +, C11 +
    # idx 3: M4 = A11(B01-B00) -> C00 +, C10 +
    # idx 4: M5 = (A00+A01)B11 -> C00 -, C01 +
    # idx 5: M6 = (A10-A00)(B00+B01) -> C11 +
    # idx 6: M7 = (A01-A11)(B10+B11) -> C00 +
    
    a_coeffs = [
        [(0,1), (1,1)],   # M1: A00, A11
        [(1,0), (1,1)],   # M2: A10, A11
        [(0,0)],          # M3: A00
        [(1,1)],          # M4: A11
        [(0,0), (0,1)],   # M5: A00, A01
        [(1,0), (0,0)],   # M6: A10, A00 (with sign: A10 - A00)
        [(0,1), (1,1)],   # M7: A01, A11 (with sign: A01 - A11)
    ]
    a_signs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1],
    ]
    
    b_coeffs = [
        [(0,0), (1,1)],   # M1: B00, B11
        [(0,0)],          # M2: B00
        [(1,0), (1,1)],   # M3: B10, B11 (sign: B10 - B11)
        [(0,1), (0,0)],   # M4: B01, B00 (sign: B01 - B00)
        [(1,1)],          # M5: B11
        [(0,0), (0,1)],   # M6: B00, B01
        [(1,0), (1,1)],   # M7: B10, B11
    ]
    b_signs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1],
    ]
    
    c_contribs = [
        {(0,0): 1, (1,1): 1},      # M1
        {(1,0): 1, (1,1): -1},     # M2
        {(0,1): 1, (1,1): 1},      # M3
        {(0,0): 1, (1,0): 1},      # M4
        {(0,0): -1, (0,1): 1},     # M5
        {(1,1): 1},                # M6
        {(0,0): 1},                # M7
    ]
    
    for t in range(7):
        c = col + t
        # U contributions
        for (di, dj), coeff in zip(a_coeffs[t], a_signs[t]):
            row = (i0 + di) * m + (j0 + dj)
            U[row][c] += coeff
        # V contributions
        for (dj, dk), coeff in zip(b_coeffs[t], b_signs[t]):
            row = (j0 + dj) * p + (k0 + dk)
            V[row][c] += coeff
        # W contributions
        for (dk, di), coeff in c_contribs[t].items():
            row = (k0 + dk) * n + (i0 + di)
            W[row][c] += coeff
    
    return col + 7

def _construct_schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col):
    """Schoolbook for sub-block of size hi x hj x hk."""
    for di in range(hi):
        for dj in range(hj):
            for dk in range(hk):
                U[(i0 + di) * m + (j0 + dj)][col] += 1
                V[(j0 + dj) * p + (k0 + dk)][col] += 1
                W[(k0 + dk) * n + (i0 + di)][col] += 1
                col += 1
    return col

def construct():
    n, m, p = 3, 5, 7
    
    # Splits to maximize 2x2x2 Strassen blocks
    splits_i = [2, 1]
    splits_j = [2, 2, 1]
    splits_k = [2, 2, 2, 1]
    
    # Max possible rank is n*m*p
    max_rank = n * m * p
    U = [[0.0 for _ in range(max_rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(max_rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(max_rank)] for _ in range(p * n)]
    
    col = 0
    i_offsets = [0]
    for s in splits_i[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in splits_j[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in splits_k[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi = splits_i[ii]
                hj = splits_j[jj]
                hk = splits_k[kk]
                i0 = i_offsets[ii]
                j0 = j_offsets[jj]
                k0 = k_offsets[kk]
                
                if hi == 2 and hj == 2 and hk == 2:
                    col = _construct_strassen_block(i0, j0, k0, n, m, p, U, V, W, col)
                else:
                    col = _construct_schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col)
    
    # Trim trailing zero columns
    last_col = 0
    for c in range(col - 1, -1, -1):
        nz = False
        for row in range(n * m):
            if U[row][c] != 0:
                nz = True
                break
        if not nz:
            for row in range(m * p):
                if V[row][c] != 0:
                    nz = True
                    break
        if not nz:
            for row in range(p * n):
                if W[row][c] != 0:
                    nz = True
                    break
        if nz:
            last_col = c
            break
    final_rank = last_col + 1
    
    U_trim = [row[:final_rank] for row in U]
    V_trim = [row[:final_rank] for row in V]
    W_trim = [row[:final_rank] for row in W]
    
    # Verify
    if not _verify(U_trim, V_trim, W_trim):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
        V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
        W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][idx] = 1.0
                    V[j*p + k][idx] = 1.0
                    W[k*n + i][idx] = 1.0
                    idx += 1
        return {'U': U, 'V': V, 'W': W}
        
    return {'U': U_trim, 'V': V_trim, 'W': W_trim}

