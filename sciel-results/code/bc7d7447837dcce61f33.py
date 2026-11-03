"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i*m+j, j*p+k, k*n+i)
                if abs(T[idx] - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2.
    U: (4,7) indexed by [i*2+j], V: (4,7) indexed by [j*2+k], W: (4,7) indexed by [k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00(b10-b11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b01-b00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b01+b11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p, row_blocks, inner_blocks, col_blocks):
    """
    Build a decomposition by applying Strassen to 2x2x2 blocks where possible.
    row_blocks: partition of rows {0..n-1}
    inner_blocks: partition of inner {0..m-1}
    col_blocks: partition of cols {0..p-1}
    
    For each combination of (row_block, inner_block, col_block):
    - If all sizes are 2, use Strassen rank-7
    - Otherwise, use schoolbook for that sub-block
    """
    U_list = []
    V_list = []
    W_list = []
    
    for rb in row_blocks:
        for ib in inner_blocks:
            for cb in col_blocks:
                r_size = len(rb)
                i_size = len(ib)
                c_size = len(cb)
                
                if r_size == 2 and i_size == 2 and c_size == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2x2()
                    for c in range(7):
                        u_col = [0] * (n * m)
                        v_col = [0] * (m * p)
                        w_col = [0] * (p * n)
                        for ii in range(2):
                            i_val = rb[ii]
                            for jj in range(2):
                                j_val = ib[jj]
                                u_val = sU[ii * 2 + jj][c]
                                if u_val != 0:
                                    u_col[i_val * m + j_val] = u_val
                        for jj in range(2):
                            j_val = ib[jj]
                            for kk in range(2):
                                k_val = cb[kk]
                                v_val = sV[jj * 2 + kk][c]
                                if v_val != 0:
                                    v_col[j_val * p + k_val] = v_val
                        for kk in range(2):
                            k_val = cb[kk]
                            for ii in range(2):
                                i_val = rb[ii]
                                w_val = sW[kk * 2 + ii][c]
                                if w_val != 0:
                                    w_col[k_val * n + i_val] = w_val
                        U_list.append(u_col)
                        V_list.append(v_col)
                        W_list.append(w_col)
                else:
                    # Use schoolbook for this sub-block
                    for ii in range(r_size):
                        i_val = rb[ii]
                        for jj in range(i_size):
                            j_val = ib[jj]
                            for kk in range(c_size):
                                k_val = cb[kk]
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                u_col[i_val * m + j_val] = 1
                                v_col[j_val * p + k_val] = 1
                                w_col[k_val * n + i_val] = 1
                                U_list.append(u_col)
                                V_list.append(v_col)
                                W_list.append(w_col)
    
    rank = len(U_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for idx, val in enumerate(U_list[c]):
            if val != 0:
                U[idx][c] = val
        for idx, val in enumerate(V_list[c]):
            if val != 0:
                V[idx][c] = val
        for idx, val in enumerate(W_list[c]):
            if val != 0:
                W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}

def _all_partitions(lst):
    """Generate all set partitions of a list."""
    if not lst:
        yield []
        return
    first = lst[0]
    rest = lst[1:]
    for partition in _all_partitions(rest):
        # Insert first into each existing block
        for i in range(len(partition)):
            new_partition = [list(block) for block in partition]
            new_partition[i].append(first)
            yield [tuple(block) for block in new_partition]
        # Create new block with first
        yield [tuple([first])] + [tuple(block) for block in partition]

def _construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try all block partitions
    rows = list(range(n))
    inners = list(range(m))
    cols = list(range(p))
    
    row_parts = list(_all_partitions(rows))
    inner_parts = list(_all_partitions(inners))
    col_parts = list(_all_partitions(cols))
    
    for rp in row_parts:
        if time.time() > deadline:
            break
        for ip in inner_parts:
            if time.time() > deadline:
                break
            for cp in col_parts:
                if time.time() > deadline:
                    break
                try:
                    candidate = _build_block_decomposition(n, m, p, rp, ip, cp)
                    rank = len(candidate["U"][0])
                    if rank < best_rank:
                        if _verify_fast(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                            best = candidate
                            best_rank = rank
                except Exception:
                    pass
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    try:
        candidate, rank = _construct()
        if rank < best_rank:
            best = candidate
            best_rank = rank
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
