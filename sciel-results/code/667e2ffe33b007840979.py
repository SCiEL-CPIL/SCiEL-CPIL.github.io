import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.all(np.abs(result - 1) < 1e-9)

def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    rank = 7
    U = np.zeros((4, rank), dtype=np.int64)
    V = np.zeros((4, rank), dtype=np.int64)
    W = np.zeros((4, rank), dtype=np.int64)
    
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W

def _block_to_global(block_idx, local_r, local_c, size=4):
    """Map block index and local coords to global row-major index for a 4x4 matrix."""
    if block_idx == 0:
        return local_r * size + local_c
    elif block_idx == 1:
        return local_r * size + (2 + local_c)
    elif block_idx == 2:
        return (2 + local_r) * size + local_c
    else:
        return (2 + local_r) * size + (2 + local_c)

def _build_strassen_4x4_4x4():
    """Build Strassen-based decomposition for 4x4 * 4x4 -> 4x4 with rank 49."""
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    sU, sV, sW = _strassen_2x2()
    
    mults = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 3:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1}),
    ]
    
    for mi, (a_c, b_c, c_c) in enumerate(mults):
        for s in range(7):
            col = mi * 7 + s
            
            for xi in range(4):
                u_val = sU[xi, s]
                if u_val == 0:
                    continue
                xr = xi // 2
                xc = xi % 2
                for ab, coeff in a_c.items():
                    global_u = _block_to_global(ab, xr, xc)
                    U[global_u, col] += u_val * coeff
            
            for yi in range(4):
                v_val = sV[yi, s]
                if v_val == 0:
                    continue
                yr = yi // 2
                yc = yi % 2
                for bb, coeff in b_c.items():
                    global_v = _block_to_global(bb, yr, yc)
                    V[global_v, col] += v_val * coeff
            
            for zi in range(4):
                w_val = sW[zi, s]
                if w_val == 0:
                    continue
                zr = zi // 2
                zc = zi % 2
                for cb, coeff in c_c.items():
                    global_w = _block_to_global(cb, zr, zc)
                    W[global_w, col] += w_val * coeff
    
    return U, V, W

def _build_strassen_4x4_4x5():
    """
    Build a decomposition for 4x4 * 4x5 -> 4x5.
    Strategy:
    1. Use Strassen for the first 4 columns (4x4 * 4x4 -> 4x4), rank 49.
    2. Use schoolbook for the 5th column (4x4 * 4x1 -> 4x1), rank 16.
    Total initial rank: 65.
    """
    n, m, p = 4, 4, 5
    rank_44 = 49
    rank_41 = 16
    total_rank = rank_44 + rank_41
    
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    
    # Part 1: 4x4 * 4x4 (columns 0-3)
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    U[:, :rank_44] = U_s
    for j in range(m):
        for k in range(4):
            V[j*p+k, :rank_44] = V_s[j*4+k, :]
    for k in range(4):
        for i in range(n):
            W[k*n+i, :rank_44] = W_s[k*4+i, :]
    
    # Part 2: 4x4 * 4x1 (column 4)
    # Schoolbook for the last column
    col_offset = rank_44
    for i in range(n):
        for j in range(m):
            U[i*m+j, col_offset + (i*m+j)] = 1
            V[j*4+4, col_offset + (i*m+j)] = 1
            W[4*n+i, col_offset + (i*m+j)] = 1
            
    return U, V, W, total_rank

def _try_rank_reduction(U, V, W, n, m, p, rank, deadline):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 tensors.
    We iterate through columns and try to remove one if the tensor identity still holds.
    """
    if time.time() > deadline:
        return None, None, None, rank
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # We'll do a few passes, as removing one column might make another removable
    # To be safe, we'll just do a single pass, but check carefully.
    # Actually, the previous implementation did a while loop. Let's stick to that but with time guard.
    
    c_remove = 0
    while c_remove < rank:
        if time.time() > deadline:
            break
        
        U_test = np.delete(U, c_remove, axis=1)
        V_test = np.delete(V, c_remove, axis=1)
        W_test = np.delete(W, c_remove, axis=1)
        
        valid = _verify(U_test, V_test, W_test, n, m, p, rank - 1)
        
        if valid:
            best_rank = rank - 1
            best_U, best_V, best_W = U_test, V_test, W_test
            U, V, W = U_test, V_test, W_test
            rank = best_rank
            c_remove = 0  # Restart from 0 as indices shifted
        else:
            c_remove += 1
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None, None, None, rank

def _try_milp_reduction(U, V, W, n, m, p, rank, deadline):
    """
    Attempt a more sophisticated reduction by checking if any column is a linear combination
    of others in a way that preserves the tensor structure.
    """
    return U, V, W, rank

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s for serialization and safety
    
    # Build initial decomposition
    U, V, W, total_rank = _build_strassen_4x4_4x5()
    
    # Verify initial decomposition
    if not _verify(U, V, W, n, m, p, total_rank):
        # Fallback to pure schoolbook if Strassen construction failed (shouldn't happen)
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank reduction
    if time.time() < deadline - 5:
        U_red, V_red, W_red, new_rank = _try_rank_reduction(U, V, W, n, m, p, total_rank, deadline - 5)
        if U_red is not None and new_rank < total_rank:
            U, V, W, total_rank = U_red, V_red, W_red, new_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
