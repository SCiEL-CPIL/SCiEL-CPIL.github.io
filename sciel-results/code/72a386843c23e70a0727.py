import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Strategy: Build a 77-rank decomposition using Strassen recursion, then attempt
    to reduce rank by eliminating linearly dependent or redundant columns.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # PART 1: 4x4 * 4x4 (columns 0-3 of B) using recursive Strassen (rank 49)
    if _time_remaining() > 5.0:
        _add_strassen_4x4(0, 0, 0, U_cols, V_cols, W_cols, n, m, p)
    
    # PART 2: 4x4 * 4x2 (columns 4-5 of B) using 4 independent 2x2*2x2 Strassen (rank 28)
    if _time_remaining() > 3.0:
        _add_strassen_2x2(0, 0, 4, U_cols, V_cols, W_cols, n, m, p)
        _add_strassen_2x2(0, 2, 4, U_cols, V_cols, W_cols, n, m, p)
        _add_strassen_2x2(2, 0, 4, U_cols, V_cols, W_cols, n, m, p)
        _add_strassen_2x2(2, 2, 4, U_cols, V_cols, W_cols, n, m, p)
    
    rank = len(U_cols)
    
    # Attempt rank reduction by removing linearly dependent columns
    if _time_remaining() > 10.0:
        U_cols, V_cols, W_cols = _reduce_rank(U_cols, V_cols, W_cols, n, m, p)
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: trivial rank-96 decomposition
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _strassen_2x2_offsets():
    """Return list of (u_dict, v_dict, w_dict) for 2x2 Strassen."""
    return [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

def _add_strassen_2x2(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    for u_pat, v_pat, w_pat in _strassen_2x2_offsets():
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (r, c), coeff in u_pat.items():
            ii = oi + r
            jj = oj + c
            if 0 <= ii < n and 0 <= jj < m:
                u_col[ii * m + jj] += coeff
        
        for (r, c), coeff in v_pat.items():
            jj = oj + r
            kk = ok + c
            if 0 <= jj < m and 0 <= kk < p:
                v_col[jj * p + kk] += coeff
        
        for (r, c), coeff in w_pat.items():
            ii = oi + r
            kk = ok + c
            if 0 <= ii < n and 0 <= kk < p:
                w_col[kk * n + ii] += coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

def _strassen_4x4_block_combos():
    """
    Return list of (u_block_map, v_block_map, w_block_map) for 4x4 Strassen.
    Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) at the 2x2 block level.
    """
    return [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]

def _add_strassen_4x4(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    block_offsets = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
    base_2x2 = _strassen_2x2_offsets()
    
    for u_blocks, v_blocks, w_blocks in _strassen_4x4_block_combos():
        for u_pat, v_pat, w_pat in base_2x2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for bidx, bcoeff in u_blocks.items():
                br, bc = block_offsets[bidx]
                for (r, c), coeff in u_pat.items():
                    ii = oi + br + r
                    jj = oj + bc + c
                    if 0 <= ii < n and 0 <= jj < m:
                        u_col[ii * m + jj] += bcoeff * coeff
            
            for bidx, bcoeff in v_blocks.items():
                br, bc = block_offsets[bidx]
                for (r, c), coeff in v_pat.items():
                    jj = oj + br + r
                    kk = ok + bc + c
                    if 0 <= jj < m and 0 <= kk < p:
                        v_col[jj * p + kk] += bcoeff * coeff
            
            for bidx, bcoeff in w_blocks.items():
                br, bc = block_offsets[bidx]
                for (r, c), coeff in w_pat.items():
                    ii = oi + br + r
                    kk = ok + bc + c
                    if 0 <= ii < n and 0 <= kk < p:
                        w_col[kk * n + ii] += bcoeff * coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

def _reduce_rank(U_cols, V_cols, W_cols, n, m, p):
    """
    Attempt to reduce rank by finding and removing redundant columns.
    A column r is redundant if its contribution can be expressed as a linear
    combination of other columns.
    """
    U_arr = np.array(U_cols, dtype=np.float64).T  # shape (rank, n*m)
    V_arr = np.array(V_cols, dtype=np.float64).T  # shape (rank, m*p)
    W_arr = np.array(W_cols, dtype=np.float64).T  # shape (rank, p*n)
    
    rank = len(U_cols)
    kept = list(range(rank))
    
    # Try to remove each column if it's linearly dependent
    for col_idx in range(rank - 1, -1, -1):
        if col_idx not in kept:
            continue
        if _time_remaining() < 5.0:
            break
        
        # Check if this column is linearly dependent on the others
        others = [k for k in kept if k != col_idx]
        if not others:
            continue
        
        # Build matrix of other columns
        U_others = U_arr[others, :]  # (len(others), n*m)
        V_others = V_arr[others, :]  # (len(others), m*p)
        W_others = W_arr[others, :]  # (len(others), p*n)
        
        # Check if U[col_idx] is in span of U_others
        U_target = U_arr[col_idx, :]
        # Solve least squares
        try:
            coeffs, residuals, rank_u, sv = np.linalg.lstsq(U_others, U_target, rcond=1e-10)
            if rank_u < len(others):
                # Check if reconstruction is exact
                U_recon = U_others @ coeffs
                if np.allclose(U_recon, U_target, atol=1e-10):
                    # Also check V and W
                    V_target = V_arr[col_idx, :]
                    V_recon = V_others @ coeffs
                    if np.allclose(V_recon, V_target, atol=1e-10):
                        W_target = W_arr[col_idx, :]
                        W_recon = W_others @ coeffs
                        if np.allclose(W_recon, W_target, atol=1e-10):
                            kept.remove(col_idx)
        except:
            pass
    
    if len(kept) == rank:
        return U_cols, V_cols, W_cols
    
    U_new = [U_cols[i] for i in kept]
    V_new = [V_cols[i] for i in kept]
    W_new = [W_cols[i] for i in kept]
    
    return U_new, V_new, W_new

def _verify_decomposition(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = int(np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :]))
                if total != 1:
                    return False
    return True
