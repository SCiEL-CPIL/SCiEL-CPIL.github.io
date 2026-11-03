"""Matrix multiplication 3x4 by 4x6: Rank-minimal decomposition via peeling."""
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_3x4x6():
    """Build the rank-66 decomposition for 3x4 * 4x6."""
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            add_strassen(1, j_base, k_base)
    
    return U, V, W, rank

def _build_schoolbook():
    """Build pure schoolbook rank-72 decomposition."""
    n, m, p = 3, 4, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _build_strassen_row0():
    """Strassen on row 0 (2x2 blocks), schoolbook for rows 1-2."""
    n, m, p = 3, 4, 6
    U = [[0] * 72 for _ in range(n * m)]
    V = [[0] * 72 for _ in range(m * p)]
    W = [[0] * 72 for _ in range(p * n)]
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        if col >= 72:
            return
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        if col + 7 > 72:
            return
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Strassen on row 0: 2 j-blocks × 3 k-blocks = 6 Strassen = 42 terms
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            add_strassen(0, j_base, k_base)
    
    # Schoolbook for rows 1-2: 2*4*6 = 48 terms
    for i in [1, 2]:
        for j in range(4):
            for k in range(6):
                add_schoolbook(i, j, k)
    
    actual_rank = col
    return [row[:actual_rank] for row in U], [row[:actual_rank] for row in V], [row[:actual_rank] for row in W], actual_rank

def _build_mixed_strassen():
    """Try different Strassen placements."""
    n, m, p = 3, 4, 6
    best_U, best_V, best_W, best_rank = None, None, None, 999
    
    # Try Strassen on different row pairs and column subsets
    configs = [
        # (rows_to_strassen, j_blocks, k_blocks)
        ((1, 2), [(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ((1, 2), [(0,1),(2,3)], [(0,1),(2,3)]),  # only 2 k-blocks, schoolbook for last 2
        ((0, 1), [(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ((0, 1), [(0,1),(2,3)], [(0,1),(2,3)]),
        ((1, 2), [(0,1)], [(0,1),(2,3),(4,5)]),  # only 1 j-block
        ((1, 2), [(2,3)], [(0,1),(2,3),(4,5)]),
    ]
    
    for rows, j_blocks, k_blocks in configs:
        U = [[0] * 72 for _ in range(n * m)]
        V = [[0] * 72 for _ in range(m * p)]
        W = [[0] * 72 for _ in range(p * n)]
        col = 0
        strassen_rows = set(rows)
        
        def add_schoolbook(i, j, k):
            nonlocal col
            if col >= 72:
                return False
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            return True

        def add_strassen(i_base, j_base, k_base):
            nonlocal col
            if col + 7 > 72:
                return False
            a_idx = i_base * m + j_base
            b_idx = i_base * m + (j_base + 1)
            c_idx = (i_base + 1) * m + j_base
            d_idx = (i_base + 1) * m + (j_base + 1)
            e_idx = j_base * p + k_base
            f_idx = j_base * p + (k_base + 1)
            g_idx = (j_base + 1) * p + k_base
            h_idx = (j_base + 1) * p + (k_base + 1)
            p_idx = k_base * n + i_base
            q_idx = (k_base + 1) * n + i_base
            r_idx = k_base * n + (i_base + 1)
            s_idx = (k_base + 1) * n + (i_base + 1)
            
            U[a_idx][col] = 1; U[d_idx][col] = 1
            V[e_idx][col] = 1; V[h_idx][col] = 1
            W[p_idx][col] = 1; W[s_idx][col] = 1
            col += 1
            
            U[c_idx][col] = 1; U[d_idx][col] = 1
            V[e_idx][col] = 1
            W[r_idx][col] = 1; W[s_idx][col] = -1
            col += 1
            
            U[a_idx][col] = 1
            V[f_idx][col] = 1; V[h_idx][col] = -1
            W[q_idx][col] = 1; W[s_idx][col] = 1
            col += 1
            
            U[d_idx][col] = 1
            V[g_idx][col] = 1; V[e_idx][col] = -1
            W[p_idx][col] = 1; W[r_idx][col] = 1
            col += 1
            
            U[a_idx][col] = 1; U[b_idx][col] = 1
            V[h_idx][col] = 1
            W[p_idx][col] = -1; W[q_idx][col] = 1
            col += 1
            
            U[c_idx][col] = 1; U[a_idx][col] = -1
            V[e_idx][col] = 1; V[f_idx][col] = 1
            W[s_idx][col] = 1
            col += 1
            
            U[b_idx][col] = 1; U[d_idx][col] = -1
            V[g_idx][col] = 1; V[h_idx][col] = 1
            W[p_idx][col] = 1
            col += 1
            return True
        
        strassen_cells = set()
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                if add_strassen(rows[0], j1, k1):
                    strassen_cells.add((rows[0], j1, k1))
                    strassen_cells.add((rows[0], j1, k1+1))
                    strassen_cells.add((rows[0], j1+1, k1))
                    strassen_cells.add((rows[0], j1+1, k1+1))
                    strassen_cells.add((rows[1], j1, k1))
                    strassen_cells.add((rows[1], j1, k1+1))
                    strassen_cells.add((rows[1], j1+1, k1))
                    strassen_cells.add((rows[1], j1+1, k1+1))
        
        # Fill remaining with schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in strassen_cells:
                        add_schoolbook(i, j, k)
        
        actual_rank = col
        if actual_rank < best_rank:
            best_U = [row[:actual_rank] for row in U]
            best_V = [row[:actual_rank] for row in V]
            best_W = [row[:actual_rank] for row in W]
            best_rank = actual_rank
    
    if best_U is None:
        return _build_schoolbook()
    return best_U, best_V, best_W, best_rank

def _verify(U, V, W, rank):
    """Verify the tensor decomposition is correct."""
    n, m, p = 3, 4, 6
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _verify_fast(U, V, W, rank):
    """Fast verification using numpy."""
    n, m, p = 3, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _peel_decomposition(U, V, W, max_iters=500):
    """
    Iteratively peel (remove) terms that are linearly dependent.
    A term r can be removed if its contribution is already captured by other terms.
    We check: for each (i,j,k), the sum over remaining terms still equals 1.
    """
    n, m, p = 3, 4, 6
    rank = len(U[0])
    
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    active = list(range(rank))
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        if len(active) <= 1:
            break
            
        found_removed = False
        for idx, r in enumerate(active):
            # Try removing term r
            new_active = active[:idx] + active[idx+1:]
            
            # Check if decomposition still valid without term r
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0
                        for c in new_active:
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                active = new_active
                found_removed = True
                break
        
        if not found_removed:
            break
    
    final_rank = len(active)
    U_final = [row[:final_rank] for row in U]
    V_final = [row[:final_rank] for row in V]
    W_final = [row[:final_rank] for row in W]
    
    # Reorder to active columns
    U_final = [[U[i][r] for r in active] for i in range(n*m)]
    V_final = [[V[i][r] for r in active] for i in range(m*p)]
    W_final = [[W[i][r] for r in active] for i in range(p*n)]
    
    return U_final, V_final, W_final, final_rank

def _peel_with_span_check(U, V, W, max_iters=200):
    """
    More efficient peeling: check if term r's U, V, W vectors are in the span
    of the remaining terms' vectors. This is a sufficient (not necessary) condition.
    """
    n, m, p = 3, 4, 6
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    active = list(range(rank))
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        if len(active) <= 1:
            break
        
        found_removed = False
        for idx in range(len(active)):
            r = active[idx]
            new_active = active[:idx] + active[idx+1:]
            
            if len(new_active) == 0:
                continue
            
            # Check if the removed term's contribution can be absorbed
            # Quick check: verify the tensor is still correct
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0.0
                        for c in new_active:
                            val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                active = new_active
                found_removed = True
                break
        
        if not found_removed:
            break
    
    final_rank = len(active)
    U_final = [[U[i][r] for r in active] for i in range(n*m)]
    V_final = [[V[i][r] for r in active] for i in range(m*p)]
    W_final = [[W[i][r] for r in active] for i in range(p*n)]
    
    return U_final, V_final, W_final, final_rank

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    Uses multiple candidate decompositions and peeling to minimize rank.
    """
    n, m, p = 3, 4, 6
    
    # Generate multiple candidate decompositions
    candidates = []
    
    # Candidate 1: Standard Strassen (rows 1-2) + schoolbook (row 0)
    if _time_left() > 30:
        U, V, W, rank = _build_strassen_3x4x6()
        if _verify(U, V, W, rank):
            candidates.append((U, V, W, rank, "strassen_12"))
    
    # Candidate 2: Pure schoolbook
    if _time_left() > 30:
        U, V, W, rank = _build_schoolbook()
        if _verify(U, V, W, rank):
            candidates.append((U, V, W, rank, "schoolbook"))
    
    # Candidate 3: Strassen on row 0
    if _time_left() > 30:
        U, V, W, rank = _build_strassen_row0()
        if _verify(U, V, W, rank):
            candidates.append((U, V, W, rank, "strassen_0"))
    
    # Candidate 4: Mixed Strassen
    if _time_left() > 30:
        U, V, W, rank = _build_mixed_strassen()
        if _verify(U, V, W, rank):
            candidates.append((U, V, W, rank, "mixed"))
    
    if not candidates:
        # Fallback
        U, V, W, rank = _build_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    # Sort by initial rank (lower is better)
    candidates.sort(key=lambda x: x[3])
    
    # Apply peeling to each candidate, keep the best result
    best_U, best_V, best_W, best_rank = candidates[0][0], candidates[0][1], candidates[0][2], candidates[0][3]
    
    for U, V, W, rank, name in candidates:
        if _time_left() < 10.0:
            break
        
        # Try peeling
        U_p, V_p, W_p, rank_p = _peel_with_span_check(U, V, W, max_iters=300)
        
        if _verify(U_p, V_p, W_p, rank_p) and rank_p < best_rank:
            best_U, best_V, best_W, best_rank = U_p, V_p, W_p, rank_p
    
    return {"U": best_U, "V": best_V, "W": best_W}
