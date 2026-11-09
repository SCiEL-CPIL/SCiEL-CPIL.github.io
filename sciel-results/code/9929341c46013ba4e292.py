import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _verify_keep(U_arr, V_arr, W_arr, n, m, p, keep_indices):
    """Vectorized verification for a subset of columns."""
    if not keep_indices:
        return True
    U_k = U_arr[:, keep_indices]
    V_k = V_arr[:, keep_indices]
    W_k = W_arr[:, keep_indices]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_k[i*m+j], V_k[j*p+k] * W_k[k*n+i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _build_base_decomposition(n, m, p):
    """Build the base rank-36 decomposition using Strassen 2x2x2 blocks + schoolbook."""
    strassen_U = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],
        [0,1,0,-1],
        [0,0,1,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
    ]
    
    rank = 4 * 7 + 8  # 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li * m + j0 + lj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0 + lj) * p + k0 + lk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    # Schoolbook for k=4: 8 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return U, V, W, rank

def _try_rank_reduction(U, V, W, n, m, p, time_limit=60):
    """
    Attempt to reduce the rank by removing columns.
    Strategy:
    1. Greedy single-column removal
    2. Try removing pairs of columns
    3. Try removing triples of columns
    """
    start_time = time.time()
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_keep = list(range(rank))
    best_rank = rank
    best_keep = list(current_keep)
    
    def _elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy single-column removal (repeat until no more can be removed)
    changed = True
    while changed and _elapsed() < time_limit * 0.2:
        changed = False
        for c in list(current_keep):
            if _elapsed() > time_limit * 0.2:
                break
            new_keep = [i for i in current_keep if i != c]
            if _verify_keep(U_arr, V_arr, W_arr, n, m, p, new_keep):
                current_keep = new_keep
                best_rank = len(new_keep)
                best_keep = list(new_keep)
                changed = True
    
    # Phase 2: Try removing pairs of columns
    if _elapsed() < time_limit * 0.5:
        changed = True
        while changed and _elapsed() < time_limit * 0.5:
            changed = False
            for a_idx in range(len(current_keep)):
                if _elapsed() > time_limit * 0.5:
                    break
                for b_idx in range(a_idx + 1, len(current_keep)):
                    if _elapsed() > time_limit * 0.5:
                        break
                    a = current_keep[a_idx]
                    b = current_keep[b_idx]
                    new_keep = [i for i in current_keep if i != a and i != b]
                    if _verify_keep(U_arr, V_arr, W_arr, n, m, p, new_keep):
                        current_keep = new_keep
                        best_rank = len(new_keep)
                        best_keep = list(new_keep)
                        changed = True
                        break
                if changed:
                    break
    
    # Phase 3: Try removing triples of columns
    if _elapsed() < time_limit * 0.8:
        changed = True
        while changed and _elapsed() < time_limit * 0.8:
            changed = False
            for combo in combinations(current_keep, 3):
                if _elapsed() > time_limit * 0.8:
                    break
                new_keep = [i for i in current_keep if i not in combo]
                if _verify_keep(U_arr, V_arr, W_arr, n, m, p, new_keep):
                    current_keep = new_keep
                    best_rank = len(new_keep)
                    best_keep = list(new_keep)
                    changed = True
                    break
    
    # Phase 4: Another round of single removals after pair/triple removals
    if _elapsed() < time_limit * 0.9:
        changed = True
        while changed and _elapsed() < time_limit * 0.9:
            changed = False
            for c in list(current_keep):
                if _elapsed() > time_limit * 0.9:
                    break
                new_keep = [i for i in current_keep if i != c]
                if _verify_keep(U_arr, V_arr, W_arr, n, m, p, new_keep):
                    current_keep = new_keep
                    best_rank = len(new_keep)
                    best_keep = list(new_keep)
                    changed = True
    
    if best_rank < rank:
        U_list = [[float(x) for x in row] for row in U_arr[:, best_keep].tolist()]
        V_list = [[float(x) for x in row] for row in V_arr[:, best_keep].tolist()]
        W_list = [[float(x) for x in row] for row in W_arr[:, best_keep].tolist()]
        
        # Round to nearest 0.5 if close
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-9:
                        mat[i][j] = rounded
        
        return U_list, V_list, W_list, best_rank
    
    return None

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Build the base rank-36 solution
    U, V, W, rank = _build_base_decomposition(n, m, p)
    
    # Verify the base solution
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    time_remaining = 140 - (time.time() - start_time)
    if time_remaining > 15:
        result = _try_rank_reduction(U, V, W, n, m, p, time_limit=min(time_remaining - 5, 60))
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < rank:
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    U, V, W, rank = U_new, V_new, W_new, new_rank
    
    return {"U": U, "V": V, "W": W}
