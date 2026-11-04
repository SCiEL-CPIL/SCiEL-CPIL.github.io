import time
import sys
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # 20 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            k1 = k0 + 1
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Optimization: Try to find rank 43 ----
    # Check if any column is linearly dependent in the tensor space
    # We check if col_i = sum(alpha_j * col_j) is possible?
    # For small ranks, we can try removing a column and solving for a correction in another column.
    # But entries must be integer/half-integer.
    # Simple check: Duplicate columns.
    
    def _try_remove_duplicates():
        nonlocal best_candidate, best_rank
        U = best_candidate["U"]
        V = best_candidate["V"]
        W = best_candidate["W"]
        rank = len(U[0])
        
        seen = {}
        for c in range(rank):
            # Create a key for the column tuple
            u_tuple = tuple(U[i][c] for i in range(n*m))
            v_tuple = tuple(V[i][c] for i in range(m*p))
            w_tuple = tuple(W[i][c] for i in range(p*n))
            key = (u_tuple, v_tuple, w_tuple)
            
            if key in seen:
                # Duplicate found
                dup_idx = seen[key]
                # Try removing c
                new_rank = rank - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for cc in range(rank):
                    if cc == c:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][cc]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][cc]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][cc]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return
            else:
                seen[key] = c
    _try_remove_duplicates()
    
    if _elapsed_ok():
        # Try to find rank 43 by checking if col_i + col_j = col_k (then remove k)
        # Or col_i = col_j + col_k
        U = best_candidate["U"]
        V = best_candidate["V"]
        W = best_candidate["W"]
        rank = len(U[0])
        
        # Precompute column tensors
        cols = []
        for c in range(rank):
            vec = [0] * num_terms
            for i in range(n):
                for a in range(m):
                    u_val = U[i * m + a][c]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = V[a * p + k][c]
                        if v_val == 0: continue
                        w_val = W[k * n + i][c]
                        if w_val == 0: continue
                        vec[i * m * p + a * p + k] = u_val * v_val * w_val
            cols.append(vec)
            
        found_improvement = False
        for i in range(rank):
            if found_improvement or not _elapsed_ok(): break
            for j in range(rank):
                if found_improvement or not _elapsed_ok(): break
                if i == j: continue
                for k in range(rank):
                    if found_improvement or not _elapsed_ok(): break
                    if i == k or j == k: continue
                    # Check if cols[i] + cols[j] == cols[k]
                    match = True
                    for idx in range(num_terms):
                        if cols[i][idx] + cols[j][idx] != cols[k][idx]:
                            match = False
                            break
                    if match:
                        # Remove k
                        new_rank = rank - 1
                        new_U = [[0] * new_rank for _ in range(n * m)]
                        new_V = [[0] * new_rank for _ in range(m * p)]
                        new_W = [[0] * new_rank for _ in range(p * n)]
                        nc = 0
                        for cc in range(rank):
                            if cc == k:
                                continue
                            for ii in range(n * m):
                                new_U[ii][nc] = U[ii][cc]
                            for ii in range(m * p):
                                new_V[ii][nc] = V[ii][cc]
                            for ii in range(p * n):
                                new_W[ii][nc] = W[ii][cc]
                            nc += 1
                        cand = {"U": new_U, "V": new_V, "W": new_W}
                        if _validate(cand):
                            best_candidate = cand
                            best_rank = new_rank
                            found_improvement = True
                            break
                            
    return best_candidate

# EVOLVE-BLOCK-END
