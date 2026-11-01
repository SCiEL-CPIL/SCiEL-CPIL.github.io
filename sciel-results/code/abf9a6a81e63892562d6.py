"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 2, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

def _strassen_block_indices(k0, k1):
    """Return the 7 (u_entries, v_entries, w_entries) for one Strassen block."""
    p = DIMENSIONS[2]
    n = DIMENSIONS[0]
    # V indices
    v_e = 0 * p + k0
    v_f = 0 * p + k1
    v_g = 1 * p + k0
    v_h = 1 * p + k1
    # W indices
    w_c00 = k0 * n + 0
    w_c10 = k0 * n + 1
    w_c01 = k1 * n + 0
    w_c11 = k1 * n + 1

    blocks = []
    # P1 = (a+d)(e+h)
    blocks.append(([0,3],[v_e,v_h],[w_c00,w_c11]))
    # P2 = (c+d)e
    blocks.append(([2,3],[v_e],[w_c10,w_c11]))
    # P3 = a(f-h)
    blocks.append(([0],[v_f,v_h],[w_c01,w_c11]))
    # P4 = d(g-e)
    blocks.append(([3],[v_g,v_e],[w_c00,w_c10]))
    # P5 = (a+b)h
    blocks.append(([0,1],[v_h],[w_c00,w_c01]))
    # P6 = (c-a)(e+f)
    blocks.append(([2,0],[v_e,v_f],[w_c11]))
    # P7 = (b-d)(g+h)
    blocks.append(([1,3],[v_g,v_h],[w_c00]))
    return blocks

def _get_strassen_decomposition(n, m, p):
    """Build the standard 3-block Strassen decomposition, rank = 3*7 = 21."""
    rank = 3 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        for step, (u_idx, v_idx, w_idx) in enumerate(_strassen_block_indices(k0, k1)):
            r = b * 7 + step
            for u in u_idx:
                U[u][r] = 1
            for v in v_idx:
                V[v][r] = 1
            for w in w_idx:
                W[w][r] = 1
            # Apply signs (the Strassen formulas have some -1 coefficients)
            # We hardcode the signs here
            if step == 1: W[w_idx[1]][r] = -1
            elif step == 2: V[v_idx[1]][r] = -1
            elif step == 3: V[v_idx[1]][r] = -1
            elif step == 4: W[w_idx[0]][r] = -1
            elif step == 5: U[u_idx[1]][r] = -1
            elif step == 6: U[u_idx[1]][r] = -1
    return U, V, W

def _try_rank_reduction(n, m, p, U, V, W, current_rank, max_rank_drop=3):
    """
    Try to reduce the rank by identifying linearly dependent columns
    in the (U,V,W) triple and eliminating them via basis change.
    This is a heuristic; for this small problem it may not find reductions
    but we implement a simple check.
    """
    best_rank = current_rank
    best_U, best_V, best_W = U[:], V[:], W[:]

    for target_rank in range(current_rank - max_rank_drop, 0, -1):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        # Simple heuristic: check if we can drop columns
        # by seeing if the product is still correct without them.
        # This is expensive; for a quick check, we sample random A, B.
        import random
        found_reduced = False
        for trial in range(20):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            # Try dropping one column at a time
            for col_to_drop in range(current_rank):
                # Build candidate without this column
                U_cand = [[U[r][c] for c in range(current_rank) if c != col_to_drop] for r in range(n*m)]
                V_cand = [[V[r][c] for c in range(current_rank) if c != col_to_drop] for r in range(m*p)]
                W_cand = [[W[r][c] for c in range(current_rank) if c != col_to_drop] for r in range(p*n)]

                # Verify with random matrices
                valid = True
                for _ in range(5):
                    A_rand = [[random.randint(-5,5) for _ in range(m)] for _ in range(n)]
                    B_rand = [[random.randint(-5,5) for _ in range(p)] for _ in range(m)]
                    # Compute C = A * B
                    C_true = [[sum(A_rand[i][j] * B_rand[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
                    # Compute via decomposition
                    C_test = [[0]*p for _ in range(n)]
                    for r in range(current_rank - 1):
                        u_prod = sum(U_cand[i*m+j][r] * A_rand[i][j] for i in range(n) for j in range(m))
                        v_prod = sum(V_cand[j*p+k][r] * B_rand[j][k] for j in range(m) for k in range(p))
                        for i in range(n):
                            for k in range(p):
                                C_test[i][k] += u_prod * v_prod * W_cand[k*n+i][r]
                    for i in range(n):
                        for k in range(p):
                            if C_test[i][k] != C_true[i][k]:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break

                if valid:
                    found_reduced = True
                    best_U, best_V, best_W = U_cand, V_cand, W_cand
                    best_rank = current_rank - 1
                    current_rank -= 1
                    break
            if found_reduced:
                break
            if not found_reduced:
                break
        if found_reduced:
            continue
        else:
            break

    if best_rank < current_rank:
        return best_U, best_V, best_W, best_rank
    return U, V, W, current_rank

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT:
        # Return the schoolbook decomposition as ultimate fallback
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    n, m, p = DIMENSIONS
    U, V, W = _get_strassen_decomposition(n, m, p)
    current_rank = len(U[0])

    # Attempt rank reduction
    if time.time() - START_TIME < TIME_LIMIT - 10:
        U, V, W, current_rank = _try_rank_reduction(n, m, p, U, V, W, current_rank, max_rank_drop=5)

    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
