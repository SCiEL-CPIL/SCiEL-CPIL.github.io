"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Exact check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    s += u_row[c] * v_row[c] * w_row[c]
                if s != 1:
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
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _search_rank(r, n, m, p, time_budget, seed, density=0.5):
    """Local search for an exact rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    rng = random.Random(seed)

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Precompute index maps for fast affected-triple lookup
    u_map = {}
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            u_map[u_idx] = [(i, j, k) for k in range(p)]
    v_map = {}
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            v_map[v_idx] = [(i, j, k) for i in range(n)]
    w_map = {}
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            w_map[w_idx] = [(i, j, k) for j in range(m)]

    triple_to_rows = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triple_to_rows[(i, j, k)] = (i * m + j, j * p + k, k * n + i)

    # Initialize with random values
    for row in range(n * m):
        for col in range(r):
            if rng.random() < density:
                U[row][col] = rng.choice([1, -1])
    for row in range(m * p):
        for col in range(r):
            if rng.random() < density:
                V[row][col] = rng.choice([1, -1])
    for row in range(p * n):
        for col in range(r):
            if rng.random() < density:
                W[row][col] = rng.choice([1, -1])

    # Compute initial sums
    sums = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i, j, k)]
                s = 0
                u_row = U[u_idx]
                v_row = V[v_idx]
                w_row = W[w_idx]
                for c in range(r):
                    s += u_row[c] * v_row[c] * w_row[c]
                sums[(i, j, k)] = s

    def compute_error():
        e = 0
        for (i, j, k), s in sums.items():
            e += (s - 1) * (s - 1)
        return e

    cur = compute_error()
    best = cur
    best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
    no_improve = 0
    steps = 0
    max_steps = 15000

    factors = [('U', U, n * m), ('V', V, m * p), ('W', W, p * n)]

    def get_affected_triples(factor, idx, col):
        if factor == 'U':
            triples = u_map[idx]
            return [(t, triple_to_rows[t][1], V[triple_to_rows[t][1]][col]) for t in triples]
        elif factor == 'V':
            triples = v_map[idx]
            return [(t, triple_to_rows[t][2], W[triple_to_rows[t][2]][col]) for t in triples]
        else:
            triples = w_map[idx]
            return [(t, triple_to_rows[t][0], U[triple_to_rows[t][0]][col]) for t in triples]

    while no_improve < 400 and steps < max_steps:
        if time.time() - start > time_budget:
            break
        steps += 1

        factor_name, M, nrows = factors[rng.randrange(3)]
        row = rng.randrange(nrows)
        col = rng.randrange(r)
        old = M[row][col]

        candidates = [v for v in [0, 1, -1] if v != old]
        new = rng.choice(candidates)

        delta_val = new - old
        affected = get_affected_triples(factor_name, row, col)

        new_err = cur
        for (t, other_idx, other_val) in affected:
            old_sum = sums[t]
            new_sum = old_sum + delta_val * other_val
            new_err -= (old_sum - 1) * (old_sum - 1)
            new_err += (new_sum - 1) * (new_sum - 1)

        if new_err < cur:
            for (t, other_idx, other_val) in affected:
                sums[t] += delta_val * other_val
            M[row][col] = new
            cur = new_err
            no_improve = 0
            if cur < best:
                best = cur
                best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        else:
            # Simulated annealing: accept worse moves with decreasing probability
            if rng.random() < 0.1:
                for (t, other_idx, other_val) in affected:
                    sums[t] += delta_val * other_val
                M[row][col] = new
                cur = new_err
            no_improve += 1

    Ub, Vb, Wb = best_state
    if _verify(Ub, Vb, Wb, n, m, p):
        return Ub, Vb, Wb
    return None


def construct():
    start_time = time.time()
    timeout = 140.0

    n, m, p = DIMENSIONS

    # Start with schoolbook rank 27 as fallback
    U0, V0, W0 = _schoolbook(n, m, p)
    best = {"U": U0, "V": V0, "W": W0}
    best_rank = n * m * p

    # Try to find rank 23 or lower
    for target_rank in range(23, 18, -1):
        if time.time() - start_time > timeout - 8:
            break
        for seed in range(5):
            if time.time() - start_time > timeout - 8:
                break
            res = _search_rank(target_rank, n, m, p, time_budget=12.0, seed=seed)
            if res is not None:
                U, V, W = res
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break

    return best
# EVOLVE-BLOCK-END
