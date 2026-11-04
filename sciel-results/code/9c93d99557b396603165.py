"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

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


def _search_rank(r, n, m, p, time_budget, seed, allow_reject=False):
    """Local search for an exact rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    rng = random.Random(seed)

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Precompute triple-to-row mappings
    triple_to_rows = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triple_to_rows[(i,j,k)] = (i*m+j, j*p+k, k*n+i)
    
    # Sparse random init
    density = 0.35
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

    # Precompute current sums for each triple
    sums = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i,j,k)]
                s = 0
                u_row = U[u_idx]
                v_row = V[v_idx]
                w_row = W[w_idx]
                for c in range(r):
                    s += u_row[c] * v_row[c] * w_row[c]
                sums[(i,j,k)] = s

    def compute_error():
        e = 0
        for (i,j,k), s in sums.items():
            e += (s - 1) * (s - 1)
        return e

    cur = compute_error()
    best = cur
    best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
    no_improve = 0
    steps = 0
    max_steps = 15000

    def get_affected_triples(factor, idx, col):
        """Get list of (triple_key, other_factor_idx, other_col_value) for affected triples."""
        if factor == 'U':
            i = idx // m
            j = idx % m
            return [(t, triple_to_rows[t][1], V[triple_to_rows[t][1]][col]) for t in [(i, j, k) for k in range(p)]]
        elif factor == 'V':
            j = idx // p
            k = idx % p
            return [(t, triple_to_rows[t][2], W[triple_to_rows[t][2]][col]) for t in [(i, j, k) for i in range(n)]]
        else:  # 'W'
            k = idx // n
            i = idx % n
            return [(t, triple_to_rows[t][0], U[triple_to_rows[t][0]][col]) for t in [(i, j, k) for j in range(m)]]

    factors = [('U', U, n * m), ('V', V, m * p), ('W', W, p * n)]
    
    while no_improve < 500 and steps < max_steps:
        if time.time() - start > time_budget:
            break
        steps += 1
        
        # Choose factor, row, col
        factor_name, M, nrows = factors[rng.randrange(3)]
        row = rng.randrange(nrows)
        col = rng.randrange(r)
        old = M[row][col]
        
        # Try a new value
        candidates = [v for v in [0, 1, -1] if v != old]
        new = rng.choice(candidates)
        
        # Compute delta error using precomputed sums
        delta_val = new - old
        affected = get_affected_triples(factor_name, row, col)
        
        # Compute new error
        new_err = cur
        for (t, other_idx, other_val) in affected:
            old_sum = sums[t]
            new_sum = old_sum + delta_val * other_val
            new_err -= (old_sum - 1) * (old_sum - 1)
            new_err += (new_sum - 1) * (new_sum - 1)
        
        if new_err < cur:
            # Accept: update sums
            for (t, other_idx, other_val) in affected:
                sums[t] += delta_val * other_val
            M[row][col] = new
            cur = new_err
            no_improve = 0
            if cur < best:
                best = cur
                best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        else:
            # Reject with some probability (simulated annealing-like)
            if allow_reject and rng.random() > 0.1:
                pass  # keep old value
            else:
                # Accept anyway to escape local minima
                for (t, other_idx, other_val) in affected:
                    sums[t] += delta_val * other_val
                M[row][col] = new
                cur = new_err
            no_improve += 1

    Ub, Vb, Wb = best_state
    if _verify(Ub, Vb, Wb, n, m, p):
        return Ub, Vb, Wb
    
    # Try a final polish: greedy improvement on the best state
    U2, V2, W2 = [row[:] for row in Ub], [row[:] for row in Vb], [row[:] for row in Wb]
    # Recompute sums for polish
    sums2 = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i,j,k)]
                s = 0
                for c in range(r):
                    s += U2[u_idx][c] * V2[v_idx][c] * W2[w_idx][c]
                sums2[(i,j,k)] = s
    
    cur2 = sum((s-1)**2 for s in sums2.values())
    improved = True
    polish_steps = 0
    while improved and polish_steps < 500:
        improved = False
        polish_steps += 1
        for factor_name, M2, nrows in [('U', U2, n*m), ('V', V2, m*p), ('W', W2, p*n)]:
            for row in range(nrows):
                for col in range(r):
                    old = M2[row][col]
                    for new in [v for v in [0,1,-1] if v != old]:
                        delta_val = new - old
                        affected = get_affected_triples(factor_name, row, col)
                        new_err2 = cur2
                        for (t, other_idx, other_val) in affected:
                            # Use original V/W/U values from best_state, not M2
                            if factor_name == 'U':
                                ov = Vb[other_idx][col]
                            elif factor_name == 'V':
                                ov = Wb[other_idx][col]
                            else:
                                ov = Ub[other_idx][col]
                            old_sum = sums2[t]
                            new_sum = old_sum + delta_val * ov
                            new_err2 -= (old_sum - 1) * (old_sum - 1)
                            new_err2 += (new_sum - 1) * (new_sum - 1)
                        if new_err2 < cur2:
                            for (t, other_idx, other_val) in affected:
                                if factor_name == 'U':
                                    sums2[t] += delta_val * Vb[other_idx][col]
                                elif factor_name == 'V':
                                    sums2[t] += delta_val * Wb[other_idx][col]
                                else:
                                    sums2[t] += delta_val * Ub[other_idx][col]
                            M2[row][col] = new
                            cur2 = new_err2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    if _verify(U2, V2, W2, n, m, p):
        return U2, W2, V2  # Note: returning in order U, V, W
    return None


def construct():
    start_time = time.time()
    timeout = 140.0  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS

    # Guaranteed valid baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = {"U": U0, "V": V0, "W": W0}
    best_rank = n * m * p

    # Try progressively lower ranks with existing search
    # The rank of 3x3 matrix multiplication is known to be 20.
    # We try 23, 22, 21, 20.
    for target_rank in range(23, 19, -1):
        if time.time() - start_time > timeout - 8:
            break
        # More seeds for lower ranks
        num_seeds = 10 if target_rank <= 20 else 6
        for seed in range(num_seeds):
            if time.time() - start_time > timeout - 8:
                break
            # Allow simulated annealing rejections for harder ranks
            allow_reject = target_rank <= 21
            res = _search_rank(target_rank, n, m, p, time_budget=10.0, seed=seed, allow_reject=allow_reject)
            if res is not None:
                U, V, W = res
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # found valid at this rank, try next (lower) rank
        if best_rank < n * m * p:
            # If we found something better than schoolbook, we could stop if we hit 20
            if best_rank == 20:
                break

    # Final safety: ensure we return something valid
    if time.time() - start_time > timeout - 1:
        return best

    return best
# EVOLVE-BLOCK-END
