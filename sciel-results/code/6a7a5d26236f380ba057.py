"""Matrix multiplication 3x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def strassen_terms():
        terms = []
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ))
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ))
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        return terms

    def get_partitions(total):
        """Generate all integer partitions of 'total' as tuples."""
        if total == 0:
            return [()]
        if total == 1:
            return [(1,)]
        result = []
        for first in range(1, total + 1):
            for rest in get_partitions(total - first):
                if not rest or first <= rest[0]:
                    result.append((first,) + rest)
        return result

    def get_all_partitions(total):
        """Generate all ordered partitions (compositions) of 'total'."""
        if total == 0:
            return [()]
        result = []
        for first in range(1, total + 1):
            for rest in get_all_partitions(total - first):
                result.append((first,) + rest)
        return result

    # Precompute optimal cost for all block sizes
    max_n, max_m, max_p = 3, 5, 6
    cost = [[[0]*(max_p+1) for _ in range(max_m+1)] for _ in range(max_n+1)]
    # cost[i][j][k] = optimal rank for i×j×k block

    def compute_cost(si, sj, sk):
        if si <= 0 or sj <= 0 or sk <= 0:
            return 0
        if cost[si][sj][sk] > 0:
            return cost[si][sj][sk]
        
        best = si * sj * sk  # schoolbook
        if si == 2 and sj == 2 and sk == 2:
            best = min(best, 7)
        
        # Try all ways to split
        # Split along i
        for split in range(1, si):
            c = compute_cost(split, sj, sk) + compute_cost(si-split, sj, sk)
            best = min(best, c)
        # Split along j
        for split in range(1, sj):
            c = compute_cost(si, split, sk) + compute_cost(si, sj-split, sk)
            best = min(best, c)
        # Split along k
        for split in range(1, sk):
            c = compute_cost(si, sj, split) + compute_cost(si, sj, sk-split)
            best = min(best, c)
        
        cost[si][sj][sk] = best
        return best

    # Compute all costs
    for si in range(1, max_n+1):
        for sj in range(1, max_m+1):
            for sk in range(1, max_p+1):
                compute_cost(si, sj, sk)

    # Find optimal partition
    def total_rank_for_partition(i_parts, j_parts, k_parts):
        total = 0
        for si in i_parts:
            for sj in j_parts:
                for sk in k_parts:
                    total += cost[si][sj][sk]
        return total

    best_rank = n * m * p
    best_partition = None
    all_i_parts = get_all_partitions(n)
    all_j_parts = get_all_partitions(m)
    all_k_parts = get_all_partitions(p)

    for ip in all_i_parts:
        for jp in all_j_parts:
            for kp in all_k_parts:
                r = total_rank_for_partition(ip, jp, kp)
                if r < best_rank:
                    best_rank = r
                    best_partition = (ip, jp, kp)

    if best_partition is None:
        best_partition = ((n,), (m,), (p,))
        best_rank = n * m * p

    i_parts, j_parts, k_parts = best_partition

    # Build the decomposition
    # We need to recursively build the decomposition for each block
    def build_decomposition(si, sj, sk, i_offset, j_offset, k_offset, col, U, V, W):
        """Recursively build decomposition for a block of size si×sj×sk."""
        if si * sj * sk == 0:
            return col
        
        c = cost[si][sj][sk]
        
        if si == 2 and sj == 2 and sk == 2 and c == 7:
            # Use Strassen
            strassen = strassen_terms()
            for u_pat, v_pat, w_pat in strassen:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = i_offset + li, j_offset + lj
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = j_offset + lj, k_offset + lk
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = k_offset + lk, i_offset + li
                    W[gk * n + gi][col] = val_w
                col += 1
            return col
        elif c == si * sj * sk:
            # Schoolbook
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = i_offset + li, j_offset + lj, k_offset + lk
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
            return col
        else:
            # Need to split - find the optimal split
            # Try all splits and pick the one that achieves the optimal cost
            found = False
            # Split along i
            for split in range(1, si):
                c1 = cost[split][sj][sk]
                c2 = cost[si-split][sj][sk]
                if c1 + c2 == c:
                    col = build_decomposition(split, sj, sk, i_offset, j_offset, k_offset, col, U, V, W)
                    col = build_decomposition(si-split, sj, sk, i_offset+split, j_offset, k_offset, col, U, V, W)
                    found = True
                    break
            if found:
                return col
            # Split along j
            for split in range(1, sj):
                c1 = cost[si][split][sk]
                c2 = cost[si][sj-split][sk]
                if c1 + c2 == c:
                    col = build_decomposition(si, split, sk, i_offset, j_offset, k_offset, col, U, V, W)
                    col = build_decomposition(si, sj-split, sk, i_offset, j_offset+split, k_offset, col, U, V, W)
                    found = True
                    break
            if found:
                return col
            # Split along k
            for split in range(1, sk):
                c1 = cost[si][sj][split]
                c2 = cost[si][sj][sk-split]
                if c1 + c2 == c:
                    col = build_decomposition(si, sj, split, i_offset, j_offset, k_offset, col, U, V, W)
                    col = build_decomposition(si, sj, sk-split, i_offset, j_offset, k_offset+split, col, U, V, W)
                    found = True
                    break
            if found:
                return col
            # Fallback: schoolbook
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = i_offset + li, j_offset + lj, k_offset + lk
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
            return col

    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]

    col = 0
    i_off = 0
    for si in i_parts:
        j_off = 0
        for sj in j_parts:
            k_off = 0
            for sk in k_parts:
                col = build_decomposition(si, sj, sk, i_off, j_off, k_off, col, U, V, W)
            k_off += sk
        j_off += sj
        i_off += si

    result = {"U": U, "V": V, "W": W}
    
    if verify(U, V, W, best_rank):
        return result
    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
