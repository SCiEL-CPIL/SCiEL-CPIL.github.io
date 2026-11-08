import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    # A block is defined by two i's, two j's, two k's
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    # Greedy search with multiple strategies
    best_num_blocks = 0
    best_packing = []

    def greedy_pack(blocks_list, used_indices):
        current_packing = []
        for block_params, block_indices in blocks_list:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices = used_indices | block_indices
        return current_packing

    # Strategy 1: Random shuffles
    num_trials = 300
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        packing = greedy_pack(shuffled, frozenset())
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            if best_num_blocks >= 13:
                break

    # Strategy 2: Priority-based ordering by constraint
    i_usage = [0] * n
    j_usage = [0] * m
    k_usage = [0] * p
    for params, indices in all_blocks:
        i1, i2, j1, j2, k1, k2 = params
        i_usage[i1] += 1
        i_usage[i2] += 1
        j_usage[j1] += 1
        j_usage[j2] += 1
        k_usage[k1] += 1
        k_usage[k2] += 1

    def block_score(params):
        i1, i2, j1, j2, k1, k2 = params
        return -(i_usage[i1] + i_usage[i2] + j_usage[j1] + j_usage[j2] + k_usage[k1] + k_usage[k2])

    sorted_blocks = sorted(all_blocks, key=lambda x: block_score(x[0]))
    packing = greedy_pack(sorted_blocks, frozenset())
    if len(packing) > best_num_blocks:
        best_num_blocks = len(packing)
        best_packing = packing

    # Strategy 3: Local improvement - try removing one block and refilling
    for _ in range(3):
        if time.time() > deadline - 15.0:
            break
        for idx in range(len(best_packing)):
            if time.time() > deadline - 5.0:
                break
            remaining = best_packing[:idx] + best_packing[idx+1:]
            used_rem = frozenset()
            for params in remaining:
                i1, i2, j1, j2, k1, k2 = params
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            used_rem = used_rem | frozenset([(i, j, k)])
            
            shuffled = all_blocks[:]
            random.shuffle(shuffled)
            new_packing = remaining[:]
            new_used = used_rem
            for block_params, block_indices in shuffled:
                if not (block_indices & new_used):
                    new_packing.append(block_params)
                    new_used = new_used | block_indices
            
            if len(new_packing) > best_num_blocks:
                best_num_blocks = len(new_packing)
                best_packing = new_packing

    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
