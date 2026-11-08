import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses maximum independent set on a conflict graph with bitset optimization to find
    the optimal packing of disjoint 2x2x2 Strassen blocks, then fills remaining terms
    with schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    num_blocks = len(all_blocks)
    
    # Build conflict graph using bitsets for efficiency
    # conflict_bits[i] is a bitmask of blocks that conflict with block i
    conflict_bits = [0] * num_blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    # Find maximum independent set using branch and bound with bitset optimization
    best_set = set()
    best_size = 0
    
    # Precompute degree for ordering
    degrees = [bin(conflict_bits[i]).count('1') for i in range(num_blocks)]
    
    # Greedy heuristic to get initial solution
    def greedy_mis():
        nonlocal best_set, best_size
        remaining = set(range(num_blocks))
        current = []
        while remaining:
            # Pick the block with minimum degree among remaining
            min_deg = min(remaining, key=lambda x: sum(1 for y in remaining if x != y and (conflict_bits[x] >> y) & 1))
            current.append(min_deg)
            # Remove min_deg and its neighbors from remaining
            neighbors = set()
            for y in remaining:
                if y != min_deg and (conflict_bits[min_deg] >> y) & 1:
                    neighbors.add(y)
            remaining -= neighbors
            remaining.discard(min_deg)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        return current
    
    greedy_mis()
    
    # Branch and bound to improve
    def max_independent_set(candidate_mask, current_set, current_size):
        nonlocal best_set, best_size
        nonlocal deadline
        
        if time.time() > deadline - 2.0:
            return
        
        if candidate_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound: current_size + popcount(candidate_mask)
        if current_size + bin(candidate_mask).count('1') <= best_size:
            return
        
        # Get the first candidate (lowest set bit)
        first_bit = (candidate_mask & -candidate_mask).bit_length() - 1
        first = first_bit
        rest_mask = candidate_mask & ~(1 << first)
        
        # Option 1: include first
        new_mask = rest_mask & ~conflict_bits[first]
        new_set = current_set + [first]
        max_independent_set(new_mask, new_set, current_size + 1)
        
        # Option 2: exclude first
        max_independent_set(rest_mask, current_set, current_size)
    
    initial_mask = (1 << num_blocks) - 1
    max_independent_set(initial_mask, [], 0)
    
    # Extract the best packing
    best_packing = [all_blocks[i][0] for i in sorted(best_set)]
    best_num_blocks = len(best_packing)
    
    # Construct U, V, W based on the best packing
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
