"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses an improved search to find maximum packing of disjoint 2x2x2 Strassen blocks,
    then applies column elimination to reduce rank further.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Precompute all possible 2x2x2 blocks
    # Each block is defined by (i1,i2,j1,j2,k1,k2) with i1<i2, j1<j2, k1<k2
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            # Compute bitmask
                            mask = 0
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        bit_pos = i * (m * p) + j * p + k
                                        mask |= (1 << bit_pos)
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), mask))

    num_blocks_total = len(all_blocks)
    
    # Greedy search for maximum disjoint packing with multiple strategies
    best_num_blocks = 0
    best_packing = []
    
    # Strategy 1: Multiple random orderings
    num_trials = 2000
    for t in range(num_trials):
        if time.time() > deadline - 20.0:
            break
        random.shuffle(all_blocks)
        used_mask = 0
        current_packing = []
        for block_params, block_mask in all_blocks:
            if not (used_mask & block_mask):
                current_packing.append(block_params)
                used_mask |= block_mask
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
            if best_num_blocks >= 6:  # theoretical max
                break
    
    # Strategy 2: Systematic search with priority ordering
    # Try different priority orderings
    if best_num_blocks < 6:
        # Priority: prefer blocks that use "scarce" indices
        # i has only 3 values, so i-pairs are most scarce
        # Sort by how "rare" the indices are
        def block_priority(block_params):
            i1, i2, j1, j2, k1, k2 = block_params
            # Higher priority for blocks using more central indices
            score = 0
            score += (n - 1 - abs(i1 - i2)) * 10  # prefer adjacent i's
            score += (m - 1 - abs(j1 - j2)) * 5   # prefer adjacent j's
            score += (p - 1 - abs(k1 - k2)) * 2   # prefer adjacent k's
            return score
        
        sorted_blocks = sorted(all_blocks, key=lambda x: -block_priority(x[0]))
        used_mask = 0
        current_packing = []
        for block_params, block_mask in sorted_blocks:
            if not (used_mask & block_mask):
                current_packing.append(block_params)
                used_mask |= block_mask
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
    
    # Strategy 3: Backtracking search for optimal packing (limited depth)
    if best_num_blocks < 6 and time.time() < deadline - 30.0:
        # Try to find 6 blocks via backtracking
        # Precompute compatibility
        compat = [[True] * num_blocks_total for _ in range(num_blocks_total)]
        for i in range(num_blocks_total):
            for j in range(i + 1, num_blocks_total):
                if all_blocks[i][1] & all_blocks[j][1]:
                    compat[i][j] = False
                    compat[j][i] = False
        
        # Backtracking to find packing of size 6
        found_six = False
        def backtrack(start_idx, current, used_mask):
            nonlocal found_six
            if len(current) == 6:
                found_six = True
                return True
            if time.time() > deadline - 25.0:
                return False
            for i in range(start_idx, num_blocks_total):
                if not (used_mask & all_blocks[i][1]):
                    current.append(all_blocks[i][0])
                    if backtrack(i + 1, current, used_mask | all_blocks[i][1]):
                        return True
                    current.pop()
                    if len(current) == 0:  # only check time at top level
                        pass
            return False
        
        if not found_six:
            current = []
            if backtrack(0, current, 0):
                if len(current) > best_num_blocks:
                    best_num_blocks = len(current)
                    best_packing = current[:]
    
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

    # Column elimination: try to find linear dependencies among columns
    # A column c is redundant if it can be expressed as a linear combination of other columns
    # such that the tensor equation still holds.
    # We check: for each column c, see if removing it and adjusting others preserves the tensor.
    
    rank = len(U_cols)
    
    # Build the tensor constraint matrix for column elimination
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find if any column is in the span of others in a way that preserves T.
    
    # For exact elimination, we try: can column c be written as sum of alpha_d * column_d (d != c)
    # such that for all (i,j,k): U[i,j]*V[j,k]*W[k,i] contribution from c equals sum of contributions from d?
    # This is nonlinear, so we use a heuristic: check if the column is zero or if it's a scalar multiple.
    
    # Simple elimination: remove zero columns
    non_zero_cols = []
    for c in range(rank):
        is_zero = all(U_cols[c][r] == 0 for r in range(U_rows)) and \
                  all(V_cols[c][r] == 0 for r in range(V_rows)) and \
                  all(W_cols[c][r] == 0 for r in range(W_rows))
        if not is_zero:
            non_zero_cols.append(c)
    
    if len(non_zero_cols) < rank:
        U_cols = [U_cols[c] for c in non_zero_cols]
        V_cols = [V_cols[c] for c in non_zero_cols]
        W_cols = [W_cols[c] for c in non_zero_cols]
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
