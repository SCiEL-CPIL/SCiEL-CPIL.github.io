import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses an aggressive randomized local search with bitmask conflict checking to find
    a maximum packing of disjoint 2x2x2 Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    # A block is defined by (i1, i2, j1, j2, k1, k2) with i1<i2, j1<j2, k1<k2
    all_blocks_params = []
    all_blocks_indices = []
    
    # Map (i,j,k) to a unique integer 0..104
    idx_map = {}
    counter = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_map[(i, j, k)] = counter
                counter += 1
    
    total_indices = counter # 105

    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.append(idx_map[(i, j, k)])
                            all_blocks_params.append((i1, i2, j1, j2, k1, k2))
                            all_blocks_indices.append(indices)
    
    num_blocks = len(all_blocks_params)
    
    # Precompute conflict bitmasks
    # For each block, we need to know which other blocks it conflicts with.
    # Since 630 blocks, a 630x630 boolean matrix is fine, or bitmasks.
    # Let's use a list of sets of conflicting block indices for faster iteration in local search.
    conflict_sets = [set() for _ in range(num_blocks)]
    
    # Optimization: Check overlap using bitmasks of indices
    block_masks = [0] * num_blocks
    for b_idx, indices in enumerate(all_blocks_indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if block_masks[i] & block_masks[j]:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)

    best_num_blocks = 0
    best_packing = []

    def greedy_init():
        available = list(range(num_blocks))
        random.shuffle(available)
        packing = []
        used_indices_mask = 0
        for b in available:
            if not (block_masks[b] & used_indices_mask):
                packing.append(b)
                used_indices_mask |= block_masks[b]
        return packing

    def local_search(packing):
        packing_set = set(packing)
        used_indices_mask = 0
        for b in packing:
            used_indices_mask |= block_masks[b]
        
        improved = True
        while improved:
            improved = False
            # Try to add a block
            # Find all blocks not in packing that don't conflict with current packing
            # This is O(num_blocks * len(packing)) which is small
            for b in range(num_blocks):
                if b in packing_set:
                    continue
                if not (block_masks[b] & used_indices_mask):
                    # We can add b
                    packing.append(b)
                    packing_set.add(b)
                    used_indices_mask |= block_masks[b]
                    improved = True
            
            # Try 2-for-1 swap: Remove 2 blocks, add 1? No, that reduces count.
            # We want to increase count.
            # Try 1-for-1 swap that allows adding more later?
            # Actually, just adding is enough if we restart.
            # But sometimes a swap is needed to reach a higher count.
            # Example: Block A and B conflict. Block C and D conflict.
            # If we have {A, C}, we can't add B or D.
            # If we swap A with B, we still have {B, C}. No gain.
            # However, if we have {A, B} (wait, they conflict so we can't have both).
            
            # A better move: Try to remove one block and add two?
            # If we remove block R, we free up its 8 indices.
            # Can we fit two new blocks X, Y in that space?
            # This is complex. Let's stick to simple additions and random restarts.
            # The greedy + random restart is often sufficient for this size.
            
            # Try a "kick": Randomly remove a few blocks to escape local optimum, then re-greedy
            if not improved and len(packing) > 5:
                # Remove 2 random blocks
                to_remove = random.sample(packing, 2)
                for b in to_remove:
                    packing_set.remove(b)
                    used_indices_mask &= ~block_masks[b]
                packing = [b for b in packing if b not in to_remove]
                
                # Re-greedy fill
                available = [b for b in range(num_blocks) if b not in packing_set]
                random.shuffle(available)
                for b in available:
                    if not (block_masks[b] & used_indices_mask):
                        packing.append(b)
                        packing_set.add(b)
                        used_indices_mask |= block_masks[b]
                improved = True
        return packing

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        packing = greedy_init()
        packing = local_search(packing)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]
            
        # Early exit if we hit theoretical max (13)
        if best_num_blocks >= 13:
            break

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
    for block_idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks_params[block_idx]
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
