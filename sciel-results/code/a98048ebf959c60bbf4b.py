import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a local search with simulated annealing to find a maximum packing of disjoint 
    2x2x2 Strassen blocks. Each Strassen block reduces the rank by 1 compared to schoolbook.
    Total rank = 105 - num_blocks.
    """
    start_time = time.time()
    deadline = start_time + 115.0

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
                            indices = frozenset()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices = indices | frozenset([(i, j, k)])
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    num_blocks_total = len(all_blocks)
    
    # Precompute overlap matrix as bitmasks for speed
    # Map each (i,j,k) to a bit position
    idx_to_bit = {}
    bit_to_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                idx_to_bit[(i, j, k)] = 1 << idx
                bit_to_idx.append((i, j, k))
    
    # Convert block indices to bitmasks
    block_bitmasks = []
    for _, indices in all_blocks:
        mask = 0
        for idx in indices:
            mask |= idx_to_bit[idx]
        block_bitmasks.append(mask)
    
    # Precompute overlap between each pair of blocks
    # overlap[i][j] = True if blocks i and j share any index
    overlap = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if block_bitmasks[i] & block_bitmasks[j]:
                overlap[i][j] = True
                overlap[j][i] = True

    # Simulated annealing / local search for maximum independent set
    best_num_blocks = 0
    best_packing = []
    
    def greedy_packing(order=None):
        if order is None:
            order = list(range(num_blocks_total))
            random.shuffle(order)
        
        selected = []
        used_mask = 0
        for b in order:
            if not (block_bitmasks[b] & used_mask):
                selected.append(b)
                used_mask |= block_bitmasks[b]
        return selected
    
    # Initial greedy solutions
    for t in range(50):
        if time.time() > deadline - 30.0:
            break
        packing = greedy_packing()
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]
    
    # Local search: try to add blocks by removing others
    # We use a simpler approach: random restarts with hill climbing
    for t in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Start from current best or random
        if t % 5 == 0 or not best_packing:
            packing = greedy_packing()
        else:
            packing = best_packing[:]
        
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Try to add a random unselected block
            selected_set = set(packing)
            unselected = [b for b in range(num_blocks_total) if b not in selected_set]
            
            if not unselected:
                break
            
            random.shuffle(unselected)
            
            for new_block in unselected[:20]:  # Try up to 20 candidates
                # Find conflicts
                conflicts = [b for b in packing if overlap[b][new_block]]
                
                if not conflicts:
                    # No conflict, just add it
                    packing.append(new_block)
                    improved = True
                    if len(packing) > best_num_blocks:
                        best_num_blocks = len(packing)
                        best_packing = packing[:]
                    break
                else:
                    # Try to replace one conflict with the new block
                    # Only worth it if we can add more than we remove (not possible with 1-for-1)
                    # So we only do this if it helps in combination
                    pass
        
        # If we found something better, update
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]
            
        # Early exit if we hit theoretical max
        if best_num_blocks == 13:
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
        i1, i2, j1, j2, k1, k2 = all_blocks[block_idx][0]
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
