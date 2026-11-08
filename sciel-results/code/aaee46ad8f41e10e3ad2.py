import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses greedy search with local improvement to find maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 105 - num_blocks.
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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks_total = len(all_blocks)
    
    # Build conflict graph
    conflicts = [set() for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    degrees = [len(conflicts[i]) for i in range(num_blocks_total)]
    
    def greedy_packing(order):
        """Greedy packing given an order of blocks."""
        used = set()
        current = []
        for idx in order:
            if not (all_blocks[idx][1] & used):
                current.append(idx)
                used |= all_blocks[idx][1]
        return current
    
    def local_improve(packing):
        """Try to improve packing by swapping out a block for a better one."""
        if not packing:
            return packing
        used = set()
        for idx in packing:
            used |= all_blocks[idx][1]
        
        improved = True
        best_packing = packing[:]
        while improved:
            improved = False
            for pos in range(len(best_packing)):
                # Try removing block at pos and see if we can add two blocks
                removed = best_packing[pos]
                remaining = best_packing[:pos] + best_packing[pos+1:]
                used_remaining = set()
                for idx in remaining:
                    used_remaining |= all_blocks[idx][1]
                
                # Find blocks that could be added (don't conflict with remaining)
                candidates = []
                for idx in range(num_blocks_total):
                    if idx == removed:
                        continue
                    if idx not in remaining and not (all_blocks[idx][1] & used_remaining):
                        candidates.append(idx)
                
                # Try adding up to 2 candidates
                best_new = None
                best_new_size = len(remaining)
                for c1 in candidates:
                    used_c1 = used_remaining | all_blocks[c1][1]
                    for c2 in candidates:
                        if c2 <= c1:
                            continue
                        if not (all_blocks[c2][1] & used_c1):
                            new_size = len(remaining) + 2
                            if new_size > best_new_size:
                                best_new_size = new_size
                                best_new = remaining + [c1, c2]
                                improved = True
                                break
                    if improved and best_new_size > len(remaining):
                        break
                
                if best_new and best_new_size > len(best_packing):
                    best_packing = best_new
                    improved = True
                    break
        
        return best_packing
    
    # Initial ordering: sort by degree ascending
    base_order = sorted(range(num_blocks_total), key=lambda x: degrees[x])
    
    best_packing = []
    best_size = 0
    
    # Phase 1: Greedy with random restarts
    num_trials = 2000
    trial = 0
    while trial < num_trials:
        if time.time() > deadline - 10.0:
            break
        
        # Create a perturbed order
        order = base_order[:]
        # Randomly swap some elements
        for _ in range(random.randint(0, 5)):
            i = random.randint(0, len(order) - 1)
            j = random.randint(0, len(order) - 1)
            order[i], order[j] = order[j], order[i]
        
        packing = greedy_packing(order)
        
        if len(packing) > best_size:
            best_size = len(packing)
            best_packing = packing[:]
            
            # Local improvement
            if time.time() < deadline - 5.0:
                improved = local_improve(best_packing)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_packing = improved
            
            if best_size >= 8:
                break
        
        trial += 1
    
    # Phase 2: More targeted search if time remains
    if time.time() < deadline - 5.0 and best_size < 8:
        # Try all pairs of blocks and see if they can be extended
        for i in range(num_blocks_total):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, num_blocks_total):
                if all_blocks[i][1] & all_blocks[j][1]:
                    continue
                # Start with {i, j} and greedily extend
                used = all_blocks[i][1] | all_blocks[j][1]
                current = [i, j]
                for k in base_order:
                    if k == i or k == j:
                        continue
                    if not (all_blocks[k][1] & used):
                        current.append(k)
                        used |= all_blocks[k][1]
                
                if len(current) > best_size:
                    best_size = len(current)
                    best_packing = current[:]
                    if best_size >= 8:
                        break
            if best_size >= 8:
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
    for idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx][0]
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
