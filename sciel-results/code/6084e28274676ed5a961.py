import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    total = n * m * p

    # Generate all blocks
    blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            triples = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = (i * m + j) * p + k
                                        mask |= (1 << idx)
                                        triples.append((i, j, k))
                            blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(triples), mask))

    B = len(blocks)
    
    # Precompute overlap
    # ov[a] is a bitmask of blocks that overlap with a
    ov = [0] * B
    for a in range(B):
        ma = blocks[a][2]
        bits = 0
        for b in range(B):
            if a != b and (ma & blocks[b][2]):
                bits |= (1 << b)
        ov[a] = bits

    def get_conflicts(packing):
        # Returns set of blocks in packing that conflict with each other? 
        # No, packing is always valid (disjoint).
        # This function is not needed if we maintain validity.
        pass

    def is_valid(packing):
        used = 0
        for b in packing:
            if used & blocks[b][2]:
                return False
            used |= blocks[b][2]
        return True

    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_packing = []
    best_len = 0
    current_packing = []
    
    # Initialize with greedy
    rng = random.Random(42)
    order = list(range(B))
    rng.shuffle(order)
    used = 0
    for b in order:
        if (used & blocks[b][2]) == 0:
            current_packing.append(b)
            used |= blocks[b][2]
    best_packing = list(current_packing)
    best_len = len(current_packing)
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if temp < 0.01:
            break
            
        # Generate a move
        move_type = rng.random()
        
        new_packing = list(current_packing)
        
        if move_type < 0.3:
            # Add a random block
            b = rng.randint(0, B - 1)
            if b not in new_packing:
                used = 0
                for x in new_packing:
                    used |= blocks[x][2]
                if (used & blocks[b][2]) == 0:
                    new_packing.append(b)
                else:
                    # Reject if invalid
                    continue
        elif move_type < 0.6:
            # Remove a random block
            if new_packing:
                idx = rng.randint(0, len(new_packing) - 1)
                new_packing.pop(idx)
        else:
            # Swap: remove one, add one
            if new_packing:
                idx = rng.randint(0, len(new_packing) - 1)
                removed = new_packing.pop(idx)
                # Try to find a replacement
                used = 0
                for x in new_packing:
                    used |= blocks[x][2]
                # Pick a random candidate that fits
                candidates = []
                for b in range(B):
                    if b != removed and (used & blocks[b][2]) == 0:
                        candidates.append(b)
                if candidates:
                    new_b = rng.choice(candidates)
                    new_packing.append(new_b)
                else:
                    # Revert
                    new_packing.append(removed)
                    continue
        
        # Evaluate
        # Check validity of new_packing
        valid = True
        used = 0
        for x in new_packing:
            if used & blocks[x][2]:
                valid = False
                break
            used |= blocks[x][2]
        
        if not valid:
            continue
            
        delta = len(new_packing) - len(current_packing)
        
        if delta > 0 or (delta == 0 and rng.random() < 0.5) or (delta < 0 and rng.random() < 1.0 / (1.0 + temp)):
            current_packing = new_packing
            if len(current_packing) > best_len:
                best_len = len(current_packing)
                best_packing = list(current_packing)
                
        temp *= cooling

    # Build matrices
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

        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for bidx in best_packing:
        i1, i2, j1, j2, k1, k2 = blocks[bidx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for t in blocks[bidx][1]:
            covered.add(t)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

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
