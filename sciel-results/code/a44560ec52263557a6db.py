import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    block_masks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            # Check if i2-i1==1, j2-j1==1, k2-k1==1 for 2x2x2
                            # The previous code allowed any pair, but Strassen is for 2x2x2 contiguous?
                            # Actually, Strassen works for any 2x2x2 sub-tensor, but typically we use contiguous indices
                            # to cover the grid efficiently. The previous code used i1, i2 etc.
                            # Let's stick to contiguous 2x2x2 blocks for maximum packing efficiency.
                            if i2 != i1 + 1 or j2 != j1 + 1 or k2 != k1 + 1:
                                continue
                            
                            indices = frozenset(
                                (i, j, k)
                                for i in (i1, i2)
                                for j in (j1, j2)
                                for k in (k1, k2)
                            )
                            # Map indices to a bitmask of size 105
                            mask = 0
                            for (i, j, k) in indices:
                                idx = (i * m + j) * p + k
                                mask |= (1 << idx)
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
                            block_masks.append(mask)

    total_blocks = len(all_blocks)
    
    # Find maximum disjoint packing
    # Use a greedy approach with random restarts and local search
    best_packing = []
    best_size = 0
    
    random.seed(42)
    
    # Precompute overlaps
    # overlap[i][j] is True if block i and j overlap
    # Actually, we can just use bitmasks
    
    for trial in range(5000):
        if time.time() > deadline - 5.0:
            break
            
        order = list(range(total_blocks))
        random.shuffle(order)
        
        used_mask = 0
        packing = []
        
        for idx in order:
            mask = block_masks[idx]
            if not (mask & used_mask):
                packing.append(idx)
                used_mask |= mask
        
        # Local search: try to swap blocks to increase count
        improved = True
        while improved:
            improved = False
            # Try removing one block and adding two others? Or just adding one?
            # Simple hill climbing: try to add a block that fits
            # If none fit, try to remove a block and see if we can add 2
            current_size = len(packing)
            packing_set = set(packing)
            
            # Try to add a block
            added = False
            for idx in range(total_blocks):
                if idx not in packing_set:
                    mask = block_masks[idx]
                    if not (mask & used_mask):
                        packing.append(idx)
                        packing_set.add(idx)
                        used_mask |= mask
                        improved = True
                        added = True
                        break
            
            if added:
                continue
                
            # Try 2-for-1 swap: remove one, add two
            # This is expensive. Let's try simpler: remove one, try to fill the gap with others
            # Or just rely on many random restarts.
            # Let's try a more sophisticated local move:
            # Pick a random block in packing, remove it, try to greedily fill the space with up to 2 new blocks
            if current_size > 0 and random.random() < 0.5:
                rem_idx = random.choice(packing)
                rem_mask = block_masks[rem_idx]
                
                # Tentatively remove
                packing.remove(rem_idx)
                packing_set.remove(rem_idx)
                used_mask ^= rem_mask # Assuming no overlap, so XOR works to remove
                
                # Try to add 2 blocks that fit in the new space
                new_adds = []
                # Sort remaining blocks by size (all same) or random
                candidates = [i for i in range(total_blocks) if i not in packing_set]
                random.shuffle(candidates)
                
                for c1 in candidates:
                    m1 = block_masks[c1]
                    if not (m1 & used_mask):
                        new_adds.append(c1)
                        used_mask |= m1
                        packing_set.add(c1)
                        for c2 in candidates:
                            if c2 == c1: continue
                            m2 = block_masks[c2]
                            if not (m2 & used_mask):
                                new_adds.append(c2)
                                used_mask |= m2
                                packing_set.add(c2)
                                break
                        break
                        
                if len(new_adds) > 1:
                    # We added 2, removed 1. Net gain +1
                    improved = True
                else:
                    # Revert
                    for a in new_adds:
                        packing.remove(a)
                        packing_set.remove(a)
                        used_mask ^= block_masks[a]
                    packing.append(rem_idx)
                    packing_set.add(rem_idx)
                    used_mask |= rem_mask
        
        if len(packing) > best_size:
            best_size = len(packing)
            best_packing = packing
            if best_size >= 13: # Theoretical max is 13
                break

    # Construct U, V, W
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

    covered = set()
    for idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered triples
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
