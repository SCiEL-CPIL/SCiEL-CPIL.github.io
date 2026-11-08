import time
import random
import itertools
import sys
from functools import lru_cache

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a Branch and Bound heuristic to solve the Integer Linear Programming problem of
    maximizing the number of disjoint 2x2x2 Strassen blocks.
    
    Strategy:
    1. Enumerate all possible 2x2x2 Strassen blocks.
    2. Build a conflict graph where nodes are blocks and edges indicate overlap.
    3. Use Branch and Bound to find the Maximum Independent Set (MIS) of this graph.
       This corresponds to the optimal selection of non-overlapping Strassen blocks.
    4. Construct the U, V, W matrices using the selected Strassen blocks and 
       schoolbook terms for the remaining uncovered entries.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # 1. Generate all possible 2x2x2 Strassen blocks
    # A block is defined by pairs of indices (i1, i2), (j1, j2), (k1, k2)
    blocks = []
    block_indices = []
    
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
                            blocks.append((i1, i2, j1, j2, k1, k2))
                            block_indices.append(frozenset(indices))

    num_blocks = len(blocks)
    
    # 2. Build conflict graph
    # conflict[i] is a set of indices j such that block i and block j overlap
    conflict = [set() for _ in range(num_blocks)]
    
    # Optimization: Two blocks overlap if they share any (i, j, k) tuple.
    # Since blocks are 2x2x2, they overlap if their index ranges intersect in all dimensions?
    # No, they overlap if the set of 8 tuples is not disjoint.
    # Efficient check:
    # Block A: (i1a, i2a, j1a, j2a, k1a, k2a)
    # Block B: (i1b, i2b, j1b, j2b, k1b, k2b)
    # They are disjoint if:
    # (i1a, i2a) is disjoint from (i1b, i2b) OR
    # (j1a, j2a) is disjoint from (j1b, j2b) OR
    # (k1a, k2a) is disjoint from (k1b, k2b)
    # Since indices are integers, two sets {x, y} and {u, v} are disjoint if max(x,y) < min(u,v) or max(u,v) < min(x,y).
    
    def ranges_disjoint(r1, r2):
        # r1 = (a, b) with a < b, r2 = (c, d) with c < d
        # Disjoint if b < c or d < a
        return r1[1] < r2[0] or r2[1] < r1[0]

    for i in range(num_blocks):
        i1, i2, j1, j2, k1, k2 = blocks[i]
        ri = (i1, i2)
        rj = (j1, j2)
        rk = (k1, k2)
        
        for j in range(i + 1, num_blocks):
            if time.time() > deadline - 5.0:
                break
            i1b, i2b, j1b, j2b, k1b, k2b = blocks[j]
            rj_b = (j1b, j2b)
            rk_b = (k1b, k2b)
            ri_b = (i1b, i2b)
            
            if ranges_disjoint(ri, ri_b) or ranges_disjoint(rj, rj_b) or ranges_disjoint(rk, rk_b):
                continue # Disjoint, no conflict
            
            # They conflict
            conflict[i].add(j)
            conflict[j].add(i)

    # 3. Branch and Bound for Maximum Independent Set
    # We want to select a subset of blocks such that no two are connected in the conflict graph.
    # Maximize the size of the subset.
    
    best_size = 0
    best_mask = 0 # Bitmask of selected blocks
    
    # Order blocks by degree (number of conflicts) to prune better?
    # Or just use the natural order.
    # For B&B, ordering by degree descending often helps.
    # Let's create an ordered list of block indices.
    order = list(range(num_blocks))
    order.sort(key=lambda x: len(conflict[x]), reverse=True)
    
    # Map original index to position in order? 
    # Actually, it's easier to just recurse on the sorted list.
    # But we need to check conflicts with previously selected blocks.
    
    selected = [False] * num_blocks
    current_size = 0

    # To speed up B&B, we can use a greedy lower bound to initialize best_size.
    # Greedy: Pick block with min degree, remove it and neighbors. Repeat.
    def greedy_mis():
        remaining = set(range(num_blocks))
        count = 0
        while remaining:
            # Pick node with min degree in remaining subgraph
            min_deg = float('inf')
            min_node = -1
            for node in remaining:
                deg = len(conflict[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
            count += 1
            remaining.discard(min_node)
            remaining -= conflict[min_node]
        return count

    best_size = greedy_mis()
    
    # Reconstruct best_mask from a greedy run? 
    # For now, just use the size for pruning. We need the actual mask to build the matrices.
    # Let's store the best mask.
    def greedy_mis_mask():
        remaining = set(range(num_blocks))
        mask = 0
        while remaining:
            min_deg = float('inf')
            min_node = -1
            for node in remaining:
                deg = len(conflict[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
            mask |= (1 << min_node)
            remaining.discard(min_node)
            remaining -= conflict[min_node]
        return mask

    best_mask = greedy_mis_mask()

    def branch_and_bound(k, current_mask, current_size, remaining_candidates):
        nonlocal best_size, best_mask
        
        if time.time() > deadline - 5.0:
            return

        # Pruning: Even if we take all remaining, can we beat best?
        if current_size + len(remaining_candidates) <= best_size:
            return
            
        if k == num_blocks:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        # Option 1: Exclude block k
        # The remaining candidates are those > k that are not in conflict with any *selected* block?
        # No, standard B&B for MIS:
        # We decide on node k.
        # If we include k, we must exclude all neighbors of k.
        # If we exclude k, we move to k+1.
        
        # However, the `remaining_candidates` set should only contain nodes > k that are not yet decided 
        # and are not in conflict with previously *included* nodes.
        
        # Let's refine the state.
        # State: k (current index in order), current_mask, current_size.
        # We need to know which nodes are "available".
        # A node j > k is available if it is not in conflict with any node i < k that was INCLUDED.
        
        # To avoid passing a large set, we can compute availability on the fly or pass a bitmask.
        # With 630 nodes, a Python int bitmask is fine.
        
        # Let's restart the B&B logic with a cleaner bitmask approach.
        # This is getting complex to patch into the existing structure without rewriting.
        # Given the constraints, a simpler recursive B&B on the sorted order with a "available" mask is best.
        
        pass # Placeholder, will implement below

    # Let's implement the B&B properly using bitmasks for speed and simplicity.
    # available_mask: bits set for nodes that can still be chosen.
    # We iterate through nodes in order.
    
    # Initial available mask: all nodes.
    # But we must respect the order.
    # Actually, standard B&B:
    # def bb(k, mask, size, avail):
    #   if k == N: update best
    #   if size + popcount(avail) <= best: return
    #   # Branch 1: Include k
    #   if avail & (1 << k):
    #      new_avail = avail & ~(1 << k) & ~conflict_mask[k]
    #      bb(k+1, mask | (1 << k), size + 1, new_avail)
    #   # Branch 2: Exclude k
    #   new_avail = avail & ~(1 << k)
    #   bb(k+1, mask, size, new_avail)
    
    # Precompute conflict masks
    conflict_masks = [0] * num_blocks
    for i in range(num_blocks):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask

    # Order nodes by degree descending for better pruning?
    # Or just use 0..N-1.
    # Let's use the sorted order for better pruning.
    # But the bitmask indices must match the order?
    # If we reorder, we must map the conflict masks accordingly.
    # It's easier to stick to 0..N-1 order and rely on the pruning `size + popcount(avail)`.
    
    # Let's try 0..N-1 first. If too slow, we can sort.
    # With 630 nodes, recursion depth 630.
    
    import sys
    sys.setrecursionlimit(2000)

    def popcount(x):
        return bin(x).count('1')

    def bb(k, current_mask, current_size, avail_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline - 2.0:
            return

        # Pruning
        if current_size + popcount(avail_mask) <= best_size:
            return

        if k == num_blocks:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        # If node k is not available, we must exclude it
        if not (avail_mask & (1 << k)):
            bb(k + 1, current_mask, current_size, avail_mask)
            return

        # Branch 1: Include k
        # New avail: remove k and all its neighbors
        new_avail_incl = avail_mask & ~(1 << k) & ~conflict_masks[k]
        bb(k + 1, current_mask | (1 << k), current_size + 1, new_avail_incl)
        
        # Branch 2: Exclude k
        new_avail_excl = avail_mask & ~(1 << k)
        bb(k + 1, current_mask, current_size, new_avail_excl)

    # Initial call
    # avail_mask has all bits set
    initial_avail = (1 << num_blocks) - 1
    bb(0, 0, 0, initial_avail)

    # 4. Construct Matrices
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
    
    # Add selected Strassen blocks
    for i in range(num_blocks):
        if best_mask & (1 << i):
            params = blocks[i]
            add_strassen_block(*params)
            # Mark cells as covered
            i1, i2, j1, j2, k1, k2 = params
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        covered.add((i, j, k))

    # Add schoolbook terms for uncovered cells
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
