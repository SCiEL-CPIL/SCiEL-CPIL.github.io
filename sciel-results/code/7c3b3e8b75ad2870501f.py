"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Strategy:
    # 1. Formulate support selection as a packing problem in H(11, 4) with min Hamming distance 6.
    #    This is equivalent to finding a Maximum Independent Set (MIS) in the graph where
    #    nodes are weight-4 supports and edges connect supports with intersection size >= 3.
    # 2. Use a vectorized greedy construction with randomized tie-breaking to find a large independent set.
    # 3. Use iterative 1-swap local search to optimize the code size.
    # 4. Expand each selected support into all 16 sign patterns.
    # 5. Verify the resulting vector set satisfies the distance constraint.
    
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    # Precompute intersection sizes or incompatibility matrix
    # incompat[i, j] = 1 if |sup_i ∩ sup_j| >= 3
    # Represent supports as bitmasks for fast intersection
    masks = np.zeros(N_sup, dtype=np.int32)
    for i, s in enumerate(supports):
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks[i] = m
    
    # Vectorized incompatibility computation using chunked processing
    incompat = np.zeros((N_sup, N_sup), dtype=np.int8)
    chunk_size = 5000
    for i_start in range(0, N_sup, chunk_size):
        if time.time() > deadline - 20.0:
            break
        i_end = min(i_start + chunk_size, N_sup)
        # For each pair (i, j) with i in [i_start, i_end), j > i
        # Use broadcasting to compute intersections
        for i in range(i_start, i_end):
            # Vectorized: count bits in (masks[i] & masks[j]) for all j
            inter_bits = np.bitwise_and(masks[i], masks)
            # Count set bits using a lookup-free approach: 
            # We can use np.unpackbits or a simple popcount via repeated shifts
            # For 11-bit numbers, we can precompute popcount for all 2048 values
            popcount_table = np.array([bin(x).count('1') for x in range(2048)], dtype=np.int8)
            popcounts = popcount_table[inter_bits.astype(np.int32)]
            # Mark incompatible where popcount >= 3
            incompat[i, popcounts >= 3] = 1
            incompat[popcounts >= 3, i] = 1
    
    # Precompute neighbors as lists for faster access in local search
    neighbors = [np.where(incompat[i] == 1)[0] for i in range(N_sup)]
    
    def greedy_min_degree(available, rng):
        """
        Greedy MIS using minimum degree heuristic.
        """
        selected = []
        avail = set(available)
        avail_arr = np.array(sorted(avail), dtype=np.int32)
        
        while len(avail) > 0:
            if time.time() > deadline - 10.0:
                break
            
            # Find node in avail with minimum degree in the induced subgraph on avail
            min_deg = float('inf')
            candidates = []
            for u in avail:
                # Count neighbors in avail
                deg = 0
                for v in neighbors[u]:
                    if v in avail:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            # Pick random from candidates to break symmetry
            u = int(rng.choice(candidates))
            selected.append(u)
            
            # Remove u and its neighbors from avail
            to_remove = {u}
            for v in neighbors[u]:
                if v in avail:
                    to_remove.add(v)
            avail -= to_remove
            
        return selected

    def local_search_1swap(selected, rng):
        """
        Try to improve the independent set using 1-2 swap.
        Remove one node v, add two nodes u1, u2 that were blocked by v.
        """
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Shuffle to avoid bias
            selected_list = list(selected_set)
            rng.shuffle(selected_list)
            
            for v in selected_list:
                if time.time() > deadline - 10.0:
                    break
                
                # Nodes compatible with selected \ {v}
                # A node x is compatible if it is not in neighbors[y] for any y in selected \ {v}
                # And x is not in selected
                # Vectorized check: for each x, check incompat[x, y] for all y in selected \ {v}
                sel_minus_v = [y for y in selected_set if y != v]
                if len(sel_minus_v) == 0:
                    # All nodes are compatible
                    potential_adds = [x for x in range(N_sup) if x not in selected_set]
                else:
                    sel_minus_v_arr = np.array(sel_minus_v, dtype=np.int32)
                    # incompat[:, sel_minus_v_arr] gives a (N_sup, len(sel_minus_v)) matrix
                    # A node x is incompatible if any entry is 1
                    incompatible_with_sel = np.any(incompat[:, sel_minus_v_arr] == 1, axis=1)
                    # Also exclude nodes already in selected
                    in_selected = np.zeros(N_sup, dtype=bool)
                    in_selected[list(selected_set)] = True
                    compatible_mask = (~incompatible_with_sel) & (~in_selected)
                    potential_adds = np.where(compatible_mask)[0].tolist()
                        
                # Need to find two nodes u1, u2 in potential_adds that are compatible
                if len(potential_adds) >= 2:
                    found = False
                    # Use vectorized check for pairs
                    pa = np.array(potential_adds, dtype=np.int32)
                    # Check incompatibility among potential adds
                    sub_incompat = incompat[np.ix_(pa, pa)]
                    # Find a pair (i, j) with i < j where sub_incompat[i,j] == 0
                    for i in range(len(pa)):
                        if time.time() > deadline - 10.0:
                            break
                        # Find first j > i where sub_incompat[i, j] == 0
                        compat_j = np.where(sub_incompat[i, i+1:] == 0)[0]
                        if len(compat_j) > 0:
                            u1 = pa[i]
                            u2 = pa[i+1+compat_j[0]]
                            # Swap
                            selected_set.remove(v)
                            selected_set.add(int(u1))
                            selected_set.add(int(u2))
                            improved = True
                            found = True
                            break
                if improved:
                    break
                    
        return list(selected_set)

    best_iset = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
            
        avail = set(range(N_sup))
        
        # Randomize the order slightly? Or just use min-degree greedy.
        # The min-degree greedy is deterministic given the tie-breaking.
        # We use rng.choice for tie-breaking.
        
        iset = greedy_min_degree(avail, rng)
        
        # Local search
        iset = local_search_1swap(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = iset.copy()
            
    if best_iset is None:
        # Fallback: return a single vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 2
        return v.reshape(1, -1)
        
    # Generate vectors
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        return np.array([[]], dtype=np.int64)

    result = np.array(vectors, dtype=np.int64)
    
    # Optional: Verify
    # M = result @ result.T
    # np.fill_diagonal(M, 0)
    # assert np.all(M <= 2)
    
    return result

# EVOLVE-BLOCK-END
