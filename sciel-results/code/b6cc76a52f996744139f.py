import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    # Generate all weight-4 vectors
    # Supports are combinations of 4 indices from 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    vectors = []
    vec_to_idx = {}
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            v_tuple = tuple(v)
            vectors.append(v)
            vec_to_idx[v_tuple] = len(vectors)
            
    N = len(vectors)
    
    # Build exact conflict graph
    # Conflict if dist^2 < 4
    # As derived:
    # 1. Identical vectors (dist^2 = 0)
    # 2. Support intersection 3 AND same signs on intersection (dist^2 = 2)
    #    (Intersection 2 or less always has dist^2 >= 4)
    
    # We can build conflict masks
    # conflicts[i] is a set of indices j such that i and j conflict
    conflicts = [set() for _ in range(N)]
    
    # To speed up, group by support
    # For each support, there are 16 vectors.
    # Any two distinct vectors with the SAME support do NOT conflict?
    # Wait. Same support (intersection 4).
    # If same support, they differ in signs.
    # Dist^2 = 4 * (number of differing signs).
    # Min dist^2 is 4 (if they differ in 1 sign).
    # So same support vectors are ALWAYS valid (unless identical).
    # So conflicts within same support are only self-loops (identical).
    
    # Conflicts exist ONLY between different supports.
    # Specifically, if |S_u \cap S_v| = 3.
    # If |S_u \cap S_v| < 3, no conflict.
    
    # Let's iterate over pairs of supports with intersection 3.
    # For each such pair (sup_i, sup_j), we have 16 vectors in group i and 16 in group j.
    # We need to check which pairs conflict.
    # Conflict condition: signs on the 3 shared positions are identical.
    
    # Map support index to list of vector indices
    sup_to_vecs = {s: [] for s in supports}
    for idx, v in enumerate(vectors):
        # Reconstruct support? Or store it.
        pass
        
    # Better: Store support for each vector
    vec_supports = []
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            vec_supports.append(sup)
            
    # Group vector indices by support
    sup_indices = {}
    for idx, sup in enumerate(vec_supports):
        if sup not in sup_indices:
            sup_indices[sup] = []
        sup_indices[sup].append(idx)
        
    # Precompute support intersections
    # We only care about pairs of supports with intersection 3
    # There are 11 choose 4 = 330 supports.
    # Pairs with intersection 3:
    # Fix 3 elements. The 4th is different.
    # For a fixed support S, how many supports T intersect S in 3?
    # T = (S \ {x}) U {y} where y not in S.
    # 4 choices for x, 7 choices for y (11-4=7).
    # So 28 neighbors per support.
    
    # Iterate over all supports
    for i, sup_i in enumerate(supports):
        idxs_i = sup_indices[sup_i]
        # Find supports j with intersection 3
        # We can just check all j > i to avoid duplicates, or use the structure
        # For efficiency, let's just iterate j in range(i+1, n_sup)
        for j in range(i + 1, n_sup):
            sup_j = supports[j]
            inter = set(sup_i).intersection(sup_j)
            if len(inter) == 3:
                idxs_j = sup_indices[sup_j]
                # For each v in idxs_i and w in idxs_j, check conflict
                # Conflict if signs on 'inter' are same.
                for vi in idxs_i:
                    vi_vec = vectors[vi]
                    for wj in idxs_j:
                        wj_vec = vectors[wj]
                        # Check signs on inter
                        conflict = True
                        for pos in inter:
                            if vi_vec[pos] != wj_vec[pos]:
                                conflict = False
                                break
                        if conflict:
                            conflicts[vi].add(wj)
                            conflicts[wj].add(vi)
                            
    # Add self conflicts? No, we don't select same vector twice.
    # But for the mask, we can ignore self.
    
    # Convert conflicts to bitmasks for speed
    # N is 5280. Python int can handle this.
    conflict_masks = [0] * N
    for i in range(N):
        mask = 0
        for j in conflicts[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask
        
    # Greedy Initialization
    def greedy_is():
        remaining = set(range(N))
        current_set = set()
        # Sort by degree (number of conflicts)
        # Degree = len(conflicts[i])
        # We want to pick low degree vertices first? Or high?
        # Standard MIS greedy: pick vertex with min degree in remaining graph.
        # This is O(N^2) if done naively.
        # With N=5280, N^2 is ~27M. Might be slow in Python if not optimized.
        # But we can approximate.
        
        # Let's just do random greedy or static degree greedy.
        # Static degree: sort once.
        order = sorted(range(N), key=lambda x: len(conflicts[x]))
        
        for v in order:
            if v in remaining:
                current_set.add(v)
                # Remove v and its neighbors from remaining
                to_remove = {v}
                for nb in conflicts[v]:
                    if nb in remaining:
                        to_remove.add(nb)
                remaining.difference_update(to_remove)
                if not remaining:
                    break
        return current_set

    best_set = greedy_is()
    best_size = len(best_set)
    
    # Simulated Annealing / Local Search
    # State: set of indices
    # Move: Remove one random vertex, then Greedy Fill
    # Greedy Fill: Add any vertex that doesn't conflict with current set.
    
    S = best_set.copy()
    S_mask = 0
    for v in S:
        S_mask |= (1 << v)
        
    T = 10.0
    cooling = 0.999
    steps = 20000
    
    # Precompute list of all vertices for random selection
    all_vertices = list(range(N))
    
    for step in range(steps):
        if not S:
            break
            
        # Pick random v in S to remove
        v = random.choice(list(S))
        
        old_size = len(S)
        
        # Remove v
        S.remove(v)
        S_mask &= ~(1 << v)
        
        # Greedy Fill
        # Try to add vertices not in S that don't conflict with S
        # To be fast, we can iterate through a random subset or all vertices?
        # N=5280. Checking all is 5280 ops.
        # 20000 steps * 5280 = 100M ops. Might be too slow in Python (120s limit).
        # We need to optimize the fill.
        
        # Optimization: Only consider vertices that were previously removed or not in S?
        # Actually, just iterating all is risky for time.
        # Let's try a "random restart" greedy fill on a subset?
        # Or limit the fill attempts?
        
        # Alternative: Use a queue of "candidates" that became available.
        # When we remove v, its neighbors become "available" (if they weren't in S).
        # But removing v might make many vertices available.
        
        # Let's just iterate through all vertices not in S.
        # To speed up, we can break early if we find enough? No, we want max.
        
        # Let's try to optimize the inner loop.
        added = 0
        # Shuffle all_vertices to avoid bias? Or just iterate.
        # Iterating all 5280 is ~0.5ms in C, but in Python loop it's ~1-5ms.
        # 20000 * 2ms = 40s. This is tight but possible.
        
        # Let's reduce steps to 10000 and make it faster.
        
        for u in all_vertices:
            if u in S:
                continue
            # Check if u conflicts with S
            # if conflict_masks[u] & S_mask == 0:
            if (conflict_masks[u] & S_mask) == 0:
                S.add(u)
                S_mask |= (1 << u)
                added += 1
                
        new_size = len(S)
        
        if new_size > best_size:
            best_size = new_size
            best_set = S.copy()
        else:
            # Accept or reject
            delta = new_size - old_size # delta <= 0 usually
            if delta < 0:
                p = np.exp(delta / T)
                if random.random() > p:
                    # Revert
                    # We need to remove the added vertices
                    # We didn't track which were added specifically, just that S changed.
                    # We can reconstruct S from best_set? No, we want to explore.
                    # Easier: keep a copy of S before the move?
                    # That's expensive.
                    # Alternative: If rejected, revert to the state before removing v?
                    # But we added new ones.
                    # Let's just revert to the state *before* the move (which was the state at start of loop).
                    # We need to store S_prev.
                    pass
            # Actually, simple approach:
            # If we don't accept, we should revert.
            # To make reverting easy, we can store the list of added vertices.
            
        T *= cooling
        
    # Reconstruct vectors from best_set
    vectors_out = []
    for idx in best_set:
        vectors_out.append(vectors[idx])
        
    return np.array(vectors_out, dtype=np.int64)
