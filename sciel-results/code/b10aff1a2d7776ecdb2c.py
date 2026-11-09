import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Strategy: Find a large 2-packing (words at distance >= 3 from each other).
    # A 2-packing corresponds to a set of disjoint balls of radius 1.
    # The size of the maximum 2-packing is alpha(G) where G is the graph of distance 2.
    # We use a greedy algorithm to find a large independent set in G.
    # Then we cover the remaining space (words not in the packing's balls) with additional codewords.
    # This is different from the seed's "Greedy Set Cover" which optimizes coverage directly.
    # Here we optimize packing (disjointness) first, which often yields better structural properties.
    
    t_start = time.time()
    deadline = 118.0

    N = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx

    # Precompute balls (radius 1)
    balls = []
    for i in range(N):
        w = list(idx_to_tuple(i))
        ball = [i]
        for pos in range(n):
            old_val = w[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w)))
            w[pos] = old_val
        balls.append(np.array(ball, dtype=np.int32))
        
    balls_np = np.array(balls, dtype=np.int32)
    
    # Precompute "distance 2 neighbors" for the packing graph
    # Two words are adjacent in the packing graph if their distance is 2.
    # We want an independent set in this graph, i.e., a set of words with pairwise distance >= 3.
    
    # To speed up greedy independent set, we can precompute the set of "conflicts" for each word.
    # Conflict[i] = set of words at distance 2 from i.
    # Note: Words at distance 1 are NOT conflicts for the packing (distance must be >= 3).
    # Words at distance 0 are the word itself.
    # Words at distance >= 3 are compatible.
    
    # Let's build a bitset or boolean mask for conflicts.
    # N=2187, so a boolean array of size N is fine.
    # conflicts[i] is a boolean array where conflicts[i][j] = 1 if dist(i,j) == 2.
    
    # Efficient construction:
    # For each word i, generate all words at distance 2.
    # There are C(n,2) * (q-1)^2 such words.
    # For n=7, q=3: C(7,2)=21, (3-1)^2=4. Total 84 neighbors at distance 2.
    
    conflict_masks = np.zeros((N, N), dtype=bool)
    for i in range(N):
        w = list(idx_to_tuple(i))
        for p1 in range(n):
            for p2 in range(p1+1, n):
                # Change positions p1 and p2
                v1 = w[p1]
                v2 = w[p2]
                for new_v1 in range(q):
                    if new_v1 == v1: continue
                    for new_v2 in range(q):
                        if new_v2 == v2: continue
                        w[p1] = new_v1
                        w[p2] = new_v2
                        j = tuple_to_idx(tuple(w))
                        conflict_masks[i, j] = True
                w[p1] = v1
                w[p2] = v2

    # Greedy Independent Set (2-Packing)
    # We want to maximize the number of vertices selected.
    # Standard greedy: pick vertex with minimum degree in remaining graph? Or maximum?
    # For independent set, picking a vertex with low degree is often good because it conflicts with fewer future candidates.
    # However, the "degree" here is fixed in the original graph, but changes as we remove neighbors.
    # Let's use a simple heuristic: 
    # 1. Order vertices by some criterion (e.g., random, or degree).
    # 2. Iterate through order, add to packing if no conflict with existing packing.
    
    # A better heuristic for Max Independent Set:
    # While vertices remain:
    #   Pick vertex v with minimum number of neighbors in the remaining graph.
    #   Add v to independent set.
    #   Remove v and all its neighbors from the graph.
    
    # This is O(N^2) or O(N * avg_degree). N=2187. N^2 ~ 4.7M. Very fast.
    
    def greedy_packing(seed=None):
        if seed is not None:
            random.seed(seed)
        
        remaining = np.ones(N, dtype=bool)
        packing = []
        
        # To break ties or add randomness, we can shuffle the initial order or use random tie-breaking in min-degree selection.
        # For simplicity and speed, let's use a static order but randomize the starting point or tie-breaking.
        # Actually, min-degree greedy is deterministic if tie-breaking is fixed.
        # Let's try multiple random permutations of vertices and pick the first valid one?
        # Or just randomize the tie-breaking in the min-degree selection.
        
        # Optimization: Instead of recalculating degrees every time (O(N^2) per step -> O(N^3) total),
        # we can just pick a random vertex from the remaining set, or use a heuristic.
        # Given N is small, O(N^3) might be too slow if we do many restarts.
        # Let's stick to a fast heuristic: Randomized Greedy.
        # Pick a random vertex from remaining. Add to packing. Remove it and its conflicts.
        
        # Even better: "Randomized Local Search" on the packing?
        # Let's try a few different strategies for the packing:
        
        # Strategy A: Random Greedy
        # Pick random vertex, add, remove conflicts.
        
        # Strategy B: Degree-based Greedy (approximate)
        # Pick vertex with fewest conflicts in remaining set.
        
        # Let's implement Strategy B efficiently.
        # We maintain a count of how many remaining neighbors each vertex has.
        # But updating counts is expensive.
        
        # Given the small size, let's just do:
        # 1. Randomly shuffle vertices.
        # 2. Iterate, if vertex is in remaining and not in packing, check conflicts with packing.
        #    This is O(N * |Packing| * conflict_check).
        #    Conflict check: is there any v in packing such that conflict_masks[i, v]?
        #    We can precompute for each i, the list of conflicts.
        
        # Let's use a simpler approach that is known to work well for small graphs:
        # Randomized Greedy Set Packing.
        
        remaining = np.ones(N, dtype=bool)
        packing = []
        
        # To improve quality, we can try to maximize the "weight" or just size.
        # Let's try multiple random restarts.
        
        for _ in range(20): # 20 restarts
            if time.time() - t_start > deadline - 10: break
            
            remaining = np.ones(N, dtype=bool)
            packing = []
            
            # Random order
            order = np.random.permutation(N)
            
            for i in order:
                if not remaining[i]:
                    continue
                
                # Check if i conflicts with any in packing
                # We need to know if any v in packing has conflict_masks[i, v] == True.
                # Since packing is small (~100-200), this is fast.
                conflict_found = False
                for v in packing:
                    if conflict_masks[i, v]:
                        conflict_found = True
                        break
                
                if not conflict_found:
                    packing.append(i)
                    # Remove i and all its conflicts from remaining
                    remaining[i] = False
                    remaining[conflict_masks[i, :]] = False
            
            if len(packing) > len(packing): # Logic error, need to store best
                pass
        
        # The above loop structure is flawed for keeping the best. Let's restructure.
        
        best_packing = []
        
        for restart in range(30):
            if time.time() - t_start > deadline - 10: break
            
            remaining = np.ones(N, dtype=bool)
            packing = []
            order = np.random.permutation(N)
            
            for i in order:
                if not remaining[i]:
                    continue
                
                # Check conflict with current packing
                # Optimization: Precompute conflict lists?
                # conflict_lists[i] = list of j such that conflict_masks[i, j]
                
                conflict_found = False
                # Use numpy for speed? 
                # conflicts_with_packing = conflict_masks[i, packing]
                # if np.any(conflicts_with_packing): ...
                
                if packing:
                    conflicts_with_packing = conflict_masks[i, packing]
                    if np.any(conflicts_with_packing):
                        continue
                
                packing.append(i)
                remaining[i] = False
                remaining[conflict_masks[i, :]] = False
            
            if len(packing) > len(best_packing):
                best_packing = packing[:]
        
        return best_packing

    # Run greedy packing
    packing = greedy_packing(seed=42)
    
    # Now, convert packing to a covering code.
    # The packing gives us disjoint balls.
    # Let's add these packing words to the code.
    code = packing[:]
    
    # Calculate covered words
    covered = np.zeros(N, dtype=bool)
    for idx in code:
        covered[balls_np[idx]] = True
    
    # Now, cover the remaining uncovered words.
    # We need to select additional codewords to cover ~covered.
    # Use a greedy set cover on the residual.
    
    # Precompute gains for residual?
    # We can just use the standard greedy: pick word that covers most uncovered.
    
    # To speed this up, we can use the incidence matrix M.
    # But we can also just iterate.
    
    # Let's use a vectorized greedy for the residual cover.
    # We need M (incidence matrix).
    row_indices = []
    col_indices = []
    data = []
    for i in range(N):
        for j in balls_np[i]:
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
    
    import scipy.sparse as sp
    M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    
    uncovered_mask = 1.0 - covered.astype(float)
    
    while not covered.all():
        if time.time() - t_start > deadline - 5: break
        
        gains = M.dot(uncovered_mask)
        max_gain = np.max(gains)
        if max_gain == 0: break
        
        candidates = np.where(gains == max_gain)[0]
        # Random tie break
        idx = candidates[random.randint(0, len(candidates)-1)]
        
        code.append(idx)
        covered[balls_np[idx]] = True
        uncovered_mask = 1.0 - covered.astype(float)
        
    # Local Search: Try to remove redundant codewords.
    # This is similar to the seed, but starting from a structurally different base.
    
    def get_counts(code_list):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code_list:
            counts[balls_np[idx]] += 1
        return counts
    
    counts = get_counts(code)
    
    improved = True
    while improved and time.time() - t_start < deadline - 2:
        improved = False
        
        # 1. Remove Redundant
        for idx in list(code):
            if time.time() - t_start > deadline - 2: break
            ball = balls_np[idx]
            if np.min(counts[ball]) > 1:
                counts[ball] -= 1
                code.remove(idx)
                improved = True
                
        # 2. Swap Moves (limited)
        if time.time() - t_start < deadline - 5:
            num_swap_attempts = 200
            for _ in range(num_swap_attempts):
                if time.time() - t_start > deadline - 2: break
                if len(code) == 0: break
                
                u = random.choice(code)
                v = random.randint(0, N - 1)
                if v in code: continue
                
                ball_u = balls_np[u]
                ball_v = balls_np[v]
                
                # Check validity
                # Words in ball_u \ ball_v must have count > 1
                # We can compute this using masks
                
                # For speed, let's just check min count in ball_u.
                # If min count in ball_u is 1, we need to be careful.
                # If the word with count 1 is also in ball_v, it's fine.
                # Otherwise, invalid.
                
                # Efficient check:
                # Create mask for ball_u
                mask_u = np.zeros(N, dtype=bool)
                mask_u[ball_u] = True
                mask_v = np.zeros(N, dtype=bool)
                mask_v[ball_v] = True
                
                losing_mask = mask_u & ~mask_v
                if np.any(losing_mask):
                    if np.min(counts[losing_mask]) <= 1:
                        continue # Invalid
                
                # Valid swap
                counts[ball_u] -= 1
                counts[ball_v] += 1
                code.remove(u)
                code.append(v)
                improved = True

    # Final Cleanup
    counts = get_counts(code)
    changed = True
    while changed and time.time() - t_start < deadline:
        changed = False
        for idx in list(code):
            ball = balls_np[idx]
            if np.min(counts[ball]) > 1:
                counts[ball] -= 1
                code.remove(idx)
                changed = True

    # Convert to result
    result = []
    for idx in code:
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result
