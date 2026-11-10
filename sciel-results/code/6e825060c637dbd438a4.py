import random
import time

def construct(q, n, radius):
    # Task: Covering Code K_q(n, radius)
    # q=3, n=8, radius=1 typically.
    
    total_words = q ** n
    
    # Precompute neighbors for distance <= radius (for covering check)
    # And neighbors for distance <= 2 (for the MIS heuristic graph)
    # Note: For q=3, n=8, total_words=6561.
    
    # Helper to convert index to tuple
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    # Helper to convert tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute all words
    all_words = [idx_to_tuple(i) for i in range(total_words)]
    
    # Precompute powers of q for index arithmetic
    q_powers = [q ** i for i in range(n)]
    
    # Precompute neighbors for covering (distance <= radius)
    # This is used for the final validation and local search
    cover_neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        cover_neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                diff = (new_val - orig_val) * q_powers[n - 1 - pos]
                cover_neighbors[i].append(i + diff)
                
    # Precompute neighbors for the "MIS" graph (distance <= 2)
    # Edges connect pairs at Hamming distance <= 2.
    # We want an Independent Set in this graph.
    # This means no two centers in the initial set are at distance <= 2.
    # i.e., initial centers are at distance >= 3.
    
    # Generating distance-2 neighbors for 6561 nodes is feasible.
    # 1 + 8*2 + 28*4 = 1 + 16 + 112 = 129 neighbors per node.
    # Total edges ~ 6561 * 129 / 2 ~ 400k. Feasible.
    
    dist2_neighbors = [set() for _ in range(total_words)]
    
    for i, w in enumerate(all_words):
        dist2_neighbors[i].add(i)
        # Distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                diff = (new_val - orig_val) * q_powers[n - 1 - pos]
                dist2_neighbors[i].add(i + diff)
        # Distance 2
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                v1 = w[pos1]
                v2 = w[pos2]
                for new_v1 in range(q):
                    if new_v1 == v1:
                        continue
                    for new_v2 in range(q):
                        if new_v2 == v2:
                            continue
                        diff = (new_v1 - v1) * q_powers[n - 1 - pos1] + (new_v2 - v2) * q_powers[n - 1 - pos2]
                        dist2_neighbors[i].add(i + diff)
                        
    # Convert sets to lists for faster iteration if needed, but sets are good for membership.
    # For the greedy MIS, we need to iterate neighbors.
    
    # Step 1: Greedy Maximal Independent Set in the dist-2 graph
    # This creates a "Packing" code.
    # Heuristic: Pick vertices with lowest degree first? Or random?
    # Low degree vertices are less likely to block other vertices, so they allow a larger independent set.
    # However, we want a *maximal* independent set, not necessarily maximum.
    # A common heuristic for MIS is to pick the vertex with the minimum degree in the remaining graph.
    
    remaining = set(range(total_words))
    centers = []
    
    # To speed up, we can just pick random vertices from remaining, 
    # but "min degree" is better for size. 
    # Given time limits, random might be faster to implement correctly without dynamic degree updates.
    # Let's try random selection from remaining, removing neighbors.
    # This is fast.
    
    # Shuffle to avoid bias
    candidates = list(remaining)
    random.shuffle(candidates)
    
    for v in candidates:
        if v in remaining:
            centers.append(v)
            # Remove v and all its dist-2 neighbors from remaining
            for u in dist2_neighbors[v]:
                if u in remaining:
                    remaining.discard(u)
            if not remaining:
                break
                
    # Step 2: Complete the cover
    # The set 'centers' is a packing (min dist 3). It may not cover the space.
    # We need to add centers to cover all uncovered words.
    
    covered = [False] * total_words
    for c in centers:
        for nb in cover_neighbors[c]:
            covered[nb] = True
            
    # Greedy Set Cover for the rest
    uncovered = [i for i in range(total_words) if not covered[i]]
    
    while uncovered:
        best_idx = -1
        best_gain = 0
        # To speed up, we can just pick a random uncovered word and cover it?
        # Or pick the center that covers the most uncovered.
        # Since 'uncovered' might be small, we can iterate through all potential centers?
        # Or just pick a random uncovered word u, and pick a random center from cover_neighbors[u]?
        # That's a "First Fit" approach. It might add too many centers.
        # Let's do a simple greedy: for each uncovered word, pick the neighbor that covers the most *other* uncovered words.
        
        # Optimization: If uncovered is small, just pick one.
        # If large, do greedy.
        
        # For efficiency in Python, iterating 6561 * len(uncovered) might be slow if len(uncovered) is large.
        # But the MIS step should cover a significant portion.
        
        # Let's use a simpler heuristic: Pick a random uncovered word, pick a random center covering it.
        # This is fast but might result in a larger code.
        # Let's try a slightly better greedy:
        # Count how many uncovered words each potential center covers.
        # Only consider centers that cover at least one uncovered word.
        
        gain_count = [0] * total_words
        for u in uncovered:
            for c in cover_neighbors[u]:
                gain_count[c] += 1
                
        max_gain = 0
        best_c = -1
        for c in range(total_words):
            if gain_count[c] > max_gain:
                max_gain = gain_count[c]
                best_c = c
                
        if best_c == -1:
            break # Should not happen
            
        centers.append(best_c)
        
        # Update covered and uncovered
        for nb in cover_neighbors[best_c]:
            if not covered[nb]:
                covered[nb] = True
                # Remove from uncovered? We'll rebuild uncovered list or use a set.
                # Rebuilding is O(N).
                
        # Rebuild uncovered list
        uncovered = [i for i in range(total_words) if not covered[i]]
        
    # Step 3: Local Search to minimize centers
    # We have a valid covering code. Now minimize.
    
    # Convert centers to a set for O(1) lookup
    center_set = set(centers)
    
    # Maintain cover_count
    cover_count = [0] * total_words
    for c in centers:
        for nb in cover_neighbors[c]:
            cover_count[nb] += 1
            
    start_time = time.time()
    max_time = 115.0
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        
        # 1. Reduction: Remove redundant centers
        # A center is redundant if all its neighbors are covered by other centers.
        # i.e., for all nb in cover_neighbors[c], cover_count[nb] > 1.
        
        # Shuffle centers to randomize
        centers_list = list(center_set)
        random.shuffle(centers_list)
        
        for c in centers_list:
            if time.time() - start_time > max_time:
                break
            redundant = True
            for nb in cover_neighbors[c]:
                if cover_count[nb] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for nb in cover_neighbors[c]:
                    cover_count[nb] -= 1
                center_set.remove(c)
                improved = True
                break # Restart loop to re-evaluate others
                
        if time.time() - start_time > max_time:
            break
        if improved:
            continue
            
        # 2. Swap: Try to swap a center c with a non-center w
        # Condition: w covers all neighbors of c that are ONLY covered by c.
        
        # Pick a random center
        if not center_set:
            break
            
        c = random.choice(list(center_set))
        
        # Find critical neighbors of c
        critical = [nb for nb in cover_neighbors[c] if cover_count[nb] == 1]
        
        if not critical:
            # Should have been removed in reduction phase
            for nb in cover_neighbors[c]:
                cover_count[nb] -= 1
            center_set.remove(c)
            improved = True
            continue
            
        # We need a w such that critical is a subset of cover_neighbors[w].
        # w must be in the intersection of cover_neighbors[x] for all x in critical.
        # Let's pick a candidate w from cover_neighbors[critical[0]].
        
        candidates_w = cover_neighbors[critical[0]]
        
        # To improve success rate, we can try a few random candidates
        found_swap = False
        for _ in range(5): # Try 5 random swaps
            if time.time() - start_time > max_time:
                break
                
            if not candidates_w:
                break
                
            w = random.choice(candidates_w)
            
            # Check if w covers all critical neighbors
            valid = True
            for nb in critical:
                if nb not in cover_neighbors[w]:
                    valid = False
                    break
                    
            if valid:
                # Perform swap
                # Remove c
                for nb in cover_neighbors[c]:
                    cover_count[nb] -= 1
                center_set.remove(c)
                
                # Add w
                for nb in cover_neighbors[w]:
                    cover_count[nb] += 1
                center_set.add(w)
                
                improved = True
                found_swap = True
                break
                
    # Final Result
    result = [all_words[c] for c in center_set]
    return [list(w) for w in result]
