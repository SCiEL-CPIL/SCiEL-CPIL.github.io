import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Calculate forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency bitsets
    # adj[v] is a bitmask where bit u is 1 if u is adjacent to v (conflict)
    adj = [0] * m
    for u in range(m):
        for v in range(u + 1, m):
            if (u - v) % m in forbidden:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    best_size = 0
    best_set = []
    
    # Global deadline
    start_time = time.time()
    TIME_LIMIT = 590  # Leave 10s margin
    
    # Greedy coloring for upper bound
    # Returns: (number_of_colors, ordered_list_of_vertices)
    # The ordered list is such that vertices in later color classes come later in the list.
    # We process the list from end to beginning.
    def greedy_coloring(vertices):
        if not vertices:
            return 0, []
        
        # Sort vertices by degree in the induced subgraph descending?
        # Tomita's algorithm often uses a static order or dynamic.
        # Let's try sorting by degree in the full graph first, or just use the given order.
        # For better bounds, sorting by degree descending in the induced subgraph is good.
        # However, calculating degree in induced subgraph for each step is expensive.
        # A common heuristic is to just process vertices in the order they are given, 
        # but the "order" itself is determined by the previous coloring step?
        # Actually, standard implementation:
        # 1. Input: set P
        # 2. While P is not empty:
        #    - Pick a vertex v from P with highest degree in G[P] (or just first one)
        #    - Assign it a new color if it conflicts with current color class?
        #    No, greedy coloring for *bound* works like this:
        #    We partition P into maximal independent sets.
        #    Color 1: Maximal independent set in P. Remove from P.
        #    Color 2: Maximal independent set in remaining P. Remove from P.
        #    ...
        #    The number of colors is the bound.
        #    To make the bound tight, we should pick vertices that are "central" (high degree) 
        #    to force many colors? Or "peripheral"?
        #    Actually, if we pick high degree vertices first for the *first* color class, 
        #    they constrain many others, potentially leaving fewer vertices for subsequent classes?
        #    Let's stick to a simple greedy: iterate through vertices, assign smallest available color.
        #    But we need the list of vertices sorted by color for the branching order.
        
        # Better approach for Tomita:
        # We maintain a list of vertices.
        # We compute colors.
        # The bound is the max color assigned.
        # The branching order is reverse color order.
        
        # Let's implement the "color sort" step.
        # Input: list of vertices
        # Output: list of vertices sorted such that all vertices of color 1 come first, then 2, etc.
        # And the max color used.
        
        colors = {}
        uncolored = set(vertices)
        max_color = 0
        
        # To get a good bound, we should process vertices in an order that maximizes conflicts early?
        # Actually, for the upper bound, we just need *a* valid coloring. 
        # The quality of the bound depends on the number of colors.
        # A "dense" packing (few colors) gives a better bound.
        # So we want to pack as many vertices as possible into each color class.
        # This means for Color 1, we want a large Independent Set.
        # Greedy: Pick vertex with highest degree in remaining graph? 
        # If we pick high degree, it removes many neighbors from consideration for Color 1?
        # No, if we pick v for Color 1, we *cannot* pick its neighbors for Color 1.
        # So picking a high degree vertex reduces the pool for Color 1 significantly.
        # This might lead to *more* colors.
        # Conversely, picking low degree vertices allows packing more into Color 1.
        # So for the *first* color class, we should pick vertices with *low* degree in the induced subgraph?
        # Or just pick them in an order that allows maximal independent set construction.
        
        # Let's try: Sort vertices by degree in induced subgraph ASCENDING.
        # Then greedily assign to smallest color.
        
        # Pre-compute degrees in induced subgraph?
        # For N=205, we can just compute on the fly or sort once.
        
        # Let's use a simple greedy coloring that produces a list.
        # We will iterate through vertices sorted by degree (descending) to find a good coloring?
        # Actually, the standard Tomita algorithm uses the order from the *previous* step or a static order.
        # Let's use a dynamic order: sort by degree in the induced subgraph, descending.
        # Why descending? Because high degree vertices are more "constrained" in terms of being placed?
        # Wait, the bound is on the *independent set*. 
        # The coloring is of the *graph*. 
        # If we color the graph with k colors, alpha <= k.
        # We want k to be small.
        # To minimize k, we should pack independent sets.
        
        # Let's just use a standard greedy coloring:
        # Order vertices by degree in G[P] descending.
        # Assign smallest possible color.
        
        # 1. Calculate degrees in G[P]
        degs = {}
        for v in vertices:
            # Count neighbors in vertices
            # adj[v] is global bitmask. 
            # We need to mask with vertices bitmask.
            v_mask = 0
            for u in vertices:
                if u == v: continue
                if (adj[v] >> u) & 1:
                    v_mask |= (1 << u)
            # This is O(N^2) per coloring step. N=205, steps ~ N. Total O(N^3) ~ 8M ops. Acceptable.
            # But we can do better with bitsets.
            # Let's create a bitmask for the set of vertices.
            pass
            
        # Let's use a bitmask for the set of vertices to speed up degree calculation.
        # But we need the list for output.
        
        # Let's try a different strategy: 
        # The "color" of a vertex v in the greedy coloring is determined by the colors of its neighbors.
        
        # Implementation of greedy coloring returning ordered list:
        # 1. Create a set of uncolored vertices.
        # 2. While uncolored:
        #    a. Find a maximal independent set (MIS) in the induced subgraph of uncolored.
        #       Greedy MIS: 
        #       - Sort uncolored by degree in induced subgraph (descending? ascending?)
        #       - Pick v. Add to current color class. Remove v and N(v) from uncolored? 
        #         No, remove v from uncolored. Mark N(v) as "cannot take this color".
        #       - To get a *large* MIS (to reduce number of colors), we should pick vertices that 
        #         don't conflict with many others? i.e. low degree?
        #       - Let's pick vertices with *lowest* degree in the remaining uncolored graph.
        #    b. Assign current color to all vertices in this MIS.
        #    c. Remove them from uncolored.
        #    d. Increment color count.
        
        # This "Maximal Independent Set per color" approach is equivalent to greedy coloring 
        # if we process vertices in a specific order.
        
        # Let's implement:
        # vertices: list
        # uncolored: set
        # color_idx: 0
        # result_list: []
        
        uncolored = set(vertices)
        color_idx = 0
        ordered_vertices = []
        
        while uncolored:
            color_idx += 1
            # Find MIS in uncolored
            # Sort uncolored by degree in G[uncolored]
            # Degree calculation:
            # For v in uncolored:
            #   deg = popcount(adj[v] & uncolored_mask)
            
            uncolored_mask = 0
            for v in uncolored:
                uncolored_mask |= (1 << v)
                
            # Calculate degrees
            degs = {}
            for v in uncolored:
                # Count bits in (adj[v] & uncolored_mask)
                # Python 3.8+ has int.bit_count()
                degs[v] = (adj[v] & uncolored_mask).bit_count()
                
            # Sort by degree ascending to pick "easy" vertices first for the color class?
            # If we pick a low degree vertex, it removes few neighbors from the color class,
            # allowing more vertices to be added to this color class.
            # This maximizes the size of the color class, minimizing the total number of colors.
            sorted_uncolored = sorted(uncolored, key=lambda x: degs[x])
            
            current_color_class = []
            # Greedily add vertices to current_color_class
            # We need to ensure they are independent.
            # We maintain a set of "forbidden" vertices for this color class.
            forbidden_for_color = set()
            
            for v in sorted_uncolored:
                if v in uncolored and v not in forbidden_for_color:
                    current_color_class.append(v)
                    # Add neighbors of v to forbidden_for_color
                    # Neighbors of v in uncolored
                    # We can just iterate through uncolored and check adjacency
                    # Or use bitmask
                    # adj[v] & uncolored_mask
                    # But we need to update uncolored? No, we just mark as forbidden for this color.
                    # Actually, if we pick v, we remove v from uncolored.
                    # And its neighbors CANNOT be in this color class.
                    # So we add N(v) to forbidden_for_color.
                    # We don't remove N(v) from uncolored yet, they might be in a later color.
                    
                    # Update forbidden_for_color
                    # Get neighbors of v in uncolored
                    # We can iterate through uncolored
                    for u in uncolored:
                        if u != v and ((adj[v] >> u) & 1):
                            forbidden_for_color.add(u)
                            
                    uncolored.remove(v)
                    
            ordered_vertices.extend(current_color_class)
            
        return color_idx, ordered_vertices

    def branch_and_bound(candidates, current_set):
        nonlocal best_size, best_set
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return
            
        # Get upper bound and ordered candidates
        # The ordered candidates are sorted by color (1, 2, ..., k)
        # We want to branch on the vertex with the *highest* color first?
        # Tomita's algorithm: 
        # The list P is ordered by color.
        # We pick the *last* vertex in P (highest color).
        # Why? Because if we exclude it, we just remove it from P.
        # If we include it, we remove its neighbors.
        # The bound is the number of colors.
        # If we pick the last vertex (color k), and include it, the new bound is k-1 (since color k is gone) + 1?
        # Actually, the new candidate set will have max color < k.
        
        bound, ordered_candidates = greedy_coloring(candidates)
        
        if len(current_set) + bound <= best_size:
            return
            
        # We process vertices in reverse order of the ordered list (highest color first)
        # The ordered list has vertices of color 1 first, then 2, etc.
        # So we iterate from end to start.
        
        # To allow backtracking and pruning, we can iterate through the vertices 
        # and for each one, try to include it.
        # If including it doesn't improve the best, we continue to the next vertex (which is effectively "excluding" the previous ones).
        
        # Let's implement the recursive step carefully.
        # Standard Tomita:
        # def expand(P, C):
        #   if P is empty: update best
        #   color(P) -> k, P ordered by color
        #   if |C| + k <= best: return
        #   for i from k down to 1:
        #     let P_i be vertices in P with color i
        #     for v in P_i (in some order):
        #       C.add(v)
        #       P_new = P \ (N(v) U {v})  # Note: P_new should only contain vertices with color < i?
        #       # Actually, if we pick v (color i), any vertex with color >= i is either v or a neighbor?
        #       # No. Vertices with color i are independent. So N(v) has no vertices of color i.
        #       # So P_new will not contain any vertices of color i (except v which is removed).
        #       # So the new max color is i-1.
        #       expand(P_new, C)
        #       C.remove(v)
        #       # Remove v from P? 
        #       # In the loop "for v in P_i", we are iterating through the vertices of color i.
        #       # After trying v, we effectively "exclude" v for the remaining iterations of the inner loop?
        #       # No, the structure is:
        #       # We iterate through color classes from k down to 1.
        #       # Within a color class, we iterate through vertices.
        #       # When we pick v, we recurse.
        #       # Then we remove v from P? 
        #       # If we don't remove v, we might pick it again? 
        #       # No, we are iterating through a snapshot of P_i.
        #       # But we must ensure that if we didn't pick v, we don't pick it again in a later color class?
        #       # Since v has color i, it won't be in color classes < i.
        #       # So we just need to remove v from P before moving to the next vertex in P_i?
        #       # Yes, because if we skip v, it's excluded from the solution in this branch.
        
        # Let's implement this loop structure.
        
        # We need to map vertices to their colors from the greedy_coloring result.
        # But greedy_coloring returns a list. We can infer color by position?
        # No, the list is just extended. We can store colors in a dict or parallel list.
        
        # Let's modify greedy_coloring to return (bound, list_of_(vertex, color))
        
        # Re-implementing greedy_coloring to return colors
        def greedy_coloring_with_colors(vertices):
            if not vertices:
                return 0, []
            
            uncolored = set(vertices)
            color_idx = 0
            colored_vertices = [] # List of (v, color)
            
            while uncolored:
                color_idx += 1
                uncolored_mask = 0
                for v in uncolored:
                    uncolored_mask |= (1 << v)
                    
                degs = {}
                for v in uncolored:
                    degs[v] = (adj[v] & uncolored_mask).bit_count()
                    
                sorted_uncolored = sorted(uncolored, key=lambda x: degs[x])
                
                forbidden_for_color = set()
                
                for v in sorted_uncolored:
                    if v in uncolored and v not in forbidden_for_color:
                        colored_vertices.append((v, color_idx))
                        for u in uncolored:
                            if u != v and ((adj[v] >> u) & 1):
                                forbidden_for_color.add(u)
                        uncolored.remove(v)
                        
            return color_idx, colored_vertices

        bound, colored_candidates = greedy_coloring_with_colors(candidates)
        
        if len(current_set) + bound <= best_size:
            return
            
        # Group candidates by color
        # We process colors from high to low
        # For each color, we process vertices.
        
        # To allow the "exclude" logic, we iterate through the vertices of the current highest color.
        # If we include v, we recurse on the remaining candidates (which will have max color < current).
        # If we exclude v, we just remove it from the candidate set for the next iteration of the loop.
        
        # We need to maintain the candidate set.
        # Let's use a list of vertices for the current color class.
        
        # We iterate i from bound down to 1
        # But the set of candidates changes as we "exclude" vertices.
        # Actually, the standard algorithm works on the set P.
        # We pick a vertex v from P (highest color).
        # Branch 1: Include v. New P = P \ (N(v) U {v}).
        # Branch 2: Exclude v. New P = P \ {v}.
        
        # We can implement this recursively.
        # But the loop version is more efficient.
        
        # Let's stick to the recursive version for clarity, as N is small.
        
        # We need to pick the vertex with the highest color.
        # If there are multiple, pick one (e.g., highest degree).
        
        # Find max color
        max_color = max(c for _, c in colored_candidates)
        
        # Get vertices with max color
        candidates_max_color = [v for v, c in colored_candidates if c == max_color]
        
        # Sort them? Maybe by degree descending?
        # Let's just pick the first one. Or iterate through all?
        # The loop version iterates through all vertices of the max color.
        # Let's implement the loop version properly.
        
        # We need to maintain the set of candidates.
        # Let's represent candidates as a set for O(1) removal.
        # But we also need the coloring for the bound.
        # The coloring is only valid for the *current* set of candidates.
        # If we remove a vertex, the bound might decrease, but we don't recalculate it immediately?
        # No, in the loop version, we recalculate the bound at the start of the function call.
        # Inside the loop, we just branch.
        
        # So, we have `candidates` (set) and `colored_candidates` (list).
        # We iterate through vertices in `candidates` that have the max color.
        
        # To do this, we can just iterate through `colored_candidates` and filter.
        
        # But we need to remove vertices from `candidates` as we "exclude" them.
        
        # Let's try:
        # For each vertex v in candidates_max_color (in some order):
        #   1. Include v:
        #      new_candidates = candidates - N(v) - {v}
        #      recurse(new_candidates, current_set + {v})
        #   2. Exclude v:
        #      candidates.remove(v)
        #      # Continue loop with updated candidates?
        #      # The bound was calculated for the original candidates.
        #      # If we remove v, the bound is still valid (it's an upper bound).
        #      # So we can just continue the loop.
        
        # This works!
        
        # We need to be careful: `candidates_max_color` is a snapshot.
        # As we remove vertices, the set `candidates` shrinks.
        # We should iterate through the snapshot.
        
        # Order of vertices in candidates_max_color:
        # Maybe highest degree first?
        
        # Let's sort candidates_max_color by degree in the induced subgraph descending.
        # This is a heuristic.
        
        # Calculate degrees in candidates
        # We already have degs from coloring? No, that was for uncolored.
        # Let's just use global degree or compute quickly.
        
        # For simplicity, let's just iterate in the order they appear in colored_candidates.
        
        for v in candidates_max_color:
            # Check if v is still in candidates (it should be, unless removed by previous exclusions? 
            # No, we only remove v itself in the exclude step. 
            # But what if v was a neighbor of a previously included vertex? 
            # No, we are in the "exclude" path of the previous iterations? 
            # Wait. 
            # If we include a vertex u in a previous iteration of the loop (from a different branch? No, this is sequential).
            # The loop structure is:
            # for v in ...:
            #   include(v) -> recurse
            #   exclude(v) -> remove v from candidates
            
            # So if we are at v, all previous vertices in the loop were "excluded" (removed from candidates).
            # So v is still in candidates.
            
            if v not in candidates:
                continue
                
            # Include v
            # new_candidates = candidates \ (N(v) U {v})
            # We need to compute N(v) intersect candidates.
            
            # Mask for candidates
            cand_mask = 0
            for u in candidates:
                cand_mask |= (1 << u)
                
            # Neighbors of v in candidates
            # adj[v] & cand_mask
            # We need to convert this back to a set? 
            # Or just iterate through candidates and check.
            
            new_candidates_set = set()
            for u in candidates:
                if u != v and ((adj[v] >> u) & 1) == 0:
                    new_candidates_set.add(u)
                    
            new_current_set = current_set | {v}
            
            branch_and_bound(new_candidates_set, new_current_set)
            
            # Exclude v
            candidates.remove(v)
            
            # Check time
            if time.time() - start_time > TIME_LIMIT:
                return

    # Initial call
    all_vertices = set(range(m))
    branch_and_bound(all_vertices, set())
    
    return np.array(best_set, dtype=np.int64)

# Run once to verify? No, just return the function.
# The function will be called by the evaluator.

# One optimization: 
# The greedy_coloring is called at every node of the B&B tree.
# For N=205, this might be slow if the tree is large.
# But with good bounds, the tree should be small.
# The bit operations are fast.

# Let's double check the logic.
# If we exclude v, we remove it from candidates.
# The bound `bound` was calculated for the *original* candidates.
# Is it still a valid upper bound for `candidates \ {v}`?
# Yes, removing a vertex cannot increase the size of the MIS, so the old bound is still an upper bound.
# So we don't need to recalculate the bound inside the loop.

# This looks correct.

# One detail: `candidates_max_color` is derived from `colored_candidates`.
# If we remove vertices from `candidates`, `colored_candidates` still contains them.
# We check `if v not in candidates: continue`. This handles it.

# Final check on return type: `np.ndarray` of `np.int64`.

