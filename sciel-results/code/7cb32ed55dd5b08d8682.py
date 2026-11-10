"""Rewritten program for the suite task using targeted cycle-breaking local search."""
# EVOLVE-BLOCK-START
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for fast lookup
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)

    def get_bicolored_cycles(coloring):
        """
        Returns a list of tuples (c1, c2, vertices_in_component) for each bicolored cycle.
        Uses Union-Find to detect cycles in the subgraph induced by colors c1 and c2.
        """
        colors_used = list(set(coloring))
        cycles = []
        
        for i in range(len(colors_used)):
            for j in range(i + 1, len(colors_used)):
                c1 = colors_used[i]
                c2 = colors_used[j]
                
                # Only consider vertices with these two colors
                verts = [v for v in range(n) if coloring[v] in (c1, c2)]
                if len(verts) < 3: # Need at least 3 vertices for a cycle
                    continue
                    
                # Map vertex to index in local array for Union-Find
                # We only care about edges between these vertices
                parent = {v: v for v in verts}
                rank = {v: 0 for v in verts}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return True # Cycle found
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return False
                
                cycle_found = False
                # Check edges within this subset
                for u, v in edges:
                    if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                        if coloring[u] != coloring[v]:
                            if union(u, v):
                                cycle_found = True
                                break
                
                if cycle_found:
                    # Identify the component containing the cycle to pick a vertex from
                    # Actually, any vertex in the connected component with the cycle is good.
                    # For simplicity, we just pick a random vertex from the list of vertices with these colors.
                    # A more targeted approach would be to find the specific cycle, but random selection in the set usually works well for small graphs.
                    cycles.append((c1, c2, verts))
        
        return cycles

    def is_valid(coloring):
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check bicolored cycles
        if get_bicolored_cycles(coloring):
            return False
        return True

    def greedy_coloring(k):
        """Greedy coloring with k colors."""
        coloring = [-1] * n
        for i in range(n):
            neighbor_colors = set()
            for nb in adj[i]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            for c in range(k):
                if c not in neighbor_colors:
                    coloring[i] = c
                    break
            if coloring[i] == -1:
                # If no color found (should not happen if k > max degree), pick first available or random
                coloring[i] = 0 
        return coloring

    def solve_with_targeted_breaking(k, seed_val=42):
        random.seed(seed_val)
        
        # Initialize with greedy
        coloring = greedy_coloring(k)
        
        # If greedy failed to color properly (unlikely for k>=5 on 4-regular), fix it first
        # Ensure proper coloring
        for _ in range(100):
            proper = True
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    proper = False
                    # Recolor v to a random different color
                    available = [c for c in range(k) if c != coloring[u]]
                    # Check neighbors of v
                    nb_colors = set(coloring[nb] for nb in adj[v])
                    valid_options = [c for c in available if c not in nb_colors]
                    if valid_options:
                        coloring[v] = random.choice(valid_options)
                    else:
                        # Force a color, might break properness temporarily but local search fixes it
                        coloring[v] = random.choice(available)
            if proper:
                break
        
        max_iters = 5000
        no_improve_count = 0
        
        for _ in range(max_iters):
            # Check for bicolored cycles
            cycles = get_bicolored_cycles(coloring)
            
            if not cycles:
                # Also ensure proper coloring (greedy might have left conflicts if k was too small)
                proper_violations = []
                for u, v in edges:
                    if coloring[u] == coloring[v]:
                        proper_violations.append((u, v))
                
                if not proper_violations:
                    return coloring
                else:
                    # Fix proper coloring violations
                    u, v = random.choice(proper_violations)
                    nb_colors = set(coloring[nb] for nb in adj[v])
                    valid_options = [c for c in range(k) if c not in nb_colors and c != coloring[u]]
                    if valid_options:
                        coloring[v] = random.choice(valid_options)
                    else:
                        # Try u
                        nb_colors_u = set(coloring[nb] for nb in adj[u])
                        valid_options_u = [c for c in range(k) if c not in nb_colors_u and c != coloring[v]]
                        if valid_options_u:
                            coloring[u] = random.choice(valid_options_u)
                        else:
                            # Random kick
                            coloring[v] = random.randint(0, k-1)
                    continue

            # If cycles exist, pick one and break it
            c1, c2, verts = random.choice(cycles)
            
            # Select a vertex to recolor
            # Prefer vertices that have neighbors of color c1 or c2 to help break the cycle structure?
            # Or just random.
            v = random.choice(verts)
            
            old_c = coloring[v]
            
            # We must change color to something NOT c1 and NOT c2 to break the bicolored cycle
            # Also must be different from neighbors
            nb_colors = set(coloring[nb] for nb in adj[v])
            
            # Candidate colors: 0..k-1
            # Exclude c1, c2, and any neighbor colors
            candidates = []
            for c in range(k):
                if c == c1 or c == c2:
                    continue
                if c in nb_colors:
                    continue
                candidates.append(c)
            
            if candidates:
                new_c = random.choice(candidates)
                coloring[v] = new_c
                no_improve_count = 0
            else:
                # No valid candidate that avoids c1, c2 and neighbors.
                # This means the vertex is surrounded by all other colors except maybe c1/c2, 
                # or c1/c2 are the only options but they are forbidden by the cycle constraint.
                # In this case, we might need to increase k or do a more complex move.
                # For now, try a random color even if it's c1 or c2 (might not break cycle, but might break properness)
                # Better: try to recolor a neighbor to free up space? 
                # Simple fallback: random color from 0..k-1
                new_c = random.randint(0, k-1)
                if new_c == old_c:
                    new_c = (new_c + 1) % k
                coloring[v] = new_c
                no_improve_count += 1
                
                if no_improve_count > 500:
                    # Restart
                    break

        return None

    # Try increasing k until we find a valid coloring
    # The graph is 4-regular. 4-coloring might be possible. 3 is unlikely.
    # We want to MINIMIZE distinct colors.
    # So we start with k=3, then 4, etc.
    
    best_coloring = None
    
    for k in [3, 4, 5, 6, 7]:
        found = False
        for seed_val in range(20):
            res = solve_with_targeted_breaking(k, seed_val)
            if res is not None:
                # Verify
                if is_valid(res):
                    # Compact the colors to 0..m-1
                    unique_colors = sorted(set(res))
                    mapping = {c: i for i, c in enumerate(unique_colors)}
                    final_res = [mapping[c] for c in res]
                    
                    # Check if this is better than current best
                    if best_coloring is None or len(set(final_res)) < len(set(best_coloring)):
                        best_coloring = final_res
                        found = True
                        # If we found a solution with k=3 or k=4, it's likely optimal or near optimal.
                        # We can break early if k is small.
                        if k <= 4:
                            # Try a few more seeds to ensure we didn't miss a better one? 
                            # No, k is the bound. If we found a valid coloring with k=4, 
                            # the number of distinct colors is <= 4.
                            # We can't do better than 3 if k=3 failed.
                            # So if k=4 works, max distinct is 4. 
                            # We should still check if we can reduce distinct colors within the valid set?
                            # The compaction already does that.
                            # However, it's possible that a k=4 solution uses only 3 distinct colors?
                            # Yes, if one color class is empty. But greedy fills them.
                            # The compaction step `unique_colors = sorted(set(res))` handles this.
                            # So if we find a valid coloring with k=4, the result will have <= 4 distinct colors.
                            # Can we have a valid coloring with 3 distinct colors using k=4? Yes.
                            # So we should keep searching for k=3 to see if it's possible.
                            # If k=3 fails after 20 seeds, it's likely impossible or hard.
                            # But we should prioritize finding *any* valid solution first, then optimize?
                            # The task says "Minimize number of distinct colors".
                            # So we MUST try k=3 first.
                            pass
        if found and k <= 4:
            # If we found a solution with k=4, we can stop trying higher k for finding *validity*, 
            # but we should ensure we have the *minimum* distinct colors.
            # Since we iterate k from 3 upwards, the first k that yields a valid solution 
            # doesn't necessarily give the minimum distinct colors if a higher k allows a sparser coloring?
            # No, distinct colors is bounded by k. 
            # If k=3 works, distinct <= 3.
            # If k=3 fails, but k=4 works, distinct <= 4.
            # Could k=5 yield a solution with only 3 distinct colors?
            # Yes, if the k=5 search finds a coloring that only uses 3 colors.
            # But if k=3 search failed, it means we couldn't find a valid 3-coloring.
            # It is possible that a valid 3-coloring exists but our search failed to find it.
            # To be safe, we should run the search for k=4 and 5 and compare the number of distinct colors.
            
            # Let's refine: Run for k=3, 4, 5. Keep the best (lowest distinct count).
            # Break early if we find 3 distinct colors, as that is the theoretical minimum for a non-bipartite 4-regular graph (chromatic number >= 3).
            # Actually, if the graph is bipartite, 2 colors suffice. 
            # This graph contains triangles? 
            # Edges: (0,1), (1,2), (2,3)... 
            # Let's check for odd cycles.
            # 0-1-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30-31-32-33-34-35-36-37-38-39-40-41-42-43-44-45-46-47-48-49-0
            # This is a long cycle. 
            # Also (0,7), (7,6), (6,5)... 
            # It looks like a grid-like structure. 
            # Let's just rely on the algorithm.
            
            # If we found a solution with 3 distinct colors, we can stop.
            if len(set(best_coloring)) == 3:
                break
            # If we found a solution with 4 distinct colors, we should still try k=5 to see if we can get 3?
            # Unlikely if k=3 failed. But possible if the search space was explored poorly.
            # Let's continue to k=5 if we haven't found 3.
            
    if best_coloring is not None:
        return best_coloring

    # Fallback: Absolute greedy with many colors
    # This ensures we return a valid answer, even if not optimal
    coloring = [-1] * n
    for i in range(n):
        neighbor_colors = set()
        for nb in adj[i]:
            if coloring[nb] != -1:
                neighbor_colors.add(coloring[nb])
        c = 0
        while c in neighbor_colors:
            c += 1
        coloring[i] = c
        
    # This greedy produces a proper coloring. 
    # It might have bicolored cycles. 
    # We run the targeted breaking on this result with a high k.
    k_fallback = max(coloring) + 1
    # Add some buffer
    k_fallback = max(k_fallback, 10)
    
    # Force validity
    for _ in range(10000):
        cycles = get_bicolored_cycles(coloring)
        if not cycles:
            break
        c1, c2, verts = random.choice(cycles)
        v = random.choice(verts)
        nb_colors = set(coloring[nb] for nb in adj[v])
        candidates = [c for c in range(k_fallback) if c != c1 and c != c2 and c not in nb_colors]
        if candidates:
            coloring[v] = random.choice(candidates)
        else:
            # Pick any color not in nb_colors, even if it's c1 or c2, to ensure properness first?
            # No, we need to break the cycle. 
            # If no candidate, we must expand k or recolor neighbor.
            # For fallback, just pick random from 0..k_fallback-1
            coloring[v] = random.randint(0, k_fallback - 1)
            
    # Final verification and compaction
    if is_valid(coloring):
        unique_colors = sorted(set(coloring))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in coloring]
    else:
        # Should not happen, but return a valid list of ints
        return list(range(n))

# EVOLVE-BLOCK-END
