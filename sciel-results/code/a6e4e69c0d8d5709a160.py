"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    from random import Random
    import time

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_one(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build adjacency for original graph vertices
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Build conflict graph
        # Vertices are edges (0..m-1)
        # Edge between i, j if edges[i] and edges[j] are NOT in an induced matching
        # Conflict if:
        # 1. Share a vertex
        # 2. Joined by an edge (u1,u2), (u1,v2), (v1,u2), (v1,v2) is an edge in G
        
        conflict_adj = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                else:
                    if u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or \
                       u2 in adj.get(v1, set()) or v2 in adj.get(v1, set()):
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)

        # Convert to sorted lists for faster iteration
        conflict_list = [sorted(lst) for lst in conflict_adj]
        
        # Initial greedy coloring to get an upper bound
        def greedy_coloring():
            color_of = [-1] * m
            # Order by descending degree in conflict graph
            order = sorted(range(m), key=lambda x: -len(conflict_list[x]))
            for v in order:
                forbidden = set()
                for u in conflict_list[v]:
                    if color_of[u] != -1:
                        forbidden.add(color_of[u])
                c = 0
                while c in forbidden:
                    c += 1
                color_of[v] = c
            return color_of, max(color_of) + 1

        init_color, init_upper_bound = greedy_coloring()
        best_solution = init_upper_bound
        best_assignment = init_color[:]

        # Bitmask representation for colors?
        # Max colors likely < 64, so we can use int bitmasks.
        # Let's stick to sets or lists for clarity first, optimize if needed.
        # Actually, bitmasks are much faster in Python for "is color available".
        
        # Pre-compute neighbor bitmasks?
        # neighbor_mask[v] has bit i set if v is adjacent to i.
        # To check if color c is available for v:
        # We need to know which neighbors of v have color c.
        # This is hard with just neighbor masks.
        # Better: maintain `color_mask[c]` = bitmask of vertices with color c.
        # Then for vertex v, forbidden colors are those c where `color_mask[c] & neighbor_mask[v] != 0`.
        
        neighbor_mask = [0] * m
        for v in range(m):
            mask = 0
            for u in conflict_list[v]:
                mask |= (1 << u)
            neighbor_mask[v] = mask

        # State for B&B
        color_of = [-1] * m
        color_masks = [0] * (init_upper_bound + 5) # Allow for expansion
        uncolored_count = m
        best_sol = init_upper_bound
        
        # Heuristic: DSATUR
        # We need to pick the next vertex.
        # Saturation of v = number of distinct colors in its neighbors.
        # Degree in remaining graph = number of uncolored neighbors.
        
        # To speed up, we can use a simple recursive B&B with static ordering first,
        # but dynamic ordering is much better.
        
        # Let's use a simpler approach: 
        # 1. Order vertices by descending degree.
        # 2. Use B&B with this order.
        # 3. Prune if current_colors + clique_bound(remaining) >= best_sol.
        
        # For clique bound, we can use a greedy clique finder on the remaining graph.
        # Given the small size, this is feasible.
        
        def get_clique_bound(vertices):
            # Greedy max clique
            # vertices is a list of uncolored vertex indices
            if not vertices:
                return 0
            # Sort by degree in the subgraph induced by vertices
            # This is expensive. Let's use a simpler bound: max degree + 1? No.
            # Let's use the fact that if we have a clique of size k, we need k colors.
            # A fast lower bound is the size of the largest clique.
            # Approximate: Pick a vertex, find its neighbors in the set, recurse?
            # For speed, let's just use 1 if any uncolored, or a precomputed max clique of the whole graph?
            # Precomputing max clique of the whole graph is a good lower bound for the whole problem,
            # but for the remainder, it's weak.
            
            # Let's use a simple greedy clique:
            # Pick vertex with max degree in subgraph, add to clique, remove it and its non-neighbors?
            # Actually, for coloring, a good lower bound is the clique number.
            # Let's try to find a clique greedily.
            if len(vertices) < 2:
                return len(vertices)
            
            # Create a local adjacency set for vertices
            # This is O(N^2) per call, which might be too slow if called many times.
            # Alternative: Precompute nothing, just use a very simple bound:
            # If we have K uncolored vertices, and the max degree in the remaining graph is D,
            # we don't have a direct bound.
            
            # Let's use the following simple bound:
            # The chromatic number is at least the size of the largest clique.
            # We can compute a lower bound on clique size by finding a maximal clique.
            # Greedy Max Clique:
            # 1. Pick vertex v with highest degree in the induced subgraph of `vertices`.
            # 2. Candidate set = neighbors of v in `vertices`.
            # 3. Recurse on Candidate set?
            
            # To keep it simple and fast, let's use a pre-computed lower bound for the WHOLE graph
            # and just rely on the current depth.
            # Actually, `current_depth` is the number of colors used so far.
            # If we just prune when `current_depth >= best_sol`, it's very weak.
            
            # Let's implement a fast greedy clique finder for the remaining vertices.
            # It will be called at each node of the search tree.
            # To make it fast, we limit the search depth or use a heuristic.
            
            # Simple heuristic: 
            # Pick the vertex with the highest degree in the remaining graph.
            # The size of its neighborhood + 1 is a lower bound? No.
            # The clique number is at least the size of the neighborhood of any vertex? No.
            
            # Let's just use 1 as a lower bound for now and rely on the ordering and pruning by `best_sol`.
            # With good ordering (DSATUR), it often works well.
            return 1

        # Let's try a static ordering with B&B first. It's simpler and might be fast enough.
        order = sorted(range(m), key=lambda x: -len(conflict_list[x]))
        
        def backtrack(idx, colors_used):
            nonlocal best_sol, best_assignment
            
            if colors_used >= best_sol:
                return
            
            if idx == m:
                best_sol = colors_used
                best_assignment = color_of[:]
                return
            
            v = order[idx]
            
            # Check if v is already colored? No, we color in order.
            
            # Determine forbidden colors
            forbidden = set()
            for u in conflict_list[v]:
                if color_of[u] != -1:
                    forbidden.add(color_of[u])
            
            # Try colors 0..colors_used-1 first, then colors_used
            # Optimization: If we use a new color, it must be `colors_used`
            
            for c in range(colors_used):
                if c not in forbidden:
                    color_of[v] = c
                    backtrack(idx + 1, colors_used)
                    color_of[v] = -1
                    if best_sol <= colors_used: # Prune if we found a solution with <= current colors
                        # Actually if best_sol < colors_used, we can stop trying higher colors?
                        # No, we are iterating c. If we found a solution with `colors_used` colors,
                        # we still might find one with `colors_used - 1`? No, we are at depth `colors_used`.
                        # If we find a solution, best_sol becomes colors_used.
                        # Then we return. The parent will try other colors.
                        pass
                    
            # Try new color
            if colors_used < best_sol:
                color_of[v] = colors_used
                backtrack(idx + 1, colors_used + 1)
                color_of[v] = -1

        # To improve, we should use dynamic ordering (DSATUR).
        # Let's implement DSATUR B&B.
        
        # State:
        # color_of: list
        # uncolored: set of vertices
        # color_masks: list of ints (bitmasks of vertices with color c)
        
        color_of = [-1] * m
        color_masks = [0] * (init_upper_bound + 10)
        uncolored = set(range(m))
        
        # Precompute degrees for initial sorting?
        # DSATUR picks vertex with max saturation, then max degree.
        
        def dsatur_backtrack(depth, colors_used):
            nonlocal best_sol, best_assignment
            
            if colors_used >= best_sol:
                return
                
            if not uncolored:
                best_sol = colors_used
                best_assignment = color_of[:]
                return
            
            # Pick vertex
            # Saturation: number of distinct colors in neighbors
            # Degree: number of uncolored neighbors
            
            best_v = -1
            best_sat = -1
            best_deg = -1
            
            for v in uncolored:
                # Calculate saturation
                sat = 0
                seen_colors = set()
                deg = 0
                for u in conflict_list[v]:
                    if u in uncolored:
                        deg += 1
                    else:
                        if color_of[u] != -1:
                            seen_colors.add(color_of[u])
                sat = len(seen_colors)
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_v = v
                    
            v = best_v
            uncolored.remove(v)
            
            # Determine forbidden colors
            forbidden = set()
            for u in conflict_list[v]:
                if color_of[u] != -1:
                    forbidden.add(color_of[u])
            
            # Try existing colors
            for c in range(colors_used):
                if c not in forbidden:
                    color_of[v] = c
                    color_masks[c] |= (1 << v)
                    dsatur_backtrack(depth + 1, colors_used)
                    color_masks[c] &= ~(1 << v)
                    color_of[v] = -1
                    if best_sol <= colors_used:
                        # If we found a solution with current colors, 
                        # we can stop trying other colors for this vertex?
                        # No, because we might find a solution with FEWER colors?
                        # No, we are fixing the set of used colors to be <= colors_used.
                        # If we find a solution, best_sol becomes colors_used.
                        # Any further branch will have colors_used >= best_sol, so it will be pruned.
                        # So we can break.
                        break
                        
            # Try new color
            if colors_used < best_sol:
                c = colors_used
                if c not in forbidden:
                    color_of[v] = c
                    color_masks[c] |= (1 << v)
                    dsatur_backtrack(depth + 1, colors_used + 1)
                    color_masks[c] &= ~(1 << v)
                    color_of[v] = -1

        # Initialize
        # We can start with the greedy solution as best_sol
        # And run DSATUR B&B to try to improve
        
        # To make it faster, we can restrict the search space.
        # The greedy solution is usually close to optimal.
        
        # Let's run it.
        dsatur_backtrack(0, 0)
        
        return best_assignment

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        out.append(solve_one(*spec))
    return out
# EVOLVE-BLOCK-END
