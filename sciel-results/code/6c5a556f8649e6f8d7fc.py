"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate a bounded degree graph
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def get_neighborhoods(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_induced_matching_valid(color_class_edges, adj, edges_set):
        # Check if the set of edges forms an induced matching
        # 1. No two edges share a vertex
        # 2. No two edges are joined by an edge (i.e., the subgraph induced by their endpoints has no other edges)
        vertices_in_class = set()
        for u, v in color_class_edges:
            if u in vertices_in_class or v in vertices_in_class:
                return False
            vertices_in_class.add(u)
            vertices_in_class.add(v)
        
        # Check for "joined by an edge"
        # For each pair of edges (u,v) and (x,y) in the class, check if any edge exists between {u,v} and {x,y} other than the matching edges themselves.
        # Since they don't share vertices, we just check cross-edges.
        for i, (u, v) in enumerate(color_class_edges):
            for x, y in color_class_edges[i+1:]:
                # Check all 4 cross connections
                if v in adj[u] and v != x and v != y: # This is wrong, adj[u] contains neighbors.
                    pass
                # Correct check:
                # Is there an edge (u,x), (u,y), (v,x), (v,y)?
                if (u, x) in edges_set or (u, y) in edges_set or (v, x) in edges_set or (v, y) in edges_set:
                    return False
        return True

    def greedy_coloring(edges, adj, n):
        # Initial coloring using greedy algorithm
        color_map = {}
        colors = 0
        for e in edges:
            u, v = e
            # Forbidden colors are those used by edges that conflict with e
            # Conflict if they share a vertex OR are joined by an edge.
            # A color c is forbidden if there exists an edge e'=(u',v') with color c such that:
            # (u,v) and (u',v') share a vertex OR there is an edge between {u,v} and {u',v'}
            
            forbidden = set()
            # Optimization: Pre-compute "conflict edges" for each edge?
            # For small graphs (n=80, m~240), O(m^2) per step is 240^2 * 240 ~ 13M ops, might be slow in Python.
            # Let's try a slightly smarter greedy.
            
            # For now, simple greedy
            for e2, c in color_map.items():
                u2, v2 = e2
                # Check conflict
                # 1. Share vertex
                if u == u2 or u == v2 or v == u2 or v == v2:
                    forbidden.add(c)
                    continue
                # 2. Joined by edge
                # Check if any edge connects {u,v} and {u2,v2}
                if (u, u2) in adj[u] or (u, v2) in adj[u] or (v, u2) in adj[v] or (v, v2) in adj[v]:
                    forbidden.add(c)
            
            # Find smallest available color
            c = 0
            while c in forbidden:
                c += 1
            color_map[e] = c
            colors = max(colors, c + 1)
        
        # Compact colors to 0..K-1
        unique_colors = sorted(set(color_map.values()))
        color_map_compact = {}
        for e, c in color_map.items():
            color_map_compact[e] = unique_colors.index(c)
        return color_map_compact, len(unique_colors)

    def local_search(edges, adj, n, max_iters=5000):
        # Start with greedy
        color_map, K = greedy_coloring(edges, adj, n)
        
        # Build conflict graph for colors?
        # Actually, let's just do random restarts and swaps.
        
        best_K = K
        best_map = color_map.copy()
        
        # Pre-compute conflict between edges to speed up moves
        # conflict[i][j] = True if edge i and edge j cannot have the same color
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        
        # This O(m^2) precomputation is acceptable for m < 300
        conflict = [[False] * m for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i+1, m):
                u2, v2 = edges[j]
                if u == u2 or u == v2 or v == u2 or v == v2:
                    conflict[i][j] = True
                    conflict[j][i] = True
                else:
                    if (u, u2) in adj[u] or (u, v2) in adj[u] or (v, u2) in adj[v] or (v, v2) in adj[v]:
                        conflict[i][j] = True
                        conflict[j][i] = True
        
        # Simulated Annealing / Hill Climbing
        # State: list of colors for each edge index
        current_colors = [color_map[e] for e in edges]
        current_K = max(current_colors) + 1
        
        # Helper to count colors used
        def count_colors(clist):
            return len(set(clist))
        
        # Helper to check validity of a specific color assignment for an edge
        # We only change one edge at a time, so we can check locally
        
        T = 1.0
        T_min = 0.001
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            i = random.randint(0, m - 1)
            old_color = current_colors[i]
            
            # Try to assign a new color
            # Strategy: Try a random color, or try to "move" to an existing color to reduce K, or to a new color if needed
            # Better: Try all valid colors? No, too slow.
            # Try: Pick a random color from 0 to current_K + 1
            
            # Determine valid colors for edge i
            valid_colors = set(range(current_K + 2))
            for j in range(m):
                if i != j and conflict[i][j]:
                    valid_colors.discard(current_colors[j])
            
            if not valid_colors:
                continue
                
            new_color = random.choice(list(valid_colors))
            
            if new_color == old_color:
                continue
                
            # Calculate change in K
            # Count how many edges have old_color
            old_count = current_colors.count(old_color)
            new_count = current_colors.count(new_color)
            
            # If old_count == 1, removing it might reduce K if old_color was the max
            # If new_count == 0, adding it might increase K if new_color is max+1
            
            # Let's just compute the new K estimate
            # Actually, computing exact K change is tricky if we don't update the set.
            # For simplicity and speed, let's just do a "random swap" and check validity, then accept if it doesn't increase K, or with prob T if it does.
            
            # Wait, we must ensure the new assignment is VALID.
            # By picking from valid_colors, it is valid.
            
            # Heuristic: Prefer colors that are already in use (to keep K low)
            # If new_color > current_K, it increases K.
            # If old_color is unique and old_color == current_K - 1 (or max), it might decrease K.
            
            # Let's calculate actual K change
            used_colors_before = set(current_colors)
            K_before = len(used_colors_before)
            
            current_colors[i] = new_color
            used_colors_after = set(current_colors)
            K_after = len(used_colors_after)
            
            delta_K = K_after - K_before
            
            if delta_K < 0:
                # Improvement
                pass
            elif delta_K == 0:
                # Neutral, accept (or random)
                if random.random() > 0.5:
                    pass # Accept
                else:
                    current_colors[i] = old_color # Reject
                    continue
            else:
                # Worse
                if random.random() < pow(2.71828, -delta_K / max(T, 1e-9)):
                    pass # Accept
                else:
                    current_colors[i] = old_color # Reject
                    continue
            
            # Update best
            if K_after < best_K:
                best_K = K_after
                best_map = {e: c for e, c in zip(edges, current_colors)}
                # Restart cooling? No, keep going.
            
            T *= cooling
            if T < T_min:
                T = T_min

        return best_map, best_K

    results = []
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = get_neighborhoods(edges, n)
        
        # Run local search with a few restarts if time permits
        best_map = None
        best_K = float('inf')
        
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Add some randomness to initial greedy to escape basins
            # We can shuffle the edge order for the greedy part
            shuffled_edges = edges.copy()
            random.shuffle(shuffled_edges)
            
            # Re-implement greedy with shuffled order for diversity?
            # Or just run the local search which starts from a standard greedy.
            # To get diversity, let's perturb the initial greedy.
            
            # Actually, the local_search function above uses a fixed greedy.
            # Let's modify local_search to accept an initial map.
            
            # For now, just run the standard local_search.
            # To improve, we could randomize the greedy step.
            
            map_candidate, K_candidate = local_search(edges, adj, n, max_iters=2000)
            
            if K_candidate < best_K:
                best_K = K_candidate
                best_map = map_candidate
        
        # If no result, fallback to greedy
        if best_map is None:
            best_map, best_K = greedy_coloring(edges, adj, n)
        
        # Convert map to list of colors in sorted edge order
        color_list = [best_map[e] for e in edges]
        
        # Compact colors to 0..K-1 to ensure minimal representation
        unique_colors = sorted(set(color_list))
        color_map_compact = {c: i for i, c in enumerate(unique_colors)}
        final_list = [color_map_compact[c] for c in color_list]
        
        results.append(final_list)
        
        # Check time
        if time.time() > deadline:
             break

    return results
# EVOLVE-BLOCK-END
