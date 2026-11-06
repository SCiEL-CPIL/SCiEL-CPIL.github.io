import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def generate_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Try to generate a graph with max degree d
        # The seed program uses a specific generation method:
        # for _ in range(4 * n * d):
        #   u, v = rng.randrange(n), rng.randrange(n)
        #   e = (min(u, v), max(u, v))
        #   if u != v and degree[u] < d and degree[v] < d and e not in edges:
        #       edges.add(e)
        #       degree[u] += 1
        #       degree[v] += 1
        # We must replicate this exactly to get the same graphs.
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_strong_conflict_graph(edges, n):
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        m = len(edges)
        
        # Build adjacency list for original graph for quick lookup
        orig_adj = defaultdict(set)
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)
            
        # Conflict graph adjacency
        # H has m vertices. H[i] is set of j such that edges[i] and edges[j] conflict.
        H = [set() for _ in range(m)]
        
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check if edges[i] and edges[j] conflict
                # 1. Share a vertex
                if u == x or u == y or v == x or v == y:
                    H[i].add(j)
                    H[j].add(i)
                else:
                    # 2. Disjoint but connected by an edge
                    # Check if any endpoint of (u,v) is adjacent to any endpoint of (x,y)
                    if (u in orig_adj[x] or u in orig_adj[y] or 
                        v in orig_adj[x] or v in orig_adj[y]):
                        H[i].add(j)
                        H[j].add(i)
                        
        return H

    def dsatur_coloring(H, m):
        """
        DSATUR heuristic for graph coloring.
        H: list of sets, H[i] is neighbors of i.
        m: number of vertices.
        Returns: list of colors for each vertex.
        """
        colors = [-1] * m
        sat = [set() for _ in range(m)]
        degrees = [len(H[i]) for i in range(m)]
        uncolored = set(range(m))
        
        # We will pick vertices one by one
        for _ in range(m):
            # Find vertex with max saturation, then max degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = len(sat[v])
                d = degrees[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_sat = s
                    best_deg = d
                    best_v = v
            
            if best_v == -1:
                break
                
            # Find smallest available color
            avail = [c for c in range(m) if c not in sat[best_v]]
            # To minimize colors, we prefer smaller indices
            # But we should also consider if using a new color is unavoidable
            # Standard DSATUR picks the smallest available
            color = min(avail)
            
            colors[best_v] = color
            uncolored.remove(best_v)
            
            # Update saturation of neighbors
            for u in H[best_v]:
                if u in uncolored:
                    sat[u].add(color)
                    
        return colors

    def local_search_optimize(colors, H, m):
        """
        Try to reduce the number of colors by re-coloring vertices to smaller indices.
        """
        changed = True
        while changed:
            changed = False
            # Sort vertices by color descending to try to drop high colors
            vertices_by_color = sorted(range(m), key=lambda x: colors[x], reverse=True)
            for v in vertices_by_color:
                current_color = colors[v]
                # Try all colors smaller than current
                for c in range(current_color):
                    if c not in H[v] and all(colors[u] != c for u in H[v]):
                        # Can we recolor v to c?
                        # Check if any neighbor has color c
                        if not any(colors[u] == c for u in H[v]):
                            colors[v] = c
                            changed = True
                            break
        # Return the max color + 1 (or 0 if no edges)
        if m == 0:
            return 0
        return max(colors) + 1

    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    for n, d, seed in specs:
        if time.time() > deadline:
            # Fallback to a very simple greedy if we run out of time
            # But we should have enough time for these small graphs
            pass
            
        edges = generate_instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            results.append([])
            continue
            
        H = build_strong_conflict_graph(edges, n)
        
        # Run DSATUR
        colors = dsatur_coloring(H, m)
        
        # Local search to reduce colors
        # Note: local search as defined above might be slow if not careful, 
        # but m is small (<= 80*6/2 approx 120 edges? No, max degree d, n nodes, max edges n*d/2.
        # 40*4/2=80, 60*5/2=150, 80*6/2=240.
        # So m is at most 240. DSATUR is O(m^2). Local search O(m^3) worst case, but usually fast.
        
        num_colors = local_search_optimize(colors, H, m)
        
        # The output requires a list of colors for each edge in sorted order.
        # colors[i] is the color of edge i.
        results.append(colors)
        
    return results

# EVOLVE-BLOCK-END
