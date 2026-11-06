import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for 150s limit
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate random bounded degree graph
        # To ensure connectivity or specific structure isn't strictly required by the prompt, 
        # we follow the seed's generation method.
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, n):
        # Map edge index to endpoints
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        num_edges = len(edges)
        
        # Adjacency list for original graph
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Conflict graph: node i conflicts with node j if edges[i] and edges[j] are strong adjacent
        # Strong adjacent: share a vertex OR are connected by an edge.
        # Let e1=(u,v), e2=(x,y).
        # Share vertex: {u,v} & {x,y} != empty
        # Connected by edge: (u,x) in E, (u,y) in E, (v,x) in E, (v,y) in E
        
        conflict_adj = [set() for _ in range(num_edges)]
        
        for i, e1 in enumerate(edges):
            u, v = e1
            # Check against all other edges? O(M^2) is fine for M~240
            for j, e2 in enumerate(edges):
                if i == j:
                    continue
                x, y = e2
                # Check share vertex
                if (u == x or u == y or v == x or v == y):
                    conflict_adj[i].add(j)
                    continue
                # Check connected by edge
                # Edge (u,x) exists?
                if (min(u,x), max(u,x)) in edge_to_idx:
                    conflict_adj[i].add(j)
                    continue
                if (min(u,y), max(u,y)) in edge_to_idx:
                    conflict_adj[i].add(j)
                    continue
                if (min(v,x), max(v,x)) in edge_to_idx:
                    conflict_adj[i].add(j)
                    continue
                if (min(v,y), max(v,y)) in edge_to_idx:
                    conflict_adj[i].add(j)
                    continue
                    
        return conflict_adj

    def dsatur_coloring(conflict_adj, time_limit):
        n = len(conflict_adj)
        if n == 0:
            return []
        
        # Precompute degrees
        degrees = [len(adj) for adj in conflict_adj]
        
        # State:
        color = [-1] * n
        saturation = [set() for _ in range(n)] # Colors of neighbors
        uncolored = set(range(n))
        best_coloring = None
        best_num_colors = n + 1
        
        # Use a heap or just scan for max saturation/degree
        # For n~240, scanning is O(n) which is fast.
        
        def backtrack(uncolored, num_colors):
            nonlocal best_coloring, best_num_colors
            
            if time.time() - start_time > time_limit:
                return False # Timeout signal
            
            if not uncolored:
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_coloring = color[:]
                return True
            
            # Select vertex with max saturation, then max degree
            max_sat = -1
            max_deg = -1
            candidate = -1
            for v in uncolored:
                sat = len(saturation[v])
                deg = degrees[v]
                if sat > max_sat or (sat == max_sat and deg > max_deg):
                    max_sat = sat
                    max_deg = deg
                    candidate = v
            
            if candidate == -1:
                return False # Should not happen if uncolored not empty
            
            # Try colors 0 to num_colors
            # Optimization: Only try colors in saturation[v] plus one new color if needed?
            # No, we must try all available colors from 0 to num_colors-1, and potentially num_colors.
            
            # To improve pruning, we can sort colors by some heuristic?
            # Usually trying 0, 1, 2... is fine.
            
            # Check if we can prune
            # Lower bound on colors needed for remaining vertices?
            # Simple bound: 1 if uncolored else 0.
            # Better: Clique number of remaining graph? Too expensive.
            
            for c in range(num_colors + 1):
                # Check if color c is available for candidate
                if c in saturation[candidate]:
                    continue
                
                # Assign
                color[candidate] = c
                new_uncolored = uncolored.copy()
                new_uncolored.remove(candidate)
                
                # Update saturation for neighbors
                # We need to revert changes, so let's collect changes
                changes = []
                valid = True
                for nb in conflict_adj[candidate]:
                    if nb in new_uncolored:
                        # Add color c to saturation[nb]
                        if c not in saturation[nb]:
                            saturation[nb].add(c)
                            changes.append((nb, c))
                
                # Recurse
                # Note: We pass num_colors, but if c == num_colors, we are using a new color
                # The recursive call will handle the new color count implicitly if we just pass num_colors
                # Wait, the number of colors used is max(color)+1.
                # In this formulation, 'num_colors' is the count of colors used so far.
                # If we assign c < num_colors, count stays same.
                # If we assign c == num_colors, count becomes num_colors + 1.
                
                # However, standard backtracking passes the current max color index or count.
                # Let's pass the current max color index used.
                # Actually, it's easier to just pass the set of used colors or the count.
                # Let's stick with 'num_colors' being the count of colors used in the current path.
                
                # If c == num_colors, the next level will have num_colors + 1 available slots?
                # No, the loop `for c in range(num_colors + 1)` allows using index `num_colors`.
                # If we use index `num_colors`, the next state has `num_colors + 1` colors used.
                
                # Let's refine the recursive call signature.
                # It should know how many colors are "active".
                
                # If we use color c:
                # If c < num_colors, next num_colors is same.
                # If c == num_colors, next num_colors is num_colors + 1.
                
                next_num_colors = num_colors + 1 if c == num_colors else num_colors
                
                if backtrack(new_uncolored, next_num_colors):
                    return True
                    
                # Revert changes
                for nb, col in changes:
                    saturation[nb].discard(col)
                color[candidate] = -1
                
            return False
        
        # Initial call
        # We can start with 1 color and expand
        backtrack(set(range(n)), 1)
        
        if best_coloring:
            return best_coloring
        else:
            # Fallback to greedy if timeout before finding any complete solution (unlikely)
            # Or just return a greedy coloring
            return greedy_coloring(conflict_adj)

    def greedy_coloring(conflict_adj):
        n = len(conflict_adj)
        color = [-1] * n
        # Order by degree
        order = sorted(range(n), key=lambda i: len(conflict_adj[i]), reverse=True)
        used_colors = set()
        for v in order:
            neighbor_colors = set()
            for nb in conflict_adj[v]:
                if color[nb] != -1:
                    neighbor_colors.add(color[nb])
            c = 0
            while c in neighbor_colors:
                c += 1
            color[v] = c
            used_colors.add(c)
        return color

    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    # Allocate time per instance
    # The last instance is largest, give it more time?
    # Or just share the time limit.
    # Since we return best found, we can just run until time limit.
    
    for n, d, seed in specs:
        edges = instance(n, d, seed)
        conflict_adj = build_conflict_graph(edges, n)
        
        # Get coloring
        coloring = dsatur_coloring(conflict_adj, time_limit)
        
        # Map conflict graph node indices back to edge colors
        # The coloring list corresponds to the sorted edges list
        # construct() expects a list of colors for each edge in sorted order
        # Our conflict_adj nodes are indexed 0..M-1 corresponding to edges[0]..edges[M-1]
        # So coloring[i] is the color for edges[i]
        
        results.append(coloring)
        
        # Check time
        if time.time() - start_time > time_limit - 5:
             # If we are running out of time, fill remaining with greedy to ensure valid output
             # Though dsatur should return a valid coloring (best found)
             pass

    return results
