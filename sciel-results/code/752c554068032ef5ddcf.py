"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Map edges to indices
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Initialize random coloring
    # Colors are 0, 1, 2
    colors = [random.randint(0, 2) for _ in range(m)]
    
    def get_penalty(colors):
        # 1. Degree penalty: each vertex should have exactly 1 edge of each color
        deg_counts = [[0, 0, 0] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            c = colors[i]
            deg_counts[u][c] += 1
            deg_counts[v][c] += 1
            
        penalty = 0
        for v in range(n):
            for c in range(3):
                if deg_counts[v][c] != 1:
                    penalty += 10 * (deg_counts[v][c] - 1) ** 2
        
        # 2. Connectivity penalty: each color class should be a single cycle
        # Only check if degree constraint is satisfied to avoid confusion, 
        # or just count components in the subgraph formed by edges of that color.
        for c in range(3):
            # Build subgraph for color c
            adj = {v: [] for v in range(n)}
            for i, (u, v) in enumerate(edges):
                if colors[i] == c:
                    adj[u].append(v)
                    adj[v].append(u)
            
            # Count components
            visited = set()
            comps = 0
            for v in range(n):
                if v not in visited:
                    # BFS/DFS
                    stack = [v]
                    visited.add(v)
                    # Check if it's a cycle: each node in component should have degree 2
                    is_cycle = True
                    while stack:
                        curr = stack.pop()
                        if len(adj[curr]) != 2:
                            is_cycle = False
                        for nb in adj[curr]:
                            if nb not in visited:
                                visited.add(nb)
                                stack.append(nb)
                    if is_cycle:
                        comps += 1
                    else:
                        # If not a cycle, it's a path or other structure, 
                        # which is invalid. We count it as a "bad" component.
                        # Actually, if degree constraint is met, it must be cycles.
                        # If degree constraint is NOT met, this check is less meaningful.
                        # Let's just add penalty for number of components > 1
                        pass
            
            # If degree constraint is met, we want exactly 1 component.
            # If degree constraint is not met, the graph might be a collection of paths.
            # A simple proxy: count connected components in the subgraph of color c.
            # If the subgraph is a single cycle, it has 1 component.
            # If it's multiple cycles, it has >1 components.
            # If it's paths, it has components too.
            
            # Let's refine: 
            # If all vertices have degree 2 in this subgraph, it's a union of cycles.
            # We want 1 cycle. So penalty += (comps - 1) if comps > 0.
            
            # Re-calculate comps properly
            visited = set()
            comps = 0
            all_degree_2 = True
            for v in range(n):
                if len(adj[v]) != 2:
                    all_degree_2 = False
                    break
            
            if all_degree_2:
                for v in range(n):
                    if v not in visited:
                        stack = [v]
                        visited.add(v)
                        while stack:
                            curr = stack.pop()
                            for nb in adj[curr]:
                                if nb not in visited:
                                    visited.add(nb)
                                    stack.append(nb)
                        comps += 1
                if comps > 1:
                    penalty += 5 * (comps - 1)
            else:
                # If not all degree 2, it's already penalized by degree penalty.
                # But we can still check connectivity to encourage merging.
                pass

        return penalty

    # Local Search
    best_penalty = get_penalty(colors)
    best_colors = colors[:]
    current_colors = colors[:]
    current_penalty = best_penalty
    
    start_time = time.time()
    timeout = 55.0
    
    while time.time() - start_time < timeout:
        # Pick two random edges and swap their colors
        i1 = random.randint(0, m - 1)
        i2 = random.randint(0, m - 1)
        if i1 == i2:
            continue
            
        c1 = current_colors[i1]
        c2 = current_colors[i2]
        if c1 == c2:
            continue
            
        # Swap
        current_colors[i1] = c2
        current_colors[i2] = c1
        
        new_penalty = get_penalty(current_colors)
        
        if new_penalty < current_penalty:
            current_penalty = new_penalty
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_colors = current_colors[:]
                if best_penalty == 0:
                    break
        else:
            # Revert
            current_colors[i1] = c1
            current_colors[i2] = c2
            
    # Extract cycles from best_colors
    cycles = []
    for c in range(3):
        # Build adjacency for this color
        adj = {v: [] for v in range(n)}
        for i, (u, v) in enumerate(edges):
            if best_colors[i] == c:
                adj[u].append(v)
                adj[v].append(u)
        
        # Find the cycle
        # Start at 0
        if len(adj[0]) != 2:
            continue # Invalid
            
        cycle = [0]
        visited = {0}
        curr = 0
        prev = -1
        while True:
            nexts = [x for x in adj[curr] if x != prev]
            if not nexts:
                break
            nxt = nexts[0]
            if nxt == 0:
                break
            cycle.append(nxt)
            visited.add(nxt)
            prev = curr
            curr = nxt
            
        if len(cycle) == n:
            cycles.append(cycle)
            
    # If we didn't find 3 valid cycles, fallback to previous logic or return best effort
    # But the task requires valid cycles. If we fail, we might return an empty list or a single valid one?
    # The task says "Maximize number of cycles".
    # If local search fails to find 3, we can try to find them sequentially with the old method?
    # Or just return what we have.
    
    # Ensure types are int
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
        
    if not result:
        # Fallback: Use the deterministic method from the seed if local search failed completely
        # This ensures we don't return an empty list if the seed could do better.
        # However, the directive asks to build on the new approach.
        # If the new approach fails, returning an empty list is bad.
        # Let's add a fallback to the sequential method if the coloring method fails.
        
        def find_hamiltonian_cycle(available_edges, used_edges):
            adj_avail = {v: set() for v in range(n)}
            for u, v in available_edges:
                if (u, v) not in used_edges and (v, u) not in used_edges:
                    adj_avail[u].add(v)
                    adj_avail[v].add(u)
            for v in range(n):
                if len(adj_avail[v]) < 2:
                    return None
            start = 0
            def dfs(path, visited):
                if len(path) == n:
                    if start in adj_avail[path[-1]]:
                        return path
                    return None
                last = path[-1]
                neighbors = sorted(adj_avail[last], key=lambda x: len(adj_avail[x]))
                for nb in neighbors:
                    if nb == start and len(path) < n - 1:
                        continue
                    if nb not in visited:
                        result = dfs(path + [nb], visited | {nb})
                        if result is not None:
                            return result
                return None
            result = dfs([start], {start})
            if result is not None:
                return result
            return None

        cycles_fb = []
        used_edges = set()
        for _ in range(3):
            cycle = find_hamiltonian_cycle(edges, used_edges)
            if cycle is None:
                break
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add((min(u, v), max(u, v)))
            cycles_fb.append(cycle)
            
        for cycle in cycles_fb:
            result.append([int(v) for v in cycle])

    return result

# EVOLVE-BLOCK-END
