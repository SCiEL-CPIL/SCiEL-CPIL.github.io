import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    N = 256
    vertices = list(range(1, N+1))
    
    # Build conflict graph: edge (a,b) if (a+b) | a*b, i.e., (a+b) | a^2
    # Use bitmasks for adjacency
    adj = [0] * (N+1)  # adj[i] is bitmask of neighbors of i
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            s = a + b
            if (a * b) % s == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    # Degree
    degree = [bin(adj[i]).count('1') for i in range(N+1)]
    
    # Greedy independent set: pick vertices, remove conflicts
    def greedy_is(vertices_list):
        # Order by degree ascending (pick low degree first)
        remaining = set(vertices_list)
        result = []
        # Use a priority: pick vertex with min degree in remaining
        while remaining:
            # Find vertex with min degree within remaining
            best = None
            best_deg = float('inf')
            for v in remaining:
                d = bin(adj[v] & 0).count('1')  # placeholder
                # count neighbors in remaining
                d = sum(1 for u in remaining if u != v and (adj[v] >> u) & 1)
                if d < best_deg:
                    best_deg = d
                    best = v
            result.append(best)
            remaining.remove(best)
            # Remove neighbors
            to_remove = set()
            for u in remaining:
                if (adj[best] >> u) & 1:
                    to_remove.add(u)
            remaining -= to_remove
        return result
    
    # Better greedy: use a fixed ordering
    def greedy_is_fast():
        # Order vertices by degree ascending
        order = sorted(range(1, N+1), key=lambda x: degree[x])
        chosen = set()
        for v in order:
            # Check if v conflicts with any chosen
            conflict = False
            mask = adj[v]
            # Check intersection with chosen
            for u in chosen:
                if (mask >> u) & 1:
                    conflict = True
                    break
            if not conflict:
                chosen.add(v)
        return chosen
    
    incumbent = greedy_is_fast()
    best_set = set(incumbent)
    best_size = len(best_set)
    
    # Branch and bound
    # Upper bound via greedy coloring
    def color_bound(vertex_set):
        # Greedy coloring of the subgraph induced by vertex_set
        # Returns number of colors (upper bound on MIS size)
        if not vertex_set:
            return 0
        vs = list(vertex_set)
        # Order by degree descending in the subgraph
        colors = {}
        color_count = 0
        # Process in order of degree descending
        vs.sort(key=lambda x: -degree[x])
        for v in vs:
            # Find smallest color not used by neighbors in same set
            used = set()
            for u in vs:
                if u != v and (adj[v] >> u) & 1 and u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            color_count = max(color_count, c + 1)
        return color_count
    
    # B&B: state = set of available vertices
    # We want max independent set in the subgraph
    
    def branch_and_bound(available, current_set, current_size):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound: current_size + color_bound(available)
        ub = current_size + color_bound(available)
        if ub <= best_size:
            return
        
        # Reduce: if a vertex has degree 0 in available, it must be in any max IS
        # (it doesn't conflict with anything in available)
        changed = True
        while changed:
            changed = False
            # Find isolated vertex
            for v in list(available):
                # Check if v has any neighbor in available
                has_neighbor = False
                for u in available:
                    if u != v and (adj[v] >> u) & 1:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    # v must be included
                    current_set.add(v)
                    available.discard(v)
                    # Remove v's neighbors (none in available)
                    changed = True
                    break
        
        if not available:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Recompute upper bound
        ub = len(current_set) + color_bound(available)
        if ub <= best_size:
            return
        
        # Branch on vertex with max degree in available
        max_deg_v = None
        max_deg = -1
        for v in available:
            d = sum(1 for u in available if u != v and (adj[v] >> u) & 1)
            if d > max_deg:
                max_deg = d
                max_deg_v = v
        
        v = max_deg_v
        
        # Branch 1: include v
        new_available = set(available)
        new_available.discard(v)
        # Remove neighbors of v
        to_remove = set()
        for u in new_available:
            if (adj[v] >> u) & 1:
                to_remove.add(u)
        new_available -= to_remove
        new_current = set(current_set)
        new_current.add(v)
        branch_and_bound(new_available, new_current, len(new_current))
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude v
        new_available2 = set(available)
        new_available2.discard(v)
        branch_and_bound(new_available2, current_set, len(current_set))
    
    # Initial: all vertices available
    all_vertices = set(range(1, N+1))
    branch_and_bound(all_vertices, set(), 0)
    
    # Ensure nonempty
    if not best_set:
        best_set = {1}
    
    return list(best_set)
