"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    adj = {i: set() for i in range(P['n'])}
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_edge_disjoint_cycles():
    """Find up to 10 edge-disjoint cycles using greedy + local search."""
    adj = _build_adjacency()
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    remaining_edges = set(all_edges)
    cycles = []
    deadline = time.time() + 50  # safety margin
    
    def find_cycle(used_edges, adj):
        """Find a cycle in the graph avoiding used_edges."""
        # Try to find a Hamiltonian cycle first, then shorter cycles
        # Use DFS with backtracking for small graphs
        n = P['n']
        
        def dfs_cycle(start, path, visited, current_used):
            if time.time() > deadline:
                return None
            if len(path) >= 3 and start in adj[path[-1]] and _edge_key(start, path[-1]) not in current_used:
                # Check if edge start->path[-1] is available
                if len(path) == n or (len(path) >= 3 and _edge_key(start, path[-1]) not in current_used):
                    # Verify all edges in path are available
                    valid = True
                    for i in range(len(path)):
                        u, v = path[i], path[(i+1) % len(path)]
                        if _edge_key(u, v) in current_used:
                            valid = False
                            break
                    if valid and _edge_key(start, path[-1]) not in current_used:
                        return list(path)
                    return None
            
            for nxt in sorted(adj[path[-1]]):
                if nxt == start and len(path) >= 3:
                    if _edge_key(start, path[-1]) not in current_used:
                        # Check all edges
                        valid = True
                        for i in range(len(path)):
                            u, v = path[i], path[(i+1) % len(path)]
                            if _edge_key(u, v) in current_used:
                                valid = False
                                break
                        if valid:
                            return list(path)
                    continue
                if nxt in visited:
                    continue
                if _edge_key(path[-1], nxt) in current_used:
                    continue
                visited.add(nxt)
                path.append(nxt)
                result = dfs_cycle(start, path, visited, current_used)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nxt)
            return None
        
        # Try starting from each vertex
        for start in range(n):
            visited = {start}
            path = [start]
            result = dfs_cycle(start, path, visited, used_edges)
            if result is not None:
                return result
        return None
    
    # Greedy: repeatedly find a cycle, remove its edges
    max_cycles = 10
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        cycle = find_cycle(set(), adj)
        if cycle is None:
            break
        # Verify cycle
        valid = True
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in remaining_edges:
                valid = False
                break
        if not valid:
            continue
        cycles.append(cycle)
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            remaining_edges.discard(_edge_key(u, v))
    
    return cycles

def _try_multiple_strategies():
    """Try different starting orders and strategies."""
    adj = _build_adjacency()
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    best_cycles = []
    deadline = time.time() + 45
    
    # Strategy 1: Greedy with fixed vertex order
    remaining = set(all_edges)
    cycles = []
    for _ in range(10):
        if time.time() > deadline:
            break
        # Find longest cycle possible
        n = P['n']
        found = None
        for start in range(n):
            if time.time() > deadline:
                break
            # DFS for cycle
            stack = [(start, [start], {start}, set())]
            while stack:
                if time.time() > deadline:
                    break
                cur, path, visited, used = stack.pop()
                # Try to close cycle
                if len(path) >= 3 and start in adj[cur]:
                    ek_close = _edge_key(start, cur)
                    if ek_close not in used and ek_close in remaining:
                        # Check all path edges
                        ok = True
                        for i in range(len(path)-1):
                            ek = _edge_key(path[i], path[i+1])
                            if ek in used or ek not in remaining:
                                ok = False
                                break
                        if ok:
                            if found is None or len(path) > len(found):
                                found = list(path)
                            break
                for nxt in sorted(adj[cur], key=lambda x: -len(adj[x])):
                    if nxt in visited:
                        continue
                    ek = _edge_key(cur, nxt)
                    if ek in used or ek not in remaining:
                        continue
                    new_used = used | {ek}
                    stack.append((nxt, path + [nxt], visited | {nxt}, new_used))
        
        if found is None:
            break
        cycles.append(found)
        for i in range(len(found)):
            u, v = found[i], found[(i+1) % len(found)]
            remaining.discard(_edge_key(u, v))
    
    if len(cycles) > len(best_cycles):
        best_cycles = cycles
    
    return best_cycles

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    # Try the strategy
    cycles = _try_multiple_strategies()
    
    # Validate
    if not cycles:
        # Fallback: return a single valid cycle if we can find one
        adj = _build_adjacency()
        # Find any 3-cycle
        for i in range(P['n']):
            for j in adj[i]:
                if j <= i:
                    continue
                for k in adj[j]:
                    if k <= j:
                        continue
                    if k in adj[i]:
                        return [[i, j, k]]
        return []
    
    # Verify all constraints
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    used_edges = set()
    vertex_set = None
    valid = True
    
    for cycle in cycles:
        # Check distinct vertices
        if len(set(cycle)) != len(cycle):
            valid = False
            break
        if len(cycle) < 3:
            valid = False
            break
        # Check all edges exist
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in all_edges:
                valid = False
                break
            if ek in used_edges:
                valid = False
                break
            used_edges.add(ek)
        if not valid:
            break
        # Check same vertex set
        cs = set(cycle)
        if vertex_set is None:
            vertex_set = cs
        elif cs != vertex_set:
            valid = False
            break
    
    if not valid or len(cycles) == 0:
        # Fallback to single Hamiltonian cycle attempt
        adj = _build_adjacency()
        n = P['n']
        # Simple DFS for Hamiltonian cycle
        def ham_dfs(path, visited):
            if time.time() > deadline:
                return None
            if len(path) == n:
                if path[0] in adj[path[-1]]:
                    return list(path)
                return None
            for nxt in adj[path[-1]]:
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    result = ham_dfs(path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(nxt)
            return None
        
        hc = ham_dfs([0], {0})
        if hc is not None:
            return [hc]
        return cycles if cycles else []
    
    return cycles

# EVOLVE-BLOCK-END
