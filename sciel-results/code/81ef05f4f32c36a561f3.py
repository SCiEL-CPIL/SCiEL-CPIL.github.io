import time
import sys
import random

N = 40
DEADLINE_SECONDS = 85

def construct():
    t0 = time.time()
    
    # 1. Initial Construction: Rim + Chords (59 edges)
    rim_edges = [(i, i + 1) for i in range(N - 1)]
    rim_edges.append((N - 1, 0))
    
    chord_edges = [(0, 2 * k) for k in range(1, N // 2)]
    
    edges = rim_edges + chord_edges
    
    # De-duplicate and normalize
    seen = set()
    unique_edges = []
    for a, b in edges:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            unique_edges.append((a, b))
    edges = unique_edges
    
    # Build adjacency list for fast lookup
    adj = {i: set() for i in range(N)}
    edge_set = set()
    for a, b in edges:
        u, v = min(a, b), max(a, b)
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    # 2. Initial Cycle Generation (Geometric)
    def get_initial_cycles(edge_set):
        cycles = {}
        for L in range(3, N + 1):
            if L == N:
                cyc = list(range(N))
            elif L % 2 == 1:
                m = (L - 1) // 2
                cyc = list(range(0, 2 * m + 1))
            else:
                m = L // 2
                if m == 1:
                    continue
                cyc = [0] + list(range(2, 2 * m + 1))
            # Verify this cycle is valid in current edge_set
            valid = True
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                if (min(u, v), max(u, v)) not in edge_set:
                    valid = False
                    break
            if valid:
                cycles[L] = cyc
            else:
                cycles[L] = None
        return cycles

    cycles = get_initial_cycles(edge_set)
    
    # Check if all cycles were found
    if any(c is None for c in cycles.values()):
        # If initial construction failed to cover all, return it (shouldn't happen)
        return {'edges': edges, 'cycles': [cycles[L] for L in range(3, N + 1) if cycles[L] is not None] + [[0,1,2]]}

    # 3. Greedy Edge Removal
    # Sort edges to try removing chords first (less critical for connectivity than rim)
    # We must keep the graph connected enough to form cycles.
    
    # Identify edges used in cycles
    edge_usage = {e: 0 for e in edge_set}
    for L in range(3, N + 1):
        cyc = cycles[L]
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            key = (min(u, v), max(u, v))
            if key in edge_usage:
                edge_usage[key] += 1
    
    # Try to remove edges with lowest usage, or chords
    # We iterate through all edges, but prioritize chords
    candidate_edges = list(edge_set)
    # Sort: Chords first (where 0 is involved and dist > 1), then others
    def edge_sort_key(e):
        u, v = e
        if u == 0 or v == 0:
            # Chord
            return (0, abs(u-v))
        else:
            return (1, 0)
            
    candidate_edges.sort(key=edge_sort_key)
    
    def find_cycle_of_length(L, adj, edge_set, start=0, time_limit=0.5):
        """
        Finds a simple cycle of length L in the graph using DFS with pruning.
        Returns the cycle as a list of nodes, or None.
        """
        if L < 3 or L > N:
            return None
        
        t_start = time.time()
        visited = [False] * N
        path = []
        path_set = set()
        
        def dfs(u, depth):
            if time.time() - t_start > time_limit:
                return None
            
            if depth == L - 1:
                if start in adj[u]:
                    return path[:]
                return None
            
            visited[u] = True
            path.append(u)
            path_set.add(u)
            
            # Order neighbors: prefer unvisited, then those closer to start
            neighbors = sorted(adj[u], key=lambda x: (x in path_set, abs(x - start)))
            
            for v in neighbors:
                if v == start:
                    continue  # Can only close at depth L-1
                if not visited[v]:
                    res = dfs(v, depth + 1)
                    if res is not None:
                        return res
            
            path.pop()
            path_set.remove(u)
            visited[u] = False
            return None
        
        # Try starting from 0 first
        res = dfs(0, 0)
        if res is not None:
            return res
        
        # Try other starting nodes
        for s in range(1, N):
            if time.time() - t_start > time_limit:
                break
            visited = [False] * N
            path = []
            path_set = set()
            
            def dfs_s(u, depth):
                if time.time() - t_start > time_limit:
                    return None
                if depth == L - 1:
                    if s in adj[u]:
                        return path[:]
                    return None
                visited[u] = True
                path.append(u)
                path_set.add(u)
                neighbors = sorted(adj[u], key=lambda x: (x in path_set, abs(x - s)))
                for v in neighbors:
                    if v == s:
                        continue
                    if not visited[v]:
                        res = dfs_s(v, depth + 1)
                        if res is not None:
                            return res
                path.pop()
                path_set.remove(u)
                visited[u] = False
                return None
            
            res = dfs_s(s, 0)
            if res is not None:
                return res
                
        return None

    for e in candidate_edges:
        if time.time() - t0 > DEADLINE_SECONDS:
            break
            
        u, v = e
        if e not in edge_set:
            continue
            
        # Check if removing e breaks any cycle
        critical_lengths = []
        for L in range(3, N + 1):
            cyc = cycles[L]
            if cyc is None:
                continue
            # Check if e is in this cycle
            for i in range(len(cyc)):
                c_u = cyc[i]
                c_v = cyc[(i + 1) % len(cyc)]
                if (min(c_u, c_v), max(c_u, c_v)) == e:
                    critical_lengths.append(L)
                    break
        
        if not critical_lengths:
            # Edge not used in any current cycle - safe to remove
            edge_set.remove(e)
            adj[u].remove(v)
            adj[v].remove(u)
            continue
            
        # Edge is used in critical_lengths. Try to find replacement cycles.
        edge_set.remove(e)
        adj[u].remove(v)
        adj[v].remove(u)
        
        all_valid = True
        new_cycles = {}
        
        for L in range(3, N + 1):
            if L in critical_lengths:
                new_cyc = find_cycle_of_length(L, adj, edge_set)
                if new_cyc is None:
                    all_valid = False
                    break
                new_cycles[L] = new_cyc
            else:
                new_cycles[L] = cycles[L]
        
        if all_valid:
            cycles = new_cycles
        else:
            edge_set.add(e)
            adj[u].add(v)
            adj[v].add(u)
            
    # Final Validation
    final_edges = list(edge_set)
    final_cycles = [cycles[L] for L in range(3, N + 1)]
    
    # Verify final result
    ok = True
    for L, cyc in zip(range(3, N + 1), final_cycles):
        if not cyc or len(cyc) != L:
            ok = False
            break
        if len(set(cyc)) != L:
            ok = False
            break
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                ok = False
                break
        if not ok:
            break
            
    if not ok:
        # Fallback to initial 59-edge construction if search broke it
        edges = rim_edges + chord_edges
        seen = set()
        unique_edges = []
        for a, b in edges:
            if a == b: continue
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                unique_edges.append((a, b))
        edges = unique_edges
        cycles = get_initial_cycles(set((min(a,b),max(a,b)) for a,b in edges))
        final_cycles = [cycles[L] for L in range(3, N + 1)]
        final_edges = edges

    return {'edges': final_edges, 'cycles': final_cycles}
