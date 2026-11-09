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
            # (It should be for the initial set)
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
                # Fallback if initial geometry fails (should not happen)
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
    
    def find_cycle_of_length(L, adj, edge_set, start=0):
        """
        Finds a simple cycle of length L in the graph.
        Returns the cycle as a list of nodes, or None.
        """
        if L < 3 or L > N:
            return None
        
        visited = set()
        path = []
        
        def dfs(u, depth):
            if depth == L:
                # Check if we can close the loop to start
                if start in adj[u]:
                    return path[:]
                return None
            
            if depth > L:
                return None
                
            visited.add(u)
            path.append(u)
            
            # Heuristic: try neighbors in a consistent order
            # Sort neighbors to ensure deterministic behavior
            for v in sorted(adj[u]):
                if v == start and depth == L - 1:
                    # Found a cycle of length L
                    return path[:]
                if v not in visited:
                    res = dfs(v, depth + 1)
                    if res:
                        return res
            
            path.pop()
            visited.remove(u)
            return None
        
        # Try starting from 0
        res = dfs(0, 0)
        if res:
            return res
        
        # If not found starting from 0, try other starts?
        # For efficiency, we might just fail, but for correctness in removal,
        # we should be robust.
        # Given the sparse structure, 0 is central.
        # Let's try a few other nodes if 0 fails.
        for s in range(1, min(5, N)):
            if s not in adj: continue
            # Reset state
            visited = set()
            path = []
            start = s
            def dfs_start(u, depth):
                if depth == L:
                    if start in adj[u]:
                        return path[:]
                    return None
                if depth > L:
                    return None
                visited.add(u)
                path.append(u)
                for v in sorted(adj[u]):
                    if v == start and depth == L - 1:
                        return path[:]
                    if v not in visited:
                        res = dfs_start(v, depth + 1)
                        if res:
                            return res
                path.pop()
                visited.remove(u)
                return None
            res = dfs_start(s, 0)
            if res:
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
            # Edge not used in any current cycle, safe to remove?
            # But it might be needed for future re-routing.
            # For greedy, if not used, remove it.
            # However, we must ensure graph stays connected?
            # Actually, we just need cycles to exist.
            # If an edge is not in the current certificate, removing it might still allow
            # the same cycles to exist (if redundant) or break them if it was essential
            # but not explicitly listed? No, the cycle IS the list of edges.
            # If the edge is not in the list, the cycle does not use it.
            # So removing it doesn't affect the *existing* certificate.
            # But does the graph still *have* a cycle of that length?
            # Yes, because the certificate is a valid cycle in the graph.
            # If we remove an edge not in the certificate, the certificate remains valid.
            # So we can safely remove edges not used in any current cycle.
            
            # Remove edge
            edge_set.remove(e)
            adj[u].remove(v)
            adj[v].remove(u)
            continue
            
        # Edge is used in critical_lengths.
        # Try to find replacement cycles for these lengths.
        
        # Temporarily remove
        edge_set.remove(e)
        adj[u].remove(v)
        adj[v].remove(u)
        
        all_valid = True
        new_cycles = {}
        
        for L in range(3, N + 1):
            if L in critical_lengths:
                # Find new cycle
                new_cyc = find_cycle_of_length(L, adj, edge_set)
                if new_cyc is None:
                    all_valid = False
                    break
                new_cycles[L] = new_cyc
            else:
                new_cycles[L] = cycles[L]
        
        if all_valid:
            # Update cycles
            cycles = new_cycles
            # Keep edge removed
        else:
            # Restore edge
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
