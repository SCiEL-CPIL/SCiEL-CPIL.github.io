import time
import sys

def construct():
    N = 40
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin before 90s kill

    # Start with perimeter + all chords from 0 to 2..20 (fan graph)
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    for k in range(2, 21):
        edges.add((0, k))

    def build_adj(edge_set):
        adj = [set() for _ in range(N)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_cycle(L, adj):
        """Find a simple cycle of length L. Returns list of vertices or None."""
        if L == N:
            # Check perimeter is intact
            for i in range(N):
                if (i+1) % N not in adj[i]:
                    return None
            return list(range(N))
        
        if L == 3:
            # Look for any triangle
            for i in range(N):
                for j in adj[i]:
                    if j > i:
                        for k in adj[j]:
                            if k > j and k in adj[i]:
                                return [i, j, k]
            return None
        
        if L == 4:
            for i in range(N):
                for j in adj[i]:
                    if j > i:
                        for k in adj[j]:
                            if k > j and k != i:
                                for m in adj[k]:
                                    if m > k and m == i:
                                        return [i, j, k, m]
            return None

        # General DFS for cycle of length L
        # Fix starting vertex to 0 to break symmetry, and fix second vertex to smallest neighbor
        if 0 not in [v for v in range(N)]:
            return None
        
        neighbors_0 = sorted(adj[0])
        if not neighbors_0:
            return None
        
        # Try different starting edges to find a cycle
        for first in neighbors_0:
            if first > 1:  # To break some symmetry, try smaller first vertices preferentially
                continue
            
            visited = [False] * N
            visited[0] = True
            visited[first] = True
            path = [0, first]
            
            # Use iterative DFS with explicit stack to avoid recursion depth issues
            # But for L <= 40, recursion is fine
            def dfs(current, depth):
                if depth == L:
                    return 0 in adj[current]
                
                for nxt in adj[current]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        path.append(nxt)
                        if dfs(nxt, depth + 1):
                            return True
                        path.pop()
                        visited[nxt] = False
                return False
            
            if dfs(first, 2):
                return path.copy()
        
        # If we didn't find with 0 as start, try other starts
        # But for fan graph structure, 0 should work if cycle exists
        return None

    def verify_all_cycles(adj):
        """Check if cycles of all lengths 3..N exist. Return dict of cycles or None."""
        cycles = {}
        for L in range(3, N + 1):
            cyc = find_cycle(L, adj)
            if cyc is None:
                return None
            cycles[L] = cyc
        return cycles

    # Greedy removal of chords
    adj = build_adj(edges)
    
    # Try to remove chords one by one
    # Order: try removing larger chords first (they're less critical for small cycles)
    # Actually, let's try all chords and see which can be removed
    chords_to_try = list(range(20, 1, -1))
    
    for k in chords_to_try:
        if time.time() - start_time > deadline:
            break
        
        if (0, k) not in edges:
            continue
        
        # Try removing
        edges.discard((0, k))
        adj = build_adj(edges)
        
        # Check if all cycles still exist
        # For speed, only check cycles that might be affected
        # But to be safe, check all
        valid = True
        for L in range(3, N + 1):
            if not find_cycle(L, adj):
                valid = False
                break
        
        if not valid:
            edges.add((0, k))
            adj = build_adj(edges)

    # Final verification and cycle extraction
    adj = build_adj(edges)
    cycles_dict = verify_all_cycles(adj)
    
    if cycles_dict is None:
        # Fallback: restore all chords
        for k in range(2, 21):
            edges.add((0, k))
        adj = build_adj(edges)
        cycles_dict = verify_all_cycles(adj)
    
    if cycles_dict is None:
        # Emergency: just return the full fan graph
        edge_list = sorted([tuple(sorted(e)) for e in edges])
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cycles.append(list(range(N)))
            elif L <= 21:
                cycles.append(list(range(L)))
            else:
                # Use perimeter + one chord
                k = 41 - L
                cyc = [0] + list(range(39, k-1, -1))
                cycles.append(cyc)
        return {'edges': edge_list, 'cycles': cycles}

    edge_list = sorted([tuple(sorted(e)) for e in edges])
    cycles = [cycles_dict[L] for L in range(3, N + 1)]
    
    return {'edges': edge_list, 'cycles': cycles}
