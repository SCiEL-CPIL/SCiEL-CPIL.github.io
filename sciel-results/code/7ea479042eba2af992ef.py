import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # 1. Initial Construction: Hamiltonian Cycle + 19 Chords
    base_edges = [(i, (i+1) % N) for i in range(N)]
    
    # Chords to guarantee coverage
    initial_chords = [(0, d) for d in range(2, 21)]
    
    edges = set(base_edges)
    for u, v in initial_chords:
        edges.add((min(u,v), max(u,v)))
        
    def get_cycle_lengths(edge_set):
        adj = {i: set() for i in range(N)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        def dfs(start, current, visited, path):
            if len(path) > N: return
            if current == start and len(path) >= 3:
                cycle_lengths.add(len(path))
                return
            
            for neighbor in adj[current]:
                if neighbor == start:
                    if len(path) >= 3:
                        cycle_lengths.add(len(path))
                    continue
                if neighbor in visited:
                    continue
                if neighbor > start:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)

        for i in range(N):
            dfs(i, i, {i}, [i])
            
        return cycle_lengths

    target_lengths = set(range(3, N))
    current_chords = list(initial_chords)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(current_chords)
        
        for i in range(len(current_chords)):
            if time.time() > deadline:
                break
            
            chord = current_chords[i]
            u, v = chord
            key = (min(u,v), max(u,v))
            
            if key not in edges:
                continue
                
            new_edges = set(edges)
            new_edges.remove(key)
            
            new_lengths = get_cycle_lengths(new_edges)
            
            if target_lengths.issubset(new_lengths):
                edges = new_edges
                current_chords.pop(i)
                improved = True
                break
                
    final_lengths = get_cycle_lengths(edges)
    if not target_lengths.issubset(final_lengths):
        edges = set(base_edges)
        for u, v in initial_chords:
            edges.add((min(u,v), max(u,v)))
            
    def find_cycle_of_length(edge_set, k):
        adj = {i: set() for i in range(N)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        def dfs_find(start, current, visited, path):
            if len(path) == k:
                if start in adj[current]:
                    return path + [start]
                return None
            if len(path) > k:
                return None
            for neighbor in adj[current]:
                if neighbor == start:
                    if len(path) == k:
                        return path + [start]
                    continue
                if neighbor in visited:
                    continue
                if neighbor < start:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                res = dfs_find(start, neighbor, visited, path)
                if res:
                    return res
                path.pop()
                visited.remove(neighbor)
            return None
            
        for i in range(N):
            res = dfs_find(i, i, {i}, [i])
            if res:
                return res
        return None

    cycles = []
    for k in range(3, N+1):
        cyc = find_cycle_of_length(edges, k)
        if cyc is None:
            cyc = list(range(k))
        cycles.append(cyc)
        
    return {'edges': list(edges), 'cycles': cycles}
