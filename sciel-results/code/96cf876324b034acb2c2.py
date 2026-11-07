"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
EDGES = P['edges']
EDGE_SET = set(frozenset(e) for e in EDGES)
ADJ = {i: set() for i in range(N)}
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def find_hamiltonian_cycle(available_edges_set, start_node=0, max_attempts=50):
        """
        Find a Hamiltonian cycle in the graph defined by available_edges_set.
        Returns the cycle as a list of vertices, or None.
        Uses Warnsdorff's heuristic with backtracking.
        """
        adj = {i: set() for i in range(N)}
        for e in available_edges_set:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
            
        for i in range(N):
            if len(adj[i]) < 2:
                return None
        
        start = start_node
        path = [start]
        visited = {start}
        
        def warnsdorff_order(current):
            def score(neighbor):
                count = 0
                for nb in adj[neighbor]:
                    if nb not in visited and nb != current:
                        count += 1
                return count
            candidates = [n for n in adj[current] if n not in visited]
            if not candidates:
                return []
            candidates.sort(key=lambda x: (score(x), x))
            return candidates
        
        def dfs(current, depth):
            if len(path) == N:
                if start in adj[current]:
                    return True
                return False
            
            neighbors = warnsdorff_order(current)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(neighbor, depth + 1):
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False
        
        for attempt in range(max_attempts):
            random.seed(attempt * 7919 + int(start_time * 1000) % 1000)
            path = [start]
            visited = {start}
            
            if dfs(start, 1):
                cycle_edges = set()
                for i in range(N):
                    u, v = path[i], path[(i+1)%N]
                    cycle_edges.add(frozenset([u,v]))
                return path
            
        return None

    def edges_to_vertex_list(edge_set):
        adj = {i: [] for i in range(N)}
        for e in edge_set:
            u, v = tuple(e)
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(N):
            if len(adj[i]) != 2:
                return None
        
        start = 0
        path = [start]
        current = start
        visited = {start}
        
        for _ in range(N-1):
            next_v = None
            for n in adj[current]:
                if n not in visited:
                    next_v = n
                    break
            if next_v is None:
                return None
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        
        if start not in adj[current]:
            return None
        return path

    best_count = 0
    best_cycles_list = []
    
    max_attempts = 30
    attempt = 0
    while time.time() < deadline and attempt < max_attempts:
        attempt += 1
        used_edges = set()
        current_cycles = []
        available = set(EDGE_SET)
        
        while time.time() < deadline:
            h_cycle = find_hamiltonian_cycle(available, start_node=attempt % N)
            
            if h_cycle is None:
                break
            
            cycle_edges = set()
            for i in range(N):
                u, v = h_cycle[i], h_cycle[(i+1)%N]
                cycle_edges.add(frozenset([u,v]))
            
            current_cycles.append(cycle_edges)
            used_edges.update(cycle_edges)
            available = set(EDGE_SET - used_edges)
            
            if not available:
                break
        
        if len(current_cycles) > best_count:
            best_count = len(current_cycles)
            best_cycles_list = current_cycles
            
            if best_count >= 3:
                break
    
    final_result = []
    for e_set in best_cycles_list:
        v_list = edges_to_vertex_list(e_set)
        if v_list:
            final_result.append(v_list)
    
    if not final_result:
        return [list(range(N))]
    
    return final_result

# EVOLVE-BLOCK-END
