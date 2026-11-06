import time
import random

N = 40
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Base cycle
    edges = set()
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    # Chords from vertex 0
    for k in range(2, 21):
        edges.add((0, k))
    
    # Cycles for the base construction
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            if L <= 21:
                k = L - 1
                cycle_verts = list(range(k + 1))
            else:
                k = N - L + 1
                cycle_verts = [0] + list(range(N - 1, k - 1, -1))
        cycles.append(cycle_verts)
    
    best_edges = set(edges)
    best_cycles = cycles
    best_score = len(best_edges) - N
    
    # Try to improve by removing chords and finding alternative cycles
    # This is a local search: try removing one chord at a time and see if we can fix cycles
    
    chord_list = [(0, k) for k in range(2, 21)]
    
    # Try removing chords in order of least utility
    # For each chord, check if it's the only way to form certain cycles
    for _ in range(50):  # limited iterations
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        for chord in chord_list[:]:
            if chord not in best_edges:
                continue
            
            # Try removing this chord
            test_edges = set(best_edges)
            test_edges.remove(chord)
            
            # Check if we can still form all cycles
            valid = True
            new_cycles = []
            
            for L in range(3, N + 1):
                if L == N:
                    # Hamiltonian cycle: check if it exists in test_edges
                    ham_cycle = _find_hamiltonian_cycle(test_edges, N)
                    if ham_cycle is None:
                        valid = False
                        break
                    new_cycles.append(ham_cycle)
                else:
                    # Try to find a cycle of length L
                    cyc = _find_cycle_of_length(test_edges, N, L)
                    if cyc is None:
                        valid = False
                        break
                    new_cycles.append(cyc)
            
            if valid:
                best_edges = test_edges
                best_cycles = new_cycles
                best_score = len(best_edges) - N
                improved = True
                chord_list.remove(chord)
                break
        
        if not improved:
            break
    
    return {'edges': [tuple(sorted(e)) for e in best_edges], 'cycles': best_cycles}


def _find_hamiltonian_cycle(edge_set, n):
    """Find a Hamiltonian cycle in a graph given as a set of edges."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Use backtracking with pruning
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    def backtrack():
        if len(path) == n:
            # Check if last vertex connects to first
            if path[0] in adj[path[-1]]:
                return True
            return False
        
        last = path[-1]
        for neighbor in sorted(adj[last]):
            if not visited[neighbor]:
                # Pruning: check if unvisited vertices are connected
                if _connected_check(adj, visited, path, n, neighbor):
                    visited[neighbor] = True
                    path.append(neighbor)
                    if backtrack():
                        return True
                    path.pop()
                    visited[neighbor] = False
        return False
    
    if backtrack():
        return path
    return None


def _connected_check(adj, visited, path, n, next_vertex):
    """Check if all unvisited vertices are in the same component as next_vertex."""
    # BFS from next_vertex through unvisited vertices
    unvisited = [i for i in range(n) if not visited[i]]
    if not unvisited:
        return True
    
    if next_vertex not in unvisited:
        # next_vertex is already visited, so we need to check connectivity of unvisited
        # through the graph
        pass
    
    # Simple check: BFS from next_vertex
    seen = set()
    stack = [next_vertex]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        for w in adj[v]:
            if not visited[w] and w not in seen:
                stack.append(w)
    
    # All unvisited vertices must be reachable from next_vertex or already in path
    for u in unvisited:
        if u not in seen and u not in path:
            return False
    return True


def _find_cycle_of_length(edge_set, n, L):
    """Find a cycle of exactly length L in the graph."""
    if L == n:
        return _find_hamiltonian_cycle(edge_set, n)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Try to find a cycle of length L starting from vertex 0
    # Use backtracking
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    def backtrack(depth):
        if depth == L:
            # Check if last connects to first
            if path[0] in adj[path[-1]]:
                return True
            return False
        
        last = path[-1]
        for neighbor in sorted(adj[last]):
            if not visited[neighbor]:
                visited[neighbor] = True
                path.append(neighbor)
                if backtrack(depth + 1):
                    return True
                path.pop()
                visited[neighbor] = False
        return False
    
    if backtrack(1):
        return path
    return None
