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

def find_hamiltonian_cycle(available_edges_set, max_restarts=50, time_limit=None):
    """
    Find a Hamiltonian cycle in the graph defined by available_edges_set.
    Uses Warnsdorff's heuristic with randomized restarts.
    Returns the cycle as a list of vertices, or None.
    """
    # Build adjacency
    adj = {i: set() for i in range(N)}
    for e in available_edges_set:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    # Check degrees: all must be >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Shuffle neighbors to randomize search
        adj_shuffled = {}
        for i in range(N):
            neigh = list(adj[i])
            random.shuffle(neigh)
            adj_shuffled[i] = neigh
        
        start = random.randint(0, N-1)
        path = [start]
        visited = {start}
        
        def dfs(current, depth):
            if time_limit is not None and time.time() - start_time > time_limit:
                return False
            
            if depth == N:
                # Check if edge back to start exists
                if start in adj_shuffled[current]:
                    return True
                return False
            
            # Get unvisited neighbors, sorted by Warnsdorff's heuristic
            # (prefer neighbors with fewer unvisited neighbors)
            unvisited_neighbors = [n for n in adj_shuffled[current] if n not in visited]
            
            if not unvisited_neighbors:
                return False
            
            # Sort by number of unvisited neighbors (Warnsdorff)
            unvisited_neighbors.sort(key=lambda n: len([m for m in adj_shuffled[n] if m not in visited]))
            
            # Randomize slightly to avoid getting stuck in local patterns
            # Pick from top 2-3 with some randomness
            if len(unvisited_neighbors) > 1 and random.random() < 0.3:
                # Occasionally pick randomly from top 3
                k = min(3, len(unvisited_neighbors))
                chosen = random.choice(unvisited_neighbors[:k])
                visited.add(chosen)
                path.append(chosen)
                if dfs(chosen, depth + 1):
                    return True
                path.pop()
                visited.remove(chosen)
            else:
                for neighbor in unvisited_neighbors:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(neighbor, depth + 1):
                        return True
                    path.pop()
                    visited.remove(neighbor)
            
            return False
        
        if dfs(start, 1):
            # Reconstruct the cycle
            return path
    
    return None

def edges_to_vertex_list(edge_set):
    """Reconstruct vertex list from edge set of a Hamiltonian cycle."""
    # Build adjacency for just this cycle
    adj = {i: [] for i in range(N)}
    for e in edge_set:
        u, v = tuple(e)
        adj[u].append(v)
        adj[v].append(u)
    
    # Each vertex should have exactly 2 neighbors in a cycle
    for i in range(N):
        if len(adj[i]) != 2:
            return None
    
    # Start at vertex 0
    path = [0]
    current = 0
    visited = {0}
    
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
    
    # Check close
    if 0 not in adj[current]:
        return None
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    best_count = 0
    best_cycles_list = []
    
    # Theoretical max: floor(72/21) = 3 Hamiltonian cycles
    max_possible = len(EDGE_SET) // N  # 3
    
    while time.time() < deadline:
        used_edges = set()
        current_cycles = []
        available = set(EDGE_SET)
        
        # Try to find as many Hamiltonian cycles as possible
        for _ in range(max_possible):
            if time.time() >= deadline:
                break
            
            # Time limit for each cycle search
            time_remaining = deadline - time.time()
            per_cycle_time = min(20.0, time_remaining / (max_possible - len(current_cycles)))
            
            h_cycle = find_hamiltonian_cycle(available, max_restarts=100, time_limit=per_cycle_time)
            
            if h_cycle is None:
                break
            
            # Convert to edge set
            cycle_edges = set()
            for i in range(N):
                u, v = h_cycle[i], h_cycle[(i+1) % N]
                cycle_edges.add(frozenset([u, v]))
            
            # Verify all edges are available
            if not cycle_edges.issubset(available):
                break
            
            current_cycles.append(cycle_edges)
            used_edges.update(cycle_edges)
            available = set(EDGE_SET - used_edges)
        
        # Update best
        if len(current_cycles) > best_count:
            best_count = len(current_cycles)
            best_cycles_list = current_cycles
            
        # If we reached the theoretical max, break
        if best_count >= max_possible:
            break
    
    # Construct return value
    final_result = []
    for e_set in best_cycles_list:
        v_list = edges_to_vertex_list(e_set)
        if v_list:
            final_result.append(v_list)
    
    # If no cycles found, return seed Hamiltonian cycle
    if not final_result:
        return [list(range(N))]
    
    return final_result

# EVOLVE-BLOCK-END
