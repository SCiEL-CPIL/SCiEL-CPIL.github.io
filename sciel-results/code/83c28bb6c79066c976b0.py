"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges, n):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle_fast(adj, n, deadline=None, rng=None):
    """
    Find a Hamiltonian cycle using DFS with backtracking and degree-based heuristics.
    Uses iterative deepening with multiple starting vertices and neighbor orderings.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    # Precompute sorted neighbor lists by degree (ascending) for better pruning
    def get_neighbors_ordered(node, visited):
        """Return unvisited neighbors sorted by degree in remaining graph (ascending)."""
        unvisited = [nb for nb in adj[node] if nb not in visited]
        if not unvisited:
            return []
        # Sort by number of unvisited neighbors (fewest first) - most constrained first
        unvisited.sort(key=lambda x: sum(1 for nb2 in adj[x] if nb2 not in visited))
        return unvisited
    
    # Try different starting vertices
    starts = list(range(n))
    if rng is not None:
        rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            break
        # Iterative DFS with backtracking
        path = [start]
        visited = {start}
        stack = [(start, 0)]  # (current_node, next_neighbor_index)
        
        # Use explicit stack for iterative DFS
        def dfs_iterative(start):
            if time.time() > deadline:
                return None
            
            path = [start]
            visited = {start}
            # For each node in path, track which neighbor index we're at
            neighbor_indices = [0]
            
            while path:
                current = path[-1]
                
                # Get ordered neighbors
                unvisited_neighbors = [nb for nb in adj[current] if nb not in visited]
                
                # Sort by degree heuristic
                unvisited_neighbors.sort(key=lambda x: sum(1 for nb2 in adj[x] if nb2 not in visited))
                
                found = False
                for nb in unvisited_neighbors:
                    if len(path) == n - 1:
                        # Last vertex: must connect back to start
                        if nb in adj[start]:
                            return path + [nb]
                        continue
                    
                    # Check if taking this edge would disconnect remaining vertices
                    # Simple connectivity check: after removing current and nb, 
                    # are remaining unvisited vertices still connected?
                    new_visited = visited | {nb}
                    remaining = [v for v in range(n) if v not in new_visited]
                    if remaining:
                        # Check connectivity of remaining vertices
                        # BFS from first remaining vertex
                        remaining_set = set(remaining)
                        bfs_queue = [remaining[0]]
                        bfs_visited = {remaining[0]}
                        while bfs_queue:
                            node = bfs_queue.pop()
                            for nb2 in adj[node]:
                                if nb2 in remaining_set and nb2 not in bfs_visited:
                                    bfs_visited.add(nb2)
                                    bfs_queue.append(nb2)
                        if len(bfs_visited) != len(remaining):
                            continue  # Would disconnect, skip this neighbor
                    
                    path.append(nb)
                    visited.add(nb)
                    found = True
                    break
                
                if not found:
                    # Backtrack
                    path.pop()
                    visited.discard(current)
                    if not path:
                        break
            
            return None
        
        result = dfs_iterative(start)
        if result is not None:
            return result
    
    return None

def _find_hamiltonian_cycle_random(adj, n, deadline=None, rng=None):
    """Find a Hamiltonian cycle with randomized neighbor ordering and restarts."""
    if rng is None:
        rng = random.Random()
    
    starts = list(range(n))
    for start in starts:
        if time.time() > deadline:
            break
        
        rng.shuffle(starts)
        path = [start]
        visited = {start}
        
        def dfs(path, visited, current):
            if time.time() > deadline:
                return None
            if len(path) == n:
                if current in adj[start]:
                    return path
                return None
            
            neighbors = list(adj[current])
            neighbors = [nb for nb in neighbors if nb not in visited]
            rng.shuffle(neighbors)
            
            for nb in neighbors:
                if len(path) == n - 1:
                    if nb in adj[start]:
                        return path + [nb]
                    continue
                
                result = dfs(path + [nb], visited | {nb}, nb)
                if result is not None:
                    return result
            
            return None
        
        result = dfs(path, visited, start)
        if result is not None:
            return result
    
    return None

def _remove_cycle_edges(edges, cycle):
    """Remove edges used by the cycle from the edge list."""
    cycle_edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        cycle_edges.add((min(u, v), max(u, v)))
    
    remaining = []
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key not in cycle_edges:
            remaining.append([u, v])
    return remaining

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    n = P['n']
    edges = [list(e) for e in P['edges']]
    
    # First cycle: the natural ordering 0,1,2,...,20,0
    first_cycle = list(range(n))
    adj0 = _build_adjacency(edges, n)
    valid = True
    for i in range(n):
        u = first_cycle[i]
        v = first_cycle[(i + 1) % n]
        if v not in adj0.get(u, set()):
            valid = False
            break
    if not valid:
        first_cycle = _find_hamiltonian_cycle_fast(adj0, n, deadline)
        if first_cycle is None:
            return []
    
    cycles = [first_cycle]
    remaining_edges = _remove_cycle_edges(edges, first_cycle)
    
    # Find second cycle
    adj1 = _build_adjacency(remaining_edges, n)
    second_cycle = None
    
    if time.time() < deadline:
        second_cycle = _find_hamiltonian_cycle_fast(adj1, n, deadline)
    
    if second_cycle is None and time.time() < deadline - 5:
        rng = random.Random(42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            second_cycle = _find_hamiltonian_cycle_random(adj1, n, deadline, rng)
            if second_cycle is not None:
                break
    
    if second_cycle is not None:
        cycles.append(second_cycle)
        remaining_edges = _remove_cycle_edges(remaining_edges, second_cycle)
        
        # Find third cycle
        adj2 = _build_adjacency(remaining_edges, n)
        third_cycle = None
        
        if time.time() < deadline:
            third_cycle = _find_hamiltonian_cycle_fast(adj2, n, deadline)
        
        if third_cycle is None and time.time() < deadline - 2:
            rng2 = random.Random(123)
            for attempt in range(50):
                if time.time() > deadline - 2:
                    break
                third_cycle = _find_hamiltonian_cycle_random(adj2, n, deadline, rng2)
                if third_cycle is not None:
                    break
        
        if third_cycle is not None:
            cycles.append(third_cycle)
    
    # Convert to proper types
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
