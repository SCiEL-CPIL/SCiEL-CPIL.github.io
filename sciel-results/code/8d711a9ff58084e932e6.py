"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edge_list):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_ham_cycle_bt(adj, start, deadline, rng):
    """Backtracking with forward checking to find Hamiltonian cycle."""
    n = P['n']
    deg = [len(adj[i]) for i in range(n)]
    
    def backtrack(path, visited, current):
        if time.time() > deadline:
            return None
        if len(path) == n:
            if start in adj[current]:
                return path[:]
            return None
        
        neighbors = [v for v in adj[current] if v not in visited]
        # Sort by degree ascending (most constrained first)
        neighbors.sort(key=lambda v: deg[v])
        
        for v in neighbors:
            # Forward check: v must have enough unvisited neighbors
            unvisited_count = n - len(visited)
            if unvisited_count > 1:
                # v needs at least 1 unvisited neighbor to continue path
                v_unvisited = [x for x in adj[v] if x not in visited]
                if len(v_unvisited) < 1:
                    continue
                # Also check: all other unvisited vertices must still be reachable
                # (simplified: check min degree in remaining)
            
            visited.add(v)
            path.append(v)
            result = backtrack(path, visited, v)
            if result is not None:
                return result
            path.pop()
            visited.remove(v)
        
        return None
    
    visited = {start}
    path = [start]
    return backtrack(path, visited, start)

def _find_ham_cycle_sa(adj, start, deadline, rng):
    """Simulated annealing to find Hamiltonian cycle."""
    n = P['n']
    vertices = list(range(n))
    
    # Initialize: start with a random permutation starting with 'start'
    rest = [v for v in vertices if v != start]
    rng.shuffle(rest)
    path = [start] + rest
    
    def is_valid_cycle(p):
        for i in range(len(p)):
            u = p[i]
            v = p[(i + 1) % len(p)]
            if v not in adj[u]:
                return False
        return True
    
    def count_invalid_edges(p):
        count = 0
        for i in range(len(p)):
            u = p[i]
            v = p[(i + 1) % len(p)]
            if v not in adj[u]:
                count += 1
        return count
    
    def two_opt_swap(p, i, j):
        """Reverse segment i..j in path (keeping start fixed)."""
        # We need to be careful: start is at position 0
        if i == 0:
            i = 1
        if j == 0:
            j = 1
        if i > j:
            i, j = j, i
        new_p = p[:i] + p[i:j+1][::-1] + p[j+1:]
        return new_p
    
    current = path[:]
    current_cost = count_invalid_edges(current)
    best = current[:]
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.999
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if current_cost == 0:
            break
        
        # Pick two positions to reverse
        i = rng.randint(1, n - 1)  # Don't move start
        j = rng.randint(1, n - 1)
        if i > j:
            i, j = j, i
        
        new_p = two_opt_swap(current, i, j)
        new_cost = count_invalid_edges(new_p)
        delta = new_cost - current_cost
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            current = new_p
            current_cost = new_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
        
        T *= cooling
    
    if best_cost == 0:
        return best
    return None

def construct():
    deadline = time.time() + 55
    rng = random.Random(12345)
    
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set()
    for u, v in base_edges:
        base_edge_set.add(_edge_key(u, v))
    
    best_cycles = []
    
    # Strategy: try to find up to 3 edge-disjoint Hamiltonian cycles
    # Use multiple restarts with different seeds and start vertices
    
    max_outer = 50
    for outer in range(max_outer):
        if time.time() > deadline:
            break
        if len(best_cycles) >= 3:
            break
        
        # For each attempt, try to build up to 3 cycles
        remaining = set(base_edge_set)
        cycles = []
        
        # Try different orderings of start vertices
        starts = list(range(n))
        rng.shuffle(starts)
        
        success = True
        for cycle_idx in range(3):
            if time.time() > deadline:
                success = False
                break
            
            edge_list = list(remaining)
            adj = _build_adj(edge_list)
            
            # Check if graph is connected and all degrees >= 2
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg < 2:
                success = False
                break
            
            # Try multiple start vertices
            cycle_found = None
            for start in starts:
                if time.time() > deadline:
                    break
                if len(adj[start]) < 2:
                    continue
                
                # Try backtracking first
                cycle_found = _find_ham_cycle_bt(adj, start, deadline, rng)
                if cycle_found is not None:
                    break
                
                # Try simulated annealing
                cycle_found = _find_ham_cycle_sa(adj, start, deadline, rng)
                if cycle_found is not None:
                    break
            
            if cycle_found is None:
                success = False
                break
            
            # Verify and extract edges
            edges_used = set()
            valid = True
            for i in range(len(cycle_found)):
                u = cycle_found[i]
                v = cycle_found[(i + 1) % len(cycle_found)]
                key = _edge_key(u, v)
                if key not in remaining:
                    valid = False
                    break
                edges_used.add(key)
            
            if not valid:
                success = False
                break
            
            cycles.append(cycle_found)
            remaining -= edges_used
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles[:]
            if len(best_cycles) >= 3:
                break
    
    # Fallback
    if not best_cycles:
        adj = _build_adj(base_edges)
        cycle = _find_ham_cycle_bt(adj, 0, deadline, rng)
        if cycle is None:
            cycle = _find_ham_cycle_sa(adj, 0, deadline, rng)
        if cycle is not None:
            best_cycles = [cycle]
        else:
            best_cycles = [list(range(n))]
    
    result = [[int(v) for v in cycle] for cycle in best_cycles]
    return result
# EVOLVE-BLOCK-END
