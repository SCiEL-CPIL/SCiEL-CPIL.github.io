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

def _find_ham_cycle_fast(adj, start, deadline, rng):
    """Fast Hamiltonian cycle finder using Warnsdorff's rule with backtracking."""
    n = P['n']
    
    def backtrack(path, visited, current, depth):
        if time.time() > deadline:
            return None
        if depth == n:
            if start in adj[current]:
                return path[:]
            return None
        
        neighbors = [v for v in adj[current] if v not in visited]
        if not neighbors:
            return None
        
        # Warnsdorff's rule: prioritize vertices with fewer unvisited neighbors
        neighbors.sort(key=lambda v: sum(1 for x in adj[v] if x not in visited))
        
        for v in neighbors:
            # Pruning: check if v can still reach all remaining vertices
            if depth < n - 1:
                unvisited = set(range(n)) - visited - {v}
                # Quick check: v must have at least one unvisited neighbor
                if not any(x in unvisited for x in adj[v]):
                    continue
            
            visited.add(v)
            path.append(v)
            result = backtrack(path, visited, v, depth + 1)
            if result is not None:
                return result
            path.pop()
            visited.remove(v)
        
        return None
    
    visited = {start}
    path = [start]
    return backtrack(path, visited, start, 1)

def _find_ham_cycle_sa(adj, start, deadline, rng):
    """Simulated annealing to find Hamiltonian cycle - optimized version."""
    n = P['n']
    vertices = list(range(n))
    
    rest = [v for v in vertices if v != start]
    rng.shuffle(rest)
    path = [start] + rest
    
    def count_invalid_edges(p):
        count = 0
        for i in range(len(p)):
            u = p[i]
            v = p[(i + 1) % len(p)]
            if v not in adj[u]:
                count += 1
        return count
    
    def two_opt_swap(p, i, j):
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
    cooling = 0.9998
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if current_cost == 0:
            break
        
        i = rng.randint(1, n - 1)
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

def _find_cycle_greedy(adj, deadline, rng):
    """Greedy approach to find a Hamiltonian cycle."""
    n = P['n']
    vertices = list(range(n))
    rng.shuffle(vertices)
    start = vertices[0]
    
    path = [start]
    visited = {start}
    current = start
    
    for _ in range(n - 1):
        if time.time() > deadline:
            return None
        
        neighbors = [v for v in adj[current] if v not in visited]
        if not neighbors:
            return None
        
        # Choose neighbor with fewest unvisited neighbors (Warnsdorff)
        neighbors.sort(key=lambda v: sum(1 for x in adj[v] if x not in visited))
        next_v = neighbors[0]
        
        path.append(next_v)
        visited.add(next_v)
        current = next_v
    
    # Check if we can close the cycle
    if start in adj[current]:
        return path
    return None

def construct():
    deadline = time.time() + 55
    rng = random.Random(42)
    
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set()
    for u, v in base_edges:
        base_edge_set.add(_edge_key(u, v))
    
    best_cycles = []
    
    # Try multiple strategies to find 3 edge-disjoint Hamiltonian cycles
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        if len(best_cycles) >= 3:
            break
        
        remaining = set(base_edge_set)
        cycles = []
        
        # Shuffle start vertices for diversity
        starts = list(range(n))
        rng.shuffle(starts)
        
        success = True
        for cycle_idx in range(3):
            if time.time() > deadline:
                success = False
                break
            
            edge_list = list(remaining)
            adj = _build_adj(edge_list)
            
            # Check minimum degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg < 2:
                success = False
                break
            
            # Try multiple methods
            cycle_found = None
            
            # Method 1: Fast backtracking with multiple starts
            for start in starts[:10]:
                if time.time() > deadline:
                    break
                if len(adj[start]) < 2:
                    continue
                cycle_found = _find_ham_cycle_fast(adj, start, deadline, rng)
                if cycle_found is not None:
                    break
            
            # Method 2: Greedy
            if cycle_found is None:
                cycle_found = _find_cycle_greedy(adj, deadline, rng)
            
            # Method 3: Simulated annealing
            if cycle_found is None:
                for start in starts[:5]:
                    if time.time() > deadline:
                        break
                    cycle_found = _find_ham_cycle_sa(adj, start, deadline, rng)
                    if cycle_found is not None:
                        break
            
            if cycle_found is None:
                success = False
                break
            
            # Verify cycle
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
    
    # Fallback: ensure we return at least one valid cycle
    if not best_cycles:
        adj = _build_adj(base_edges)
        cycle = _find_ham_cycle_fast(adj, 0, deadline, rng)
        if cycle is None:
            cycle = _find_cycle_greedy(adj, deadline, rng)
        if cycle is None:
            cycle = _find_ham_cycle_sa(adj, 0, deadline, rng)
        if cycle is not None:
            best_cycles = [cycle]
        else:
            # Last resort: find any cycle of length >= 3
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        for w in adj[v]:
                            if w != u and w in adj[u]:
                                best_cycles = [[u, v, w]]
                                break
                if best_cycles:
                    break
            if not best_cycles:
                best_cycles = [[0, 1, 2]]
    
    result = [[int(v) for v in cycle] for cycle in best_cycles]
    return result
# EVOLVE-BLOCK-END
