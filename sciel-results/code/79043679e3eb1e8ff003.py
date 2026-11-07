"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, n, start=0, deadline=None):
    """Find a Hamiltonian cycle using DFS with Warnsdorff's heuristic."""
    if deadline is not None and time.time() > deadline:
        return None
    
    path = [start]
    visited = {start}
    
    def dfs(current):
        if time.time() > deadline:
            return None
        if len(path) == n:
            if current in adj.get(start, set()) or start in adj.get(current, set()):
                return list(path)
            return None
        # Warnsdorff: sort neighbors by degree of unvisited neighbors
        neighbors = list(adj[current])
        neighbors.sort(key=lambda x: len(adj[x] - visited) if x not in visited else 999)
        for nxt in neighbors:
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                result = dfs(nxt)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nxt)
        return None
    
    result = dfs(start)
    if result is not None:
        return result
    return None

def _find_ham_cycle_random(adj, n, deadline=None, rng=None):
    """Find a Hamiltonian cycle using randomized DFS."""
    if rng is None:
        rng = random.Random()
    if deadline is not None and time.time() > deadline:
        return None
    
    start = rng.randint(0, n - 1)
    path = [start]
    visited = {start}
    
    def dfs(current):
        if time.time() > deadline:
            return None
        if len(path) == n:
            if start in adj.get(current, set()):
                return list(path)
            return None
        neighbors = [x for x in adj[current] if x not in visited]
        rng.shuffle(neighbors)
        # Add some Warnsdorff bias
        neighbors.sort(key=lambda x: len(adj[x] - visited), reverse=True)
        for nxt in neighbors:
            visited.add(nxt)
            path.append(nxt)
            result = dfs(nxt)
            if result is not None:
                return result
            path.pop()
            visited.remove(nxt)
        return None
    
    result = dfs(start)
    if result is not None:
        return result
    return None

def _cycle_edges(cycle):
    """Get the set of edges in a cycle."""
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.add((min(u, v), max(u, v)))
    return edges

def _remove_edges(adj, edges_to_remove, n):
    """Remove edges from adjacency, returning new adj dict."""
    new_adj = defaultdict(set)
    for i in range(n):
        new_adj[i] = set(adj[i])
    for u, v in edges_to_remove:
        new_adj[u].discard(v)
        new_adj[v].discard(u)
    return new_adj

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    n = P['n']
    original_edges = set()
    for u, v in P['edges']:
        original_edges.add((min(u, v), max(u, v)))
    
    best_cycles = []
    best_adj = None
    best_count = 0
    
    # Try multiple strategies with different random seeds
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Reset adjacency
        adj = defaultdict(set)
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        current_adj = adj
        success = True
        
        for cycle_idx in range(10):
            if time.time() > deadline:
                success = False
                break
            
            # Try to find a Hamiltonian cycle in current_adj
            found = False
            
            # First try deterministic with Warnsdorff
            cycle = _find_ham_cycle(current_adj, n, start=0, deadline=deadline)
            if cycle is None:
                # Try with random start
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    s = rng.randint(0, n - 1)
                    cycle = _find_ham_cycle_random(current_adj, n, deadline=deadline, rng=rng)
                    if cycle is not None:
                        found = True
                        break
            
            if cycle is not None:
                found = True
            elif cycle is None:
                # More random attempts
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    cycle = _find_ham_cycle_random(current_adj, n, deadline=deadline, rng=rng)
                    if cycle is not None:
                        found = True
                        break
            
            if not found:
                break
            
            # Verify cycle
            if len(set(cycle)) != n:
                break
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                if v not in current_adj.get(u, set()):
                    break
            else:
                cycles.append(cycle)
                ce = _cycle_edges(cycle)
                current_adj = _remove_edges(current_adj, ce, n)
                continue
            break
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
            if best_count >= 3:
                break
    
    # Return best result found
    if not best_cycles:
        # Fallback: return the seed's single cycle
        return [list(range(P['n']))]
    
    # Ensure all cycles use exactly the same vertex set (all 0..n-1)
    result = []
    for c in best_cycles:
        result.append(list(c))
    
    return result
# EVOLVE-BLOCK-END
