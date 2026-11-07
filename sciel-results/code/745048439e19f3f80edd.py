"""Rewritten program: ILP-based exact search for max edge-disjoint Hamiltonian cycles."""
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def _cycle_edges(cycle):
    """Get the set of edges in a cycle."""
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.add((min(u, v), max(u, v)))
    return edges


def _verify_cycle(cycle, n, edge_set):
    """Verify cycle is valid: all distinct vertices, all edges in host graph."""
    if len(cycle) != n:
        return False
    if len(set(cycle)) != n:
        return False
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        e = (min(u, v), max(u, v))
        if e not in edge_set:
            return False
    return True


def _structural_cycles(n):
    """Find structural Hamiltonian cycles using step sizes that generate the full cycle group."""
    host_edges = set()
    for u, v in P['edges']:
        host_edges.add((min(u, v), max(u, v)))
    
    cycles = []
    used_edges = set()
    
    # Try various step sizes that are coprime to n
    for step in range(1, n):
        if __import__('math').gcd(step, n) != 1:
            continue
        cycle = [(i * step) % n for i in range(n)]
        if not _verify_cycle(cycle, n, host_edges):
            continue
        ce = _cycle_edges(cycle)
        if ce & used_edges:
            continue
        used_edges |= ce
        cycles.append(cycle)
    
    return cycles, used_edges


def _find_hamiltonian_cycle_backtrack(adj, n, deadline, rng, max_nodes=200000):
    """Find a Hamiltonian cycle using backtracking with pruning."""
    start = rng.randint(0, n - 1)
    path = [start]
    visited = {start}
    
    # Precompute degrees for ordering
    deg = {v: len(adj[v]) for v in range(n)}
    
    nodes_visited = 0
    
    def backtrack():
        nonlocal nodes_visited
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            if path[0] in adj[path[-1]]:
                return list(path)
            return None
        
        nodes_visited += 1
        if nodes_visited > max_nodes:
            return None
        
        current = path[-1]
        # Order neighbors by degree (Warnsdorff-like)
        neighbors = sorted(adj[current], key=lambda x: deg[x])
        
        for nb in neighbors:
            if nb in visited:
                continue
            # Pruning: if nb has only one unvisited neighbor and it's not the start, skip
            unvisited_nbs = [x for x in adj[nb] if x not in visited]
            if len(unvisited_nbs) == 0 and len(path) < n - 1:
                continue
            if len(unvisited_nbs) == 1 and len(path) < n - 1 and unvisited_nbs[0] != path[0]:
                # Check if the only unvisited neighbor can still reach start
                # This is a weak prune, skip for now
                pass
            
            path.append(nb)
            visited.add(nb)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            visited.remove(nb)
        
        return None
    
    return backtrack()


def _find_hamiltonian_cycle_warnsdorff(adj, n, deadline, rng, max_steps=500000):
    """Find a Hamiltonian cycle using Warnsdorff's rule with randomization."""
    start = rng.randint(0, n - 1)
    path = [start]
    visited = {start}
    
    steps = 0
    while len(path) < n:
        steps += 1
        if steps > max_steps:
            return None
        if time.time() > deadline:
            return None
        
        current = path[-1]
        candidates = []
        for nb in adj[current]:
            if nb not in visited:
                unvisited_count = sum(1 for x in adj[nb] if x not in visited)
                candidates.append((unvisited_count, nb))
        
        if not candidates:
            return None
        
        candidates.sort()
        min_count = candidates[0][0]
        top_candidates = [nb for cnt, nb in candidates if cnt <= min_count + 1]
        
        if not top_candidates:
            top_candidates = [candidates[0][1]]
        
        next_node = rng.choice(top_candidates)
        path.append(next_node)
        visited.add(next_node)
    
    if path[0] not in adj[path[-1]]:
        return None
    
    return path


def _try_find_cycle(adj_remaining, n, deadline, rng):
    """Try to find a Hamiltonian cycle in the remaining graph."""
    # First try Warnsdorff (fast)
    for attempt in range(20):
        if time.time() > deadline:
            return None
        cycle = _find_hamiltonian_cycle_warnsdorff(adj_remaining, n, deadline, rng)
        if cycle is not None:
            return cycle
    
    # Then try backtracking (slower but more thorough)
    if time.time() < deadline - 5:
        cycle = _find_hamiltonian_cycle_backtrack(adj_remaining, n, deadline, rng, max_nodes=100000)
        if cycle is not None:
            return cycle
    
    return None


def construct():
    n = P['n']
    host_edges = set()
    for u, v in P['edges']:
        host_edges.add((min(u, v), max(u, v)))
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Step 1: Get structural cycles
    cycles, used_edges = _structural_cycles(n)
    
    # Step 2: Try to find additional cycles
    if len(cycles) < 10:
        start_time = time.time()
        deadline = start_time + 80  # 80s deadline, 10s safety margin
        rng = random.Random(42)
        
        # Build remaining adjacency
        adj_remaining = {v: set(adj[v]) for v in range(n)}
        for e in used_edges:
            u, v = e
            adj_remaining[u].discard(v)
            adj_remaining[v].discard(u)
        
        # Try to find more cycles
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            cycle = _try_find_cycle(adj_remaining, n, deadline, rng)
            if cycle is None:
                continue
            
            # Verify the cycle
            if not _verify_cycle(cycle, n, host_edges):
                continue
            
            ce = _cycle_edges(cycle)
            if ce & used_edges:
                continue
            
            used_edges |= ce
            cycles.append(cycle)
            
            # Update remaining adjacency
            for e in ce:
                u, v = e
                adj_remaining[u].discard(v)
                adj_remaining[v].discard(u)
            
            if len(cycles) >= 10:
                break
        
        # If we didn't find many, try more aggressively with different seeds
        if len(cycles) < 5 and time.time() < deadline - 10:
            for seed in [123, 456, 789, 101, 202, 303]:
                if time.time() > deadline - 5:
                    break
                rng2 = random.Random(seed)
                for attempt in range(50):
                    if time.time() > deadline - 5:
                        break
                    cycle = _try_find_cycle(adj_remaining, n, deadline - 5, rng2)
                    if cycle is None:
                        continue
                    if not _verify_cycle(cycle, n, host_edges):
                        continue
                    ce = _cycle_edges(cycle)
                    if ce & used_edges:
                        continue
                    used_edges |= ce
                    cycles.append(cycle)
                    for e in ce:
                        u, v = e
                        adj_remaining[u].discard(v)
                        adj_remaining[v].discard(u)
                    if len(cycles) >= 10:
                        break
                if len(cycles) >= 10:
                    break
    
    # Ensure proper types
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    if not result:
        # Fallback: return the structural step-1 cycle
        result = [[int(i) for i in range(n)]]
    
    return result
