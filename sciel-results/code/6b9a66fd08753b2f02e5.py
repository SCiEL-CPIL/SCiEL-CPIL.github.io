import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle_dfs(adj, start, rng, max_nodes=200000):
    """Find Hamiltonian cycle using DFS with Warnsdorff heuristic."""
    n = P['n']
    nodes_visited = 0
    
    def backtrack(path, visited):
        nonlocal nodes_visited
        nodes_visited += 1
        if nodes_visited > max_nodes:
            return None
        
        if len(path) == n:
            if start in adj[path[-1]]:
                return path[:]
            return None
        
        current = path[-1]
        candidates = []
        for nb in adj[current]:
            if nb not in visited:
                # Warnsdorff: prefer neighbors with fewer unvisited neighbors
                onward = sum(1 for x in adj[nb] if x not in visited and x != current)
                candidates.append((onward, rng.random(), nb))
        
        if not candidates:
            return None
        
        # Sort by Warnsdorff, then random tiebreak
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        for _, _, nb in candidates:
            result = backtrack(path + [nb], visited | {nb})
            if result is not None:
                return result
        
        return None
    
    return backtrack([start], {start})

def _extract_cycle(remaining_edges, rng, max_attempts=100):
    """Try to extract one Hamiltonian cycle from remaining edges."""
    n = P['n']
    
    for attempt in range(max_attempts):
        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check min degree
        if any(len(adj[v]) < 2 for v in range(n)):
            return None
        
        valid_starts = [v for v in range(n) if len(adj[v]) >= 2]
        if not valid_starts:
            return None
        
        # Try multiple starts
        rng.shuffle(valid_starts)
        for start in valid_starts[:5]:  # Try up to 5 different starts
            cycle = _find_hamiltonian_cycle_dfs(adj, start, rng, max_nodes=100000)
            if cycle is not None:
                # Verify
                if len(set(cycle)) != n:
                    continue
                valid = True
                for i in range(n):
                    u, v = cycle[i], cycle[(i+1) % n]
                    ek = _edge_key(u, v)
                    if ek not in remaining_edges:
                        valid = False
                        break
                if valid:
                    return cycle
    
    return None

def _verify_packing(cycles, all_edges_set):
    """Verify that all cycles are valid, edge-disjoint, and use the same vertex set."""
    if not cycles:
        return False
    
    n = P['n']
    used_edges = set()
    
    for cycle in cycles:
        if not isinstance(cycle, (list, tuple)):
            return False
        if len(cycle) < 3:
            return False
        
        for v in cycle:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if v < 0 or v >= n:
                return False
        
        if len(set(cycle)) != len(cycle):
            return False
        
        if set(cycle) != set(range(n)):
            return False
        
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in all_edges_set:
                return False
            if ek in used_edges:
                return False
            used_edges.add(ek)
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    all_edges = P['edges']
    all_edges_set = set()
    for u, v in all_edges:
        all_edges_set.add(_edge_key(u, v))
    
    best_packing = []
    
    # Try multiple random seeds with increasing effort
    for seed in range(500):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        remaining = set(all_edges_set)
        cycles = []
        
        # Try to extract up to 3 cycles
        for _ in range(3):
            if time.time() > deadline:
                break
            
            # Use more attempts for earlier cycles (they're easier)
            max_att = 150 if len(cycles) == 0 else 100
            cycle = _extract_cycle(remaining, rng, max_attempts=max_att)
            if cycle is None:
                break
            
            cycles.append(cycle)
            
            # Remove used edges
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                ek = _edge_key(u, v)
                remaining.discard(ek)
        
        if len(cycles) > len(best_packing):
            best_packing = cycles[:]
        
        if len(best_packing) == 3:
            break
    
    # Verify best packing
    if best_packing:
        if _verify_packing(best_packing, all_edges_set):
            return [list(c) for c in best_packing]
    
    # Fallback
    if not best_packing:
        rng = random.Random(42)
        remaining = set(all_edges_set)
        cycle = _extract_cycle(remaining, rng, max_attempts=200)
        if cycle is not None:
            if _verify_packing([cycle], all_edges_set):
                return [list(cycle)]
    
    return []
# EVOLVE-BLOCK-END
