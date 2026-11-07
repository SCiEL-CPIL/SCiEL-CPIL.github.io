"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
ALL_VERTS = list(range(N))

def _build_adj():
    adj = [set() for _ in range(N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, forbidden_edges, start=0, seed=0, deadline=None):
    """Find a Hamiltonian cycle in the graph with forbidden_edges removed.
    Uses backtracking with Warnsdorff heuristic."""
    random.seed(seed)
    # Build adjacency without forbidden edges
    adj2 = [set(s) for s in adj]
    for u, v in forbidden_edges:
        if v in adj2[u]:
            adj2[u].discard(v)
        if u in adj2[v]:
            adj2[v].discard(u)
    
    # Check degree constraints
    for v in range(N):
        if len(adj2[v]) < 2:
            return None
    
    path = [start]
    visited = {start}
    
    def dfs():
        if deadline and time.time() > deadline:
            return False
        if len(path) == N:
            # Check closing edge
            if path[-1] in adj2[start]:
                return True
            return False
        
        cur = path[-1]
        # Warnsdorff: order neighbors by their remaining degree
        neighbors = list(adj2[cur] - visited)
        if not neighbors:
            return False
        # Sort by degree in remaining graph (Warnsdorff)
        neighbors.sort(key=lambda x: len(adj2[x] - visited))
        # Add some randomness
        if seed and random.random() < 0.3:
            random.shuffle(neighbors)
        
        for nxt in neighbors:
            if deadline and time.time() > deadline:
                return False
            path.append(nxt)
            visited.add(nxt)
            if dfs():
                return True
            path.pop()
            visited.discard(nxt)
        return False
    
    if dfs():
        return list(path)
    return None

def construct():
    deadline = time.time() + 80  # 80s safety margin
    adj = _build_adj()
    
    # Try to find edge-disjoint Hamiltonian cycles
    # Strategy: try multiple random seeds and orderings
    best_cycles = []
    
    # First, try a deterministic approach with various starting vertices
    # and edge-removal strategies
    
    # Approach: iteratively find Hamiltonian cycles, removing their edges
    # Try multiple randomizations
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        forbidden = []
        cycles = []
        success = True
        
        for cycle_idx in range(3):
            if time.time() > deadline - 5:
                success = False
                break
            
            # Find a Hamiltonian cycle avoiding forbidden edges
            seed = attempt * 100 + cycle_idx
            found = False
            for sub_attempt in range(50):
                if time.time() > deadline - 5:
                    break
                s = seed + sub_attempt * 7
                cycle = _find_ham_cycle(adj, forbidden, start=0, seed=s, deadline=deadline - 5)
                if cycle is not None:
                    found = True
                    break
                # Also try different start vertices
                start_v = (s + cycle_idx) % N
                cycle = _find_ham_cycle(adj, forbidden, start=start_v, seed=s, deadline=deadline - 5)
                if cycle is not None:
                    found = True
                    break
            
            if not found:
                success = False
                break
            
            cycles.append(cycle)
            # Add edges of this cycle to forbidden
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                forbidden.append((u, v))
        
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) >= 3:
                break
    
    if not best_cycles:
        # Fallback: just return one Hamiltonian cycle
        cycle = _find_ham_cycle(adj, [], start=0, seed=42, deadline=deadline - 2)
        if cycle is not None:
            return [list(cycle)]
        # Absolute fallback
        return [list(range(N))]
    
    # Ensure types are correct
    result = []
    for c in best_cycles:
        result.append([int(v) for v in c])
    return result

# EVOLVE-BLOCK-END
