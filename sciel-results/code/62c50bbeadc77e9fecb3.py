import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 50.0
    n = P['n']
    
    edge_set = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def find_hamiltonian_cycle(available, seed_val=0):
        """Find a Hamiltonian cycle using only edges in available.
        Uses randomized DFS with degree-based pruning."""
        rng = random.Random(seed_val)
        start = 0
        path = [start]
        visited = {start}
        # Precompute which neighbors are available
        avail_adj = {i: set() for i in range(n)}
        for u in range(n):
            for v in adj[u]:
                if edge_key(u, v) in available:
                    avail_adj[u].add(v)
        
        def backtrack(current, depth):
            if time.time() > deadline:
                return None
            if depth == n:
                last = path[-1]
                if edge_key(last, start) in available:
                    return list(path)
                return None
            last = path[-1]
            # Get available neighbors not yet visited
            candidates = [v for v in avail_adj[last] if v not in visited]
            if not candidates:
                return None
            
            # Pruning: check if any unvisited vertex has only 1 available edge to path or unvisited
            # This helps avoid dead ends
            # Actually, let's use a simpler heuristic: order candidates by degree in remaining graph
            # Vertices with fewer remaining connections are more constrained, visit them sooner? No, we visit from last.
            # Better: if a candidate leads to a vertex that has no available neighbors except the one we came from, it's a dead end (unless it's the last).
            # Let's just sort by degree in available graph for better exploration
            candidates.sort(key=lambda v: len(avail_adj[v] - visited))
            
            # Also check: is there an unvisited vertex that has 0 available edges to other unvisited vertices?
            # If so, it must be adjacent to 'last' and be the last vertex visited.
            unvisited = [v for v in range(n) if v not in visited]
            # Count available edges within unvisited
            for u in unvisited:
                avail_to_unvisited = [v for v in avail_adj[u] if v in unvisited and v != u]
                if len(avail_to_unvisited) == 0:
                    # u must be visited next (if depth == n-1) or it's a dead end
                    if depth == n - 1:
                        if u in candidates:
                            return [last, u] if False else None  # This logic is getting complex
                    # If not the last step, u is isolated in the remaining graph -> dead end unless it's adjacent to last and will be the last
                    # Actually if u has no edges to other unvisited, it can only be reached from visited or from 'last'
                    # If depth < n-1, we still need to visit more vertices, so u must be adjacent to 'last' and be the next, but then it would be isolated immediately
                    # So if len(unvisited) > 1 and u has 0 edges to other unvisited, it's a dead end (can't be reached later)
                    # Wait, u can be reached from 'last' now, but then u has no edges to other unvisited, so after visiting u, we'd be stuck if there are other unvisited.
                    # So if len(unvisited) > 1 and u has 0 edges to other unvisited, return None (dead end)
                    if len(unvisited) > 1:
                        return None
                    # If len(unvisited) == 1, then u must be the next one
                    elif depth == n - 1:
                        if u in candidates:
                            path.append(u)
                            visited.add(u)
                            # Check closing edge
                            if edge_key(u, start) in available:
                                return list(path)
                            path.pop()
                            visited.remove(u)
                            return None
                        else:
                            return None
                    else:
                        # depth < n-1 but unvisited has only 1 vertex? Impossible since depth + len(unvisited) = n
                        return None
            
            for nb in candidates:
                path.append(nb)
                visited.add(nb)
                result = backtrack(nb, depth + 1)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nb)
            
            return None
        
        return backtrack(start, 1)
    
    # Greedily find edge-disjoint Hamiltonian cycles
    cycles = []
    available = set(edge_set)
    random.seed(42)
    
    # Try to find 3 cycles (theoretical max)
    for attempt in range(10):
        if time.time() > deadline:
            break
        found = None
        # Try multiple seeds for randomized restarts
        for seed_val in range(100):
            if time.time() > deadline:
                break
            found = find_hamiltonian_cycle(available, seed_val)
            if found is not None:
                break
        if found is None:
            break
        # Remove used edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i + 1) % len(found)]
            available.discard(edge_key(u, v))
        cycles.append(found)
    
    if cycles:
        return [list(c) for c in cycles]
    else:
        return [list(range(n))]
# EVOLVE-BLOCK-END
