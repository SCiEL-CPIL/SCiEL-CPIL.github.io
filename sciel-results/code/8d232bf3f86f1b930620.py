"""Rewritten program: exploit circulant structure for guaranteed 3 edge-disjoint Hamiltonian cycles."""
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    n = P['n']
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    # Try circulant construction: find step sizes that give Hamiltonian cycles
    # For a circulant graph C(n, S), step k gives a Hamiltonian cycle if gcd(k,n)=1
    # and all edges {i, i+k} are in the graph for all i.
    
    def make_cycle_step(k):
        """Generate a Hamiltonian cycle by stepping by k mod n."""
        cycle = []
        v = 0
        for _ in range(n):
            cycle.append(v)
            v = (v + k) % n
        # Verify all edges exist
        for i in range(n):
            u = cycle[i]
            w = cycle[(i + 1) % n]
            if _edge_key(u, w) not in all_edges:
                return None
        # Verify it's a simple cycle (all distinct)
        if len(set(cycle)) != n:
            return None
        return cycle
    
    # Find valid step sizes (coprime to n, and edges exist)
    valid_steps = []
    for k in range(1, n):
        import math
        if math.gcd(k, n) != 1:
            continue
        # Check if step k and step n-k give edges in the graph
        # (we just need one direction since edge is undirected)
        ok = True
        for i in range(n):
            if _edge_key(i, (i + k) % n) not in all_edges:
                ok = False
                break
        if ok:
            valid_steps.append(k)
    
    # Sort by step size for determinism
    valid_steps.sort()
    
    cycles = []
    used_edges = set()
    
    for k in valid_steps:
        if len(cycles) >= 10:
            break
        cycle = make_cycle_step(k)
        if cycle is None:
            continue
        # Check edge-disjointness
        cycle_edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            w = cycle[(i + 1) % len(cycle)]
            cycle_edges.add(_edge_key(u, w))
        if cycle_edges & used_edges:
            continue
        used_edges |= cycle_edges
        cycles.append(cycle)
    
    # If we got 3 cycles, we're done (optimal for 6-regular with 63 edges)
    if len(cycles) >= 3:
        result = []
        for cycle in cycles[:10]:
            result.append([int(v) for v in cycle])
        return result
    
    # Fallback: use backtracking to find more cycles
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 75
    
    def find_one_cycle():
        if time.time() > deadline:
            return None
        for start in range(n):
            if time.time() > deadline:
                return None
            visited = {start}
            path = [start]
            
            def dfs(current, depth):
                if time.time() > deadline:
                    return None
                if depth == n:
                    edge = _edge_key(current, path[0])
                    if edge not in used_edges:
                        return path[:]
                    return None
                for neighbor in sorted(adj[current]):
                    if neighbor not in visited:
                        edge = _edge_key(current, neighbor)
                        if edge not in used_edges:
                            visited.add(neighbor)
                            path.append(neighbor)
                            result = dfs(neighbor, depth + 1)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(neighbor)
                return None
            
            result = dfs(start, 1)
            if result is not None:
                return result
        return None
    
    for i in range(10 - len(cycles)):
        if time.time() > deadline - 5:
            break
        cycle = find_one_cycle()
        if cycle is None:
            break
        for j in range(len(cycle)):
            u = cycle[j]
            w = cycle[(j + 1) % len(cycle)]
            used_edges.add(_edge_key(u, w))
        cycles.append(cycle)
    
    if not cycles:
        # Absolute fallback: return a single known cycle
        cycles = [[i for i in range(n)]]
    
    result = []
    for cycle in cycles[:10]:
        result.append([int(v) for v in cycle])
    return result
