import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute a good initial Hamiltonian cycle using a simple heuristic
    # For a 6-regular graph on 21 vertices, we can often find one quickly
    
    def find_hamiltonian_fast(adj_cur, start, deadline):
        """Fast Hamiltonian cycle finder using DFS with backtracking and heuristic ordering."""
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        
        def dfs():
            if time.time() > deadline:
                return False
            if len(path) == n2:
                return start in adj_cur[path[-1]]
            
            u = path[-1]
            # Order neighbors by degree (Warnsdorff-like)
            nbrs = sorted(adj_cur[u], key=lambda w: len(adj_cur[w]))
            
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    path.append(w)
                    if dfs():
                        return True
                    path.pop()
                    visited[w] = False
            return False
        
        return list(path) if dfs() else None
    
    # Strategy: Find 3 edge-disjoint Hamiltonian cycles
    # Since the graph is 6-regular with 21 vertices and 63 edges,
    # and each Hamiltonian cycle uses 21 edges, we can have at most 3.
    
    def attempt_3_cycles(seed):
        rng = random.Random(seed)
        adj_cur = [set(s) for s in adj]
        cycles = []
        
        # Find first Hamiltonian cycle
        c1 = None
        for start in rng.sample(range(n), n):
            if time.time() > deadline:
                return None
            c1 = find_hamiltonian_fast(adj_cur, start, deadline)
            if c1:
                break
        if not c1:
            return None
        cycles.append(c1)
        
        # Remove edges of C1
        for i in range(len(c1)):
            a, b = c1[i], c1[(i + 1) % len(c1)]
            adj_cur[a].discard(b)
            adj_cur[b].discard(a)
        
        # Check min degree >= 2
        if any(len(adj_cur[v]) < 2 for v in range(n)):
            return None
        
        # Find second Hamiltonian cycle
        c2 = None
        for start in rng.sample(range(n), n):
            if time.time() > deadline:
                return None
            c2 = find_hamiltonian_fast(adj_cur, start, deadline)
            if c2:
                break
        if not c2:
            return None
        cycles.append(c2)
        
        # Remove edges of C2
        for i in range(len(c2)):
            a, b = c2[i], c2[(i + 1) % len(c2)]
            adj_cur[a].discard(b)
            adj_cur[b].discard(a)
        
        # Check if remaining graph is a single Hamiltonian cycle
        if any(len(adj_cur[v]) != 2 for v in range(n)):
            return None
        
        # Verify it's a single cycle (not multiple disjoint cycles)
        visited = set()
        curr = 0
        path = [0]
        visited.add(0)
        for _ in range(n - 1):
            nbrs = list(adj_cur[curr])
            next_v = None
            for nb in nbrs:
                if nb not in visited:
                    next_v = nb
                    break
            if next_v is None:
                return None
            curr = next_v
            visited.add(curr)
            path.append(curr)
        
        if len(path) == n and 0 in adj_cur[curr]:
            cycles.append(path)
            return cycles
        return None
    
    # Try multiple seeds to find a valid decomposition
    best = None
    for seed in range(1000):
        if time.time() > deadline:
            break
        result = attempt_3_cycles(seed)
        if result:
            # Validate
            used_edges = set()
            valid = True
            for cyc in result:
                if len(cyc) < 3 or len(set(cyc)) != len(cyc):
                    valid = False
                    break
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                    k = _edge_key(a, b)
                    if k not in full_edges or k in used_edges:
                        valid = False
                        break
                    used_edges.add(k)
                if not valid:
                    break
            if valid:
                best = [[int(v) for v in cyc] for cyc in result]
                break
    
    if best:
        return best
    
    # Fallback: return a single valid cycle if we can find one
    adj_cur = [set(s) for s in adj]
    for start in range(n):
        if time.time() > deadline:
            break
        c = find_hamiltonian_fast(adj_cur, start, deadline)
        if c:
            return [[int(v) for v in c]]
    
    # Last resort: return an empty list (invalid, but better than crashing)
    return []
