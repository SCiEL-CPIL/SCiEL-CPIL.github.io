"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_one(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find odd-degree vertices
        odd_verts = sorted([v for v in range(n) if len(adj[v]) % 2 == 1])
        k = len(odd_verts)
        
        if k == 0:
            # Eulerian - find one circuit
            circuit = find_eulerian_circuit(adj, n)
            return [circuit] if circuit and len(circuit) > 1 else []
        
        # Pair up odd-degree vertices using minimum weight perfect matching
        # Cost = shortest path distance between the two vertices
        pairs = find_min_cost_pairs(odd_verts, adj, n)
        
        # Add virtual edges and find Eulerian circuit
        aug_adj = defaultdict(list)
        for u, v in edges:
            aug_adj[u].append(v)
            aug_adj[v].append(u)
        for u, v in pairs:
            aug_adj[u].append(v)
            aug_adj[v].append(u)
        
        circuit = find_eulerian_circuit(aug_adj, n)
        if not circuit:
            return [[u, v] for u, v in edges]
        
        # Split circuit at virtual edges
        virt_set = set()
        for u, v in pairs:
            virt_set.add(frozenset((u, v)))
        
        paths = []
        current = []
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            if not current:
                current = [u]
            current.append(v)
            if frozenset((u, v)) in virt_set:
                if len(current) >= 2:
                    paths.append(current)
                current = []
        
        if current and len(current) >= 2:
            paths.append(current)
        
        # Verify and fallback if needed
        if not paths:
            return [[u, v] for u, v in edges]
        
        return paths
    
    def find_eulerian_circuit(adj, n):
        """Hierholzer's algorithm for Eulerian circuit."""
        if not any(adj.values()):
            return []
        
        # Find start vertex
        start = None
        for v in range(n):
            if adj.get(v):
                start = v
                break
        if start is None:
            return []
        
        # Build edge list for tracking
        adj_dict = defaultdict(list)
        for u in adj:
            for v in adj[u]:
                adj_dict[u].append(v)
        
        used = set()
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            found = False
            for w in adj_dict[v]:
                e = frozenset((v, w))
                if e not in used:
                    used.add(e)
                    stack.append(w)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Verify it's a valid circuit
        if len(circuit) < 2:
            return []
        
        return circuit
    
    def find_min_cost_pairs(odd_verts, adj, n):
        """Find minimum cost pairing of odd-degree vertices using shortest paths."""
        k = len(odd_verts)
        if k <= 2:
            if k == 2:
                return [(odd_verts[0], odd_verts[1])]
            return []
        
        # Compute shortest path distances between all pairs of odd vertices
        # Using BFS since unweighted
        dist = {}
        for i, u in enumerate(odd_verts):
            # BFS from u
            from collections import deque
            dist_from_u = {u: 0}
            queue = deque([u])
            while queue:
                v = queue.popleft()
                for w in adj.get(v, []):
                    if w not in dist_from_u:
                        dist_from_u[w] = dist_from_u[v] + 1
                        queue.append(w)
            for j in range(i + 1, k):
                v = odd_verts[j]
                dist[(i, j)] = dist_from_u.get(v, n)  # n as infinity
        
        # Now find minimum weight perfect matching
        # Build cost matrix
        cost_mat = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    cost_mat[i, j] = 1e9
                elif i < j:
                    cost_mat[i, j] = dist.get((i, j), n)
                    cost_mat[j, i] = cost_mat[i, j]
        
        # For perfect matching, we need to select k/2 pairs
        # Use a simple approach: since k is small (at most n), 
        # we can use the fact that for the path cover, any pairing works
        # Let's just use greedy: sort pairs by cost and pick
        
        # Actually, let's use a simpler greedy: pair closest vertices
        # Build list of all pairs with costs
        pair_list = []
        for i in range(k):
            for j in range(i + 1, k):
                c = dist.get((i, j), n)
                pair_list.append((c, i, j))
        
        pair_list.sort()
        
        used = [False] * k
        pairs = []
        for c, i, j in pair_list:
            if not used[i] and not used[j]:
                pairs.append((odd_verts[i], odd_verts[j]))
                used[i] = True
                used[j] = True
                if len(pairs) == k // 2:
                    break
        
        return pairs
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        paths = solve_one(*spec)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END

