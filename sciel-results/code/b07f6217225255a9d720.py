import random
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edge_list):
        if not edge_list:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edge_list)
        n = max(max(e) for e in edge_set) + 1
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # 1. Generate candidate cycles
        candidate_cycles = []
        seen_edges = set()
        
        def normalize_cycle(cyc):
            if not cyc or len(cyc) < 3:
                return None
            # Rotate to start with min vertex
            min_val = min(cyc)
            min_idx = cyc.index(min_val)
            cyc = cyc[min_idx:] + cyc[:min_idx]
            # Ensure direction: second vertex < last vertex (if len > 2)
            # Actually, just check if reverse is "smaller" in some order?
            # Simpler: store as frozenset of edges for uniqueness, but keep vertex list.
            edges_in_c = frozenset((min(cyc[i], cyc[(i+1)%len(cyc)]), max(cyc[i], cyc[(i+1)%len(cyc)])) for i in range(len(cyc)))
            return (tuple(cyc), edges_in_c)

        rng = Random(42)
        # Run multiple DFS attempts with shuffled adjacency
        for attempt in range(300):
            # Pick a start vertex with degree >= 2
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            start = rng.choice(candidates)
            
            # Shuffle adjacency lists for this attempt
            adj_list = {v: list(adj[v]) for v in range(n)}
            for v in adj_list:
                rng.shuffle(adj_list[v])
            
            visited = {start}
            path = [start]
            found_cycle = None
            
            def dfs(node):
                nonlocal found_cycle
                if found_cycle:
                    return
                for nb in adj_list[node]:
                    if nb == start and len(path) >= 3:
                        found_cycle = path[:]
                        return
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb)
                        if found_cycle:
                            return
                        path.pop()
                        visited.discard(nb)
            
            dfs(start)
            
            if found_cycle:
                norm = normalize_cycle(found_cycle)
                if norm and norm[1] not in seen_edges:
                    seen_edges.add(norm[1])
                    candidate_cycles.append(norm)
                    
        # If no cycles found, return all edges as singles
        if not candidate_cycles:
            return [[u, v] for u, v in edge_set]

        # 2. Formulate MILP
        K = len(candidate_cycles)
        M = len(edge_set)
        edges_list = list(edge_set)
        edge_index = {e: i for i, e in enumerate(edges_list)}
        
        # Objective: Minimize sum(x_i) + sum(y_j)
        c_obj = np.zeros(K + M)
        c_obj[:K] = 1.0
        c_obj[K:] = 1.0
        
        # Constraints: For each edge e_j, sum(x_i where e_j in cycle_i) + y_j = 1
        A = np.zeros((M, K + M))
        for i, (verts, edges_f) in enumerate(candidate_cycles):
            for e in edges_f:
                j = edge_index[e]
                A[j, i] = 1.0
        
        for j in range(M):
            A[j, K+j] = 1.0
            
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(K + M)
        constraint = LinearConstraint(A, lb=np.ones(M), ub=np.ones(M))
        
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status != 0:
            # Fallback: Greedy longest cycle
            pieces = []
            remaining = set(edge_set)
            while remaining:
                best_cyc = None
                for verts, edges_f in candidate_cycles:
                    if edges_f.issubset(remaining):
                        if best_cyc is None or len(verts) > len(best_cyc[0]):
                            best_cyc = (verts, edges_f)
                if best_cyc:
                    pieces.append(list(best_cyc[0]))
                    remaining.difference_update(best_cyc[1])
                else:
                    e = next(iter(remaining))
                    pieces.append([e[0], e[1]])
                    remaining.discard(e)
            return pieces

        # Extract solution
        x = res.x[:K]
        # y = res.x[K:]
        
        pieces = []
        for i in range(K):
            if x[i] > 0.5:
                verts, _ = candidate_cycles[i]
                pieces.append(list(verts))
                
        # Add remaining edges as singles
        # Determine which edges are covered by cycles
        covered_edges = set()
        for i in range(K):
            if x[i] > 0.5:
                covered_edges.update(candidate_cycles[i][1])
                
        for e in edge_set:
            if e not in covered_edges:
                pieces.append([e[0], e[1]])
                
        return pieces

    return [decompose(instance(*spec)) for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))]
