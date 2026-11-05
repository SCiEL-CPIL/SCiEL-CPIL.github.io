import time
import random
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, adj, n):
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        
        if P:
            pivot = max(P, key=lambda p: len(P & adj[p]))
        elif X:
            pivot = max(X, key=lambda p: len(P & adj[p]))
        else:
            return
            
        for v in list(P - adj[pivot]):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X, adj, n)
            P.remove(v)
            X.add(v)

    def find_max_matching_bipartite(adj_left, adj_right, n_left, n_right):
        """Find maximum matching in a bipartite graph using Hopcroft-Karp."""
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            from collections import deque
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj_left[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found
            
        def dfs(u):
            for v in adj_left[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
            
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u, pair_v, matching

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        all_cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        valid_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        C = len(valid_cliques)
        
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].add(i)
                
        best_set = None
        best_size = float('inf')
        
        # Strategy 1: Exact ILP via scipy.optimize.milp
        try:
            # Build constraint matrix A: C x n
            # A[i][v] = 1 if vertex v is in clique i
            A = np.zeros((C, n))
            for i, c in enumerate(valid_cliques):
                for v in c:
                    A[i][v] = 1.0
            
            # Objective: minimize sum of x_v
            c_obj = np.ones(n)
            
            # Constraints: A @ x >= 1 for each clique
            cons = LinearConstraint(A, lb=np.ones(C), ub=np.inf)
            
            # Bounds: 0 <= x_v <= 1, integer
            bounds = Bounds(lb=0, ub=1)
            
            # Solve MILP
            res = milp(c=c_obj, constraints=cons, integrality=np.ones(n), bounds=bounds)
            
            if res.status == 0:  # Optimal
                x = res.x
                selected = set()
                for v in range(n):
                    if x[v] > 0.5:
                        selected.add(v)
                
                # Verify validity
                valid = True
                for i, c in enumerate(valid_cliques):
                    if not any(v in selected for v in c):
                        valid = False
                        break
                
                if valid:
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_set = set(selected)
                else:
                    # Fallback: add vertices to fix invalid solution
                    for i, c in enumerate(valid_cliques):
                        if not any(v in selected for v in c):
                            # Pick a vertex from this clique
                            for v in c:
                                if v not in selected:
                                    selected.add(v)
                                    break
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_set = set(selected)
            elif res.status == 2:  # Feasible but not optimal (due to time limit)
                x = res.x
                selected = set()
                for v in range(n):
                    if x[v] > 0.5:
                        selected.add(v)
                
                # Verify and fix
                valid = True
                for i, c in enumerate(valid_cliques):
                    if not any(v in selected for v in c):
                        valid = False
                        break
                
                if not valid:
                    for i, c in enumerate(valid_cliques):
                        if not any(v in selected for v in c):
                            for v in c:
                                if v not in selected:
                                    selected.add(v)
                                    break
                
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = set(selected)
        except Exception:
            pass
        
        # Strategy 2: Greedy + local search as fallback
        rng_solver = random.Random(seed + 123)
        
        # Standard Greedy with random restarts
        num_restarts = 500
        
        for run in range(num_restarts):
            if time.time() > deadline:
                break
                
            hit_cliques = set()
            current_set = set()
            
            while len(hit_cliques) < C:
                best_v = -1
                max_hits = -1
                candidates = []
                
                for v in range(n):
                    if v in current_set:
                        continue
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if ci not in hit_cliques:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)
                        
                if max_hits == 0:
                    break
                    
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)
                    
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = valid_cliques[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
