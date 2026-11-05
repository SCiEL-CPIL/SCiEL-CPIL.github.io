import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_vc_ilp(n, edges):
        if not edges:
            return set()
        
        # Number of variables = n
        # Minimize sum(x_v)
        c = np.ones(n)
        
        # Constraints: x_u + x_v >= 1 for each edge (u, v)
        # We need to build the constraint matrix A
        # A * x >= 1
        # A is |E| x n
        m = len(edges)
        A = np.zeros((m, n))
        for i, (u, v) in enumerate(edges):
            A[i, u] = 1
            A[i, v] = 1
            
        # LinearConstraint(A, 1, np.inf)
        # x_i in [0, 1]
        integrality = np.ones(n)
        
        try:
            res = milp(
                c=c,
                constraints=[LinearConstraint(A, 1, np.inf)],
                integrality=integrality,
                bounds=[Bounds(0, 1)] * n
            )
            if res.status == 0: # Optimal
                # res.x contains the solution
                # Round to nearest integer
                cover = set(int(x) for x in res.x if x > 0.5)
                return cover
            else:
                # Fallback if ILP fails
                pass
        except Exception:
            pass
        
        # Fallback: Greedy + Local Search
        return greedy_local_search(n, edges, DEADLINE)

    def greedy_local_search(n, edges, deadline):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        uncovered = set(edges)
        cover = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            edge_count = {i: 0 for i in range(n)}
            for u, v in uncovered:
                edge_count[u] += 1
                edge_count[v] += 1
            best_v = max(range(n), key=lambda x: edge_count[x])
            if edge_count[best_v] == 0:
                break
            cover.add(best_v)
            uncovered = {(u, v) for u, v in uncovered if u != best_v and v != best_v}
            
        # Local search improvement
        edges_set = set(edges)
        def is_valid_cover(c):
            for u, v in edges:
                if u not in c and v not in c:
                    return False
            return True
            
        vertex_edges = {i: set() for i in range(n)}
        for u, v in edges:
            vertex_edges[u].add((u, v))
            vertex_edges[v].add((u, v))
            
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and time.time() < deadline and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 1. Try removing vertices
            for u in sorted(list(cover)):
                if time.time() > deadline:
                    break
                if u not in cover:
                    continue
                new_cover = cover - {u}
                if is_valid_cover(new_cover):
                    cover = new_cover
                    improved = True
            
            # 2. Try 1-swap moves
            if not improved and time.time() < deadline:
                found_swap = False
                for u in sorted(list(cover)):
                    if time.time() > deadline:
                        break
                    if u not in cover:
                        continue
                    for v in range(n):
                        if v in cover:
                            continue
                        new_cover = (cover - {u}) | {v}
                        if is_valid_cover(new_cover):
                            cover = new_cover
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
                        
        return cover

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            edges = get_graph(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        edges = get_graph(n, p, seed)
        cover = solve_vc_ilp(n, edges)
        results.append(sorted(cover))

    return results
