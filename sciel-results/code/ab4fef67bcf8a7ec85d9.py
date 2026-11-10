import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def max_clique(n, adj_comp):
        # Find Maximum Clique in adj_comp (which is MIS in original graph)
        
        # Order vertices by degree in adj_comp (descending)
        vertices = list(range(n))
        vertices.sort(key=lambda v: len(adj_comp[v]), reverse=True)
        
        best_clique = []
        best_size = 0
        
        # Greedy initial solution
        def greedy_clique(P):
            C = []
            P = set(P)
            while P:
                # Pick vertex with max degree in P
                v = max(P, key=lambda x: len(adj_comp[x] & P))
                C.append(v)
                P.remove(v)
                P = P & adj_comp[v]
            return C

        initial_sol = greedy_clique(set(range(n)))
        best_size = len(initial_sol)
        best_clique = initial_sol
        
        def color_bound(P):
            # Greedy coloring of P in adj_comp
            # Returns number of colors and mapping
            P_list = [v for v in vertices if v in P]
            color_map = {}
            num_colors = 0
            for v in P_list:
                used_colors = set()
                for u in adj_comp[v]:
                    if u in P and u in color_map:
                        used_colors.add(color_map[u])
                c = 0
                while c in used_colors:
                    c += 1
                color_map[v] = c
                if c + 1 > num_colors:
                    num_colors = c + 1
            return num_colors, P_list, color_map

        def solve(C, P):
            nonlocal best_size, best_clique
            if time.time() > deadline:
                return
            if not P:
                if len(C) > best_size:
                    best_size = len(C)
                    best_clique = C[:]
                return
            
            num_colors, P_list, color_map = color_bound(P)
            
            if len(C) + num_colors <= best_size:
                return
            
            # Pick vertex with highest color
            v = max(P_list, key=lambda x: color_map[x])
            
            # Branch 1: Include v
            C.append(v)
            P_new = P & adj_comp[v]
            solve(C, P_new)
            C.pop()
            
            # Branch 2: Exclude v
            P_new = P - {v}
            solve(C, P_new)

        solve([], set(range(n)))
        return best_clique

    def maximum_independent_set(n, adj_orig):
        # Build complement graph
        adj_comp = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj_orig[u]:
                    adj_comp[u].add(v)
                    adj_comp[v].add(u)
        
        clique = max_clique(n, adj_comp)
        return set(clique)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return greedy solution or empty
            # We must return a valid transversal.
            # Greedy MIS is valid.
            pass
        
        adj = build_graph(n, p, seed)
        mis_set = maximum_independent_set(n, adj)
        
        all_vertices = set(range(n))
        transversal = list(all_vertices - mis_set)
        results.append(transversal)

    return results

# EVOLVE-BLOCK-END
