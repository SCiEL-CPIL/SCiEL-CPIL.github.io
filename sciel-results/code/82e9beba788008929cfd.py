import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        C_edges = set()
        cross_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                C_edges.add((u, v))
            elif (u < k and v >= k) or (u >= k and v < k):
                cross_edges.add((u, v))
        
        N_i = [set() for _ in range(n)]
        for u, v in cross_edges:
            if u < k:
                i, c = u, v
            else:
                i, c = v, u
            N_i[i].add(c)
        
        result_cliques = []
        
        for i in range(k, n):
            if N_i[i]:
                clique = [i] + list(N_i[i])
                result_cliques.append(clique)
                members = list(N_i[i])
                for idx in range(len(members)):
                    for jdx in range(idx + 1, len(members)):
                        c1, c2 = members[idx], members[jdx]
                        if c1 < c2:
                            C_edges.discard((c1, c2))
                        else:
                            C_edges.discard((c2, c1))
        
        if C_edges:
            v_set = set()
            for u, v in C_edges:
                v_set.add(u)
                v_set.add(v)
            if v_set:
                result_cliques.append(list(v_set))
        
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results
