import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if adj[u][v]:
                    neighbors[u].add(v)
        
        def bk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            if not P:
                return
            
            # Pivot
            max_count = -1
            pivot = -1
            for u in P:
                count = len(P & neighbors[u])
                if count > max_count:
                    max_count = count
                    pivot = u
            
            to_process = [v for v in P if v not in neighbors[pivot]]
            
            for v in to_process:
                R_new = R | {v}
                P_new = P & neighbors[v]
                X_new = X & neighbors[v]
                
                bk(R_new, P_new, X_new)
                
                P.remove(v)
                X.add(v)
                
                if time.time() > deadline:
                    return

        bk(set(), set(range(n)), set())
        return cliques

    def chvatal_greedy(cliques, vertex_to_cliques):
        if not cliques:
            return []
        
        num_cliques = len(cliques)
        clique_to_idx = {c: i for i, c in enumerate(cliques)}
        
        active_vertices = {}
        for v, cliques_set in vertex_to_cliques.items():
            idx_set = set()
            for c in cliques_set:
                idx_set.add(clique_to_idx[c])
            if idx_set:
                active_vertices[v] = idx_set
        
        if not active_vertices:
            return []
        
        remaining_cliques = set(range(num_cliques))
        selected = []
        selected_set = set()
        
        while remaining_cliques:
            if time.time() > deadline:
                break
                
            freq_current = [0] * num_cliques
            for v, idx_set in active_vertices.items():
                if v not in selected_set:
                    for idx in idx_set:
                        if idx in remaining_cliques:
                            freq_current[idx] += 1
            
            best_v = -1
            best_cost = float('inf')
            
            for v, idx_set in active_vertices.items():
                if v in selected_set:
                    continue
                cost = 0.0
                valid = True
                for idx in idx_set:
                    if idx in remaining_cliques:
                        if freq_current[idx] == 0:
                            valid = False
                            break
                        cost += 1.0 / freq_current[idx]
                if valid and cost < best_cost:
                    best_cost = cost
                    best_v = v
            
            if best_v == -1:
                break
            
            selected.append(best_v)
            selected_set.add(best_v)
            
            for idx in active_vertices[best_v]:
                if idx in remaining_cliques:
                    remaining_cliques.remove(idx)
        
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            results.append(list(range(n)))
            continue

        try:
            adj = instance(n, p, seed)
            cliques = bron_kerbosch(adj, n)
            
            if not cliques:
                results.append([])
                continue
            
            vertex_to_cliques = defaultdict(set)
            for c in cliques:
                for v in c:
                    vertex_to_cliques[v].add(c)
            
            hitting_set = chvatal_greedy(cliques, vertex_to_cliques)
            
            # Local search: remove redundant
            def is_valid(hs):
                hs_set = set(hs)
                for c in cliques:
                    if not (c & hs_set):
                        return False
                return True
            
            changed = True
            while changed:
                changed = False
                for i in range(len(hitting_set)):
                    new_hs = [h for j, h in enumerate(hitting_set) if j != i]
                    if is_valid(new_hs):
                        hitting_set = new_hs
                        changed = True
                        break
            
            results.append(sorted(hitting_set))
        except Exception:
            results.append(list(range(n)))
            
    return results
