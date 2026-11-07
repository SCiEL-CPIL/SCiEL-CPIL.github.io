import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def clique_cover_k_edges(remaining_edges, k, deadline):
        """Cover remaining K-edges with minimum number of cliques (greedy)."""
        if not remaining_edges:
            return []
        cliques = []
        rem = set(remaining_edges)
        while rem:
            if time.time() > deadline:
                # Fallback
                for e in rem:
                    cliques.append(list(e))
                return cliques
            u, v = next(iter(rem))
            # Find max clique containing (u,v)
            cand = set()
            for (a, b) in rem:
                if a == u: cand.add(b)
                elif b == u: cand.add(a)
                if a == v: cand.add(b)
                elif b == v: cand.add(a)
            valid = []
            for c in cand:
                if c == u or c == v: continue
                if (min(u,c),max(u,c)) in rem and (min(v,c),max(v,c)) in rem:
                    valid.append(c)
            clique = {u, v}
            for c in valid:
                ok = all((min(x,c),max(x,c)) in rem for x in clique if x != c)
                if ok:
                    clique.add(c)
            cl = list(clique)
            cliques.append(cl)
            to_rm = set()
            for i in range(len(cl)):
                for j in range(i+1, len(cl)):
                    to_rm.add((min(cl[i],cl[j]), max(cl[i],cl[j])))
            rem -= to_rm
        return cliques

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Key insight: each K-edge can be used by at most one v's clique.
        # For each v, partition N(v) into subsets. Each subset S gives clique {v}∪S.
        # K-edges within S are "consumed" by v.
        # 
        # Optimal strategy: use pairs (best ratio of savings to K-edges consumed).
        # But we need to be strategic about WHICH pairs to form to leave
        # remaining K-edges easily coverable.
        #
        # Approach: 
        # 1. For each v, greedily form pairs from N(v) using available K-edges.
        # 2. Cover remaining K-edges.
        # 3. Try to improve: if using a triple for some v saves a net clique, do it.
        
        available_k_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        cliques = []
        
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = neighbors[v]
            if not N_v:
                continue
            
            matched = set()
            pairs = []
            N_v_list = list(N_v)
            
            i = 0
            while i < len(N_v_list):
                if N_v_list[i] in matched:
                    i += 1
                    continue
                u = N_v_list[i]
                for j in range(i+1, len(N_v_list)):
                    w = N_v_list[j]
                    if w in matched:
                        continue
                    edge = (min(u,w), max(u,w))
                    if edge in available_k_edges:
                        pairs.append((u, w))
                        matched.add(u)
                        matched.add(w)
                        available_k_edges.discard(edge)
                        break
                i += 1
            
            for u, w in pairs:
                cliques.append([u, w, v])
            for u in N_v:
                if u not in matched:
                    cliques.append([u, v])
        
        # Cover remaining K-edges
        k_cliques = clique_cover_k_edges(list(available_k_edges), k, deadline)
        cliques.extend(k_cliques)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        cliques = solve_instance(*spec)
        clean_cliques = [[int(x) for x in c] for c in cliques]
        results.append(clean_cliques)
    
    return results
