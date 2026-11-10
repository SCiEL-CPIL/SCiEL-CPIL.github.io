import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp
from random import Random

def construct():
    def greedy_partition(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        remaining_adj = list(adj)
        cliques = []
        
        def find_max_clique(P, local_adj):
            best = []
            def bk(P, X, R):
                nonlocal best
                if not P and not X:
                    if len(R) > len(best):
                        best = list(R)
                    return
                if not P:
                    return
                pivot = -1
                max_deg = -1
                p_temp = P
                while p_temp:
                    lsb = p_temp & -p_temp
                    idx = lsb.bit_length() - 1
                    deg = (local_adj[idx] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        pivot = idx
                    p_temp ^= lsb
                if pivot == -1:
                    return
                candidates = P & ~(local_adj[pivot] & P)
                c_temp = candidates
                while c_temp:
                    lsb = c_temp & -c_temp
                    v = lsb.bit_length() - 1
                    new_P = P & local_adj[v]
                    bk(new_P, X, R + [v])
                    P ^= lsb
                    X |= lsb
                    c_temp ^= lsb
            bk(P, 0, [])
            return best

        while True:
            has_edges = False
            for i in range(n):
                if remaining_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            deg = [0] * n
            for i in range(n):
                deg[i] = remaining_adj[i].bit_count()
            active = [i for i in range(n) if deg[i] > 0]
            if not active:
                break
            active.sort(key=lambda x: deg[x], reverse=True)
            top_k = active[:5]
            
            best_clique = []
            for v in top_k:
                P = remaining_adj[v]
                if P == 0:
                    continue
                clique_in_neighbors = find_max_clique(P, remaining_adj)
                candidate = clique_in_neighbors + [v]
                if len(candidate) > len(best_clique):
                    best_clique = candidate
            
            if not best_clique:
                found = False
                for i in range(n):
                    if remaining_adj[i]:
                        lsb = remaining_adj[i] & -remaining_adj[i]
                        j = lsb.bit_length() - 1
                        best_clique = [i, j]
                        found = True
                        break
                if not found:
                    break
            
            for i in range(len(best_clique)):
                u = best_clique[i]
                mask = 0
                for j in range(len(best_clique)):
                    if i != j:
                        mask |= (1 << best_clique[j])
                remaining_adj[u] &= ~mask
                
            cliques.append(best_clique)
            
        return cliques

    def ilp_partition(n, p, seed):
        rng = Random(seed)
        edges = []
        edge_index = {}
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edge_index[(u, v)] = len(edges)
                    edges.append((u, v))
        
        if not edges:
            return []

        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        cliques = []
        
        def expand_clique(P, X, R):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(tuple(R))
                return
            if not P:
                return
            pivot = -1
            max_deg = -1
            p_temp = P
            while p_temp:
                lsb = p_temp & -p_temp
                idx = lsb.bit_length() - 1
                deg = (adj[idx] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = idx
                p_temp ^= lsb
            if pivot == -1:
                return
            candidates = P & ~(adj[pivot] & P)
            c_temp = candidates
            while c_temp:
                lsb = c_temp & -c_temp
                v = lsb.bit_length() - 1
                new_P = P & adj[v]
                expand_clique(new_P, X, R + [v])
                P ^= lsb
                X |= lsb
                c_temp ^= lsb
                
        expand_clique((1 << n) - 1, 0, [])
        
        if not cliques:
            return []
            
        if len(cliques) > 50000:
            return greedy_partition(n, p, seed)

        m_edges = len(edges)
        m_cliques = len(cliques)
        
        row_indices = []
        col_indices = []
        data = []
        
        for c_idx, clique in enumerate(cliques):
            for u in clique:
                for v in clique:
                    if u < v:
                        e_idx = edge_index.get((u, v))
                        if e_idx is not None:
                            row_indices.append(e_idx)
                            col_indices.append(c_idx)
                            data.append(1)
                            
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(m_edges, m_cliques))
        
        c = np.ones(m_cliques)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(m_cliques)
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb=np.ones(m_edges), ub=np.ones(m_edges)),
                integrality=integrality,
                bounds=bounds
            )
            if res.status == 0:
                x = res.x
                selected_cliques = []
                for i in range(m_cliques):
                    if x[i] > 0.5:
                        selected_cliques.append(list(cliques[i]))
                return selected_cliques
        except Exception:
            pass
            
        return greedy_partition(n, p, seed)

    res1 = ilp_partition(32, 0.6, 601)
    res2 = ilp_partition(64, 0.55, 602)
    
    return [res1, res2]
