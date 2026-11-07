import time
from random import Random
from collections import defaultdict
import sys

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((u, v) if u < v else (v, u) for u, v in edges)
        
        # H is the set of K-edges currently "free" (not covered by I-cliques)
        # Initially all K-edges are free
        H = set((u, v) for u in range(k) for v in range(u + 1, k))
        
        # Build N(v) for v in I
        N = defaultdict(set)
        for v in range(k, n):
            for u in range(k):
                if (u, v) in edge_set:
                    N[v].add(u)
        
        cliques = []
        
        # Sort I vertices by degree in K (descending) to prioritize large cliques
        i_vertices = sorted([v for v in range(k, n) if N[v]], key=lambda v: len(N[v]), reverse=True)
        
        for v in i_vertices:
            nv = N[v]
            if not nv:
                continue
                
            # Find a maximal clique S in nv using only edges in H
            # Graph G_free on nv: edge (u, w) exists if (u, w) in H
            # Greedy clique finding
            candidates = list(nv)
            # Sort candidates by degree in G_free (within H)
            def deg_in_h(u):
                count = 0
                for w in candidates:
                    if u != w and (min(u, w), max(u, w)) in H:
                        count += 1
                return count
            
            candidates.sort(key=deg_in_h, reverse=True)
            
            S = []
            for u in candidates:
                # Check if u is connected to all in S via H
                compatible = True
                for s in S:
                    if (min(u, s), max(u, s)) not in H:
                        compatible = False
                        break
                if compatible:
                    S.append(u)
            
            if len(S) >= 2:
                # Use clique {v} U S
                cliques.append([v] + sorted(S))
                # Remove edges in S from H
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        H.discard(e)
                
                # Cover remaining cross-edges for v with 2-cliques
                for u in nv:
                    if u not in S:
                        cliques.append([u, v])
            else:
                # No large clique possible, use 2-cliques for all
                for u in nv:
                    cliques.append([u, v])
        
        # Cover remaining K-edges in H with K-cliques
        # Greedy: while H has edges, find maximal clique in H
        while H:
            # Pick vertex with max degree in H
            degrees = defaultdict(int)
            for u, v in H:
                degrees[u] += 1
                degrees[v] += 1
            
            if not degrees:
                break
                
            max_v = max(degrees, key=degrees.get)
            
            # Find neighbors of max_v in H
            neighbors = set()
            for u, v in H:
                if u == max_v:
                    neighbors.add(v)
                elif v == max_v:
                    neighbors.add(u)
            
            # Find maximal clique containing max_v
            clique_vertices = [max_v]
            candidate_neighbors = list(neighbors)
            candidate_neighbors.sort(key=lambda x: degrees.get(x, 0), reverse=True)
            
            for cand in candidate_neighbors:
                is_connected = True
                for cv in clique_vertices:
                    if cv == cand:
                        continue
                    if (min(cv, cand), max(cv, cand)) not in H:
                        is_connected = False
                        break
                if is_connected:
                    clique_vertices.append(cand)
            
            cliques.append(sorted(clique_vertices))
            
            # Remove edges within this clique from H
            for i in range(len(clique_vertices)):
                for j in range(i + 1, len(clique_vertices)):
                    u, w = clique_vertices[i], clique_vertices[j]
                    e = (min(u, w), max(u, w))
                    H.discard(e)
            
            if time.time() > deadline - 5:
                # Fallback
                for e in H:
                    cliques.append([e[0], e[1]])
                H.clear()
                break
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
