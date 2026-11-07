import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def bron_kerbosch_max_clique(verts, adj, time_limit=None):
        """Find a maximum clique in a graph using Bron-Kerbosch with pivot.
        adj: list of sets where adj[v] is the set of neighbors of v.
        Returns a list of vertices forming a maximum clique.
        """
        best_clique = []
        
        def bk(R, P, X):
            nonlocal best_clique
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            # Pivot: choose u from P ∪ X that maximizes |P ∩ N(u)|
            U = P | X
            if U:
                pivot = max(U, key=lambda u: len(P & adj[u]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(verts), set())
        return best_clique

    def clique_partition(n, k, p, seed):
        rng = random.Random(seed)
        # Build adjacency for K vertices (K is a clique, so all K-K edges exist)
        adj_K = [set(range(k)) for _ in range(k)]
        for u in range(k):
            adj_K[u].discard(u)
        
        # adj_I[i] is set of K neighbors of vertex k+i
        adj_I = [set() for _ in range(n - k)]
        
        # Generate cross edges
        for i in range(n - k):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_I[i].add(u)
        
        used_K_edges = set()
        cliques = []
        
        # Process I vertices in order of decreasing degree (number of K-neighbors)
        i_indices = sorted(range(n - k), key=lambda i: -len(adj_I[i]))
        
        for i in i_indices:
            v = k + i
            U = set(adj_I[i])
            while U:
                # Find largest subset S of U such that all pairs in S are not in used_K_edges
                # Build "available" graph: edge (u,w) exists if (u,w) NOT in used_K_edges
                # We want max clique in this available graph restricted to U
                
                # Build adjacency for the available graph on U
                # For efficiency, we only need adj within U
                avail_adj = {u: set() for u in U}
                for u in U:
                    for w in U:
                        if u == w:
                            continue
                        if (min(u, w), max(u, w)) not in used_K_edges:
                            avail_adj[u].add(w)
                
                # Use Bron-Kerbosch to find max clique in available graph
                S = bron_kerbosch_max_clique(U, avail_adj)
                
                if S:
                    # Add clique {v} ∪ S
                    clique = [v] + sorted(S)
                    cliques.append(clique)
                    # Update used_K_edges
                    for u, w in combinations(sorted(S), 2):
                        used_K_edges.add((u, w))
                    U -= set(S)
                else:
                    # Should not happen, but fallback: pick one vertex
                    u = next(iter(U))
                    clique = [v, u]
                    cliques.append(clique)
                    U.remove(u)
        
        # Cover remaining K edges
        rem_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in used_K_edges:
                    rem_edges.append((u, w))
        
        if rem_edges:
            # Build graph of remaining edges
            adj_rem = [set() for _ in range(k)]
            for u, w in rem_edges:
                adj_rem[u].add(w)
                adj_rem[w].add(u)
            
            remaining = set(rem_edges)
            
            while remaining:
                # Find vertices involved in remaining edges
                verts = set()
                for u, w in remaining:
                    verts.add(u)
                    verts.add(w)
                
                # Find max clique in the remaining graph
                C = bron_kerbosch_max_clique(verts, adj_rem)
                
                if not C:
                    break
                    
                clique = sorted(C)
                cliques.append(clique)
                
                # Remove edges in C from remaining and update adj_rem
                for u, w in combinations(sorted(C), 2):
                    edge = (min(u, w), max(u, w))
                    if edge in remaining:
                        remaining.remove(edge)
                        adj_rem[u].discard(w)
                        adj_rem[w].discard(u)

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
