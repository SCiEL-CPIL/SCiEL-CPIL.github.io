import time
from random import Random
import sys

def construct():
    t_start = time.time()
    deadline = t_start + 138.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_adj(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
        return adj, edge_set, edges

    def max_clique_bron_kerbosch(adj, vertices):
        """Find a maximum clique in the subgraph induced by vertices."""
        if not vertices:
            return []
        
        # Use a simple recursive Bron-Kerbosch with pivot
        # adj is the global adjacency list, but we only care about edges within 'vertices'
        # To speed up, we can create a local adjacency dict for the subgraph
        local_adj = {v: set() for v in vertices}
        for u in vertices:
            for w in adj[u]:
                if w in local_adj:
                    local_adj[u].add(w)
        
        best_clique = []
        
        def bk(R, P, X):
            nonlocal best_clique
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = R.copy()
                return
            
            # Pivot selection: choose u in P U X with max degree in P
            pivot_candidates = list(P.union(X))
            if not pivot_candidates:
                return
            u = max(pivot_candidates, key=lambda x: len(local_adj[x].intersection(P)))
            
            # Recurse on vertices in P that are not neighbors of u
            for v in list(P.difference(local_adj[u])):
                bk(R.union({v}), P.intersection(local_adj[v]), X.intersection(local_adj[v]))
                P.remove(v)
                X.add(v)
                
        bk(set(), set(vertices), set())
        return list(best_clique)

    def solve_instance(n, k, p, seed):
        adj, edge_set, edges = build_adj(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        # G_avail: set of available edges in C
        # Initially all edges in C
        avail_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        
        partition = []
        
        # Sort I by degree in C descending (greedy: handle high degree first to absorb more)
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=True)
        
        for v in I_sorted:
            N_v = adj[v] & set(C)
            if not N_v:
                continue
            
            # Find Max Clique in G_avail induced by N_v
            # G_avail is defined by avail_edges
            # We need to check if a set S is a clique in G_avail
            # This is equivalent to: for all u, w in S, (min(u,w), max(u,w)) in avail_edges
            
            # To use Bron-Kerbosch, we need an adjacency structure for G_avail[N_v]
            # Let's construct it
            verts = list(N_v)
            local_avail_adj = {u: set() for u in verts}
            for i in range(len(verts)):
                u = verts[i]
                for j in range(i + 1, len(verts)):
                    w = verts[j]
                    e = (min(u, w), max(u, w))
                    if e in avail_edges:
                        local_avail_adj[u].add(w)
                        local_avail_adj[w].add(u)
            
            # Find Max Clique
            best_clique = []
            
            def bk(R, P, X):
                nonlocal best_clique
                if not P and not X:
                    if len(R) > len(best_clique):
                        best_clique = R.copy()
                    return
                if not P:
                    return
                # Pivot
                u = max(P.union(X), key=lambda x: len(local_avail_adj[x].intersection(P))) if (P.union(X)) else None
                if u is None:
                    return
                for v in list(P.difference(local_avail_adj[u])):
                    bk(R.union({v}), P.intersection(local_avail_adj[v]), X.intersection(local_avail_adj[v]))
                    P.remove(v)
                    X.add(v)
                    
            bk(set(), set(verts), set())
            
            if len(best_clique) >= 2:
                # Form clique {v} U best_clique
                clique = sorted([v] + best_clique)
                partition.append(clique)
                # Remove edges in best_clique from avail_edges
                bc = list(best_clique)
                for i in range(len(bc)):
                    for j in range(i + 1, len(bc)):
                        a, b = bc[i], bc[j]
                        avail_edges.discard((min(a, b), max(a, b)))
            else:
                # No clique of size >= 2 available in N_v
                # Must cover cross edges with size-2 cliques
                for u in N_v:
                    partition.append(sorted([v, u]))
                    # No K-edges removed
        
        # Cover remaining avail_edges
        # These are edges in C
        # Use greedy Max Clique partition
        rem_edges = set(avail_edges)
        while rem_edges:
            # Get vertices involved
            verts_involved = set()
            for u, v in rem_edges:
                verts_involved.add(u)
                verts_involved.add(v)
            
            # Build local adj for rem_edges
            local_adj = {u: set() for u in verts_involved}
            for u, v in rem_edges:
                local_adj[u].add(v)
                local_adj[v].add(u)
            
            # Find Max Clique
            best_clique = []
            def bk_rem(R, P, X):
                nonlocal best_clique
                if not P and not X:
                    if len(R) > len(best_clique):
                        best_clique = R.copy()
                    return
                if not P:
                    return
                u = max(P.union(X), key=lambda x: len(local_adj[x].intersection(P))) if (P.union(X)) else None
                if u is None:
                    return
                for v in list(P.difference(local_adj[u])):
                    bk_rem(R.union({v}), P.intersection(local_adj[v]), X.intersection(local_adj[v]))
                    P.remove(v)
                    X.add(v)
                    
            bk_rem(set(), set(verts_involved), set())
            
            if not best_clique:
                # Fallback: take one edge
                e = next(iter(rem_edges))
                best_clique = [e[0], e[1]]
            
            clique = sorted(best_clique)
            partition.append(clique)
            
            # Remove edges in best_clique
            bc = list(best_clique)
            for i in range(len(bc)):
                for j in range(i + 1, len(bc)):
                    a, b = bc[i], bc[j]
                    rem_edges.discard((min(a, b), max(a, b)))
                    
        return partition

    def validate_partition(partition, n, k, p, seed):
        adj, edge_set, edges = build_adj(n, k, p, seed)
        seen_edges = set()
        for clique in partition:
            if len(clique) < 2:
                return False
            if len(set(clique)) != len(clique):
                return False
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if b not in adj[a]:
                        return False
                    e = (min(a, b), max(a, b))
                    if e in seen_edges:
                        return False
                    seen_edges.add(e)
        if seen_edges != edge_set:
            return False
        return True

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for spec in specs:
        if time.time() > deadline - 5:
            while len(results) < 3:
                results.append([])
            return results
        
        n, k, p, seed = spec
        result = solve_instance(n, k, p, seed)
        
        if not validate_partition(result, n, k, p, seed):
            # Fallback to simple edge cover if validation fails (should not happen)
            # Just cover each edge individually
            adj, edge_set, edges = build_adj(n, k, p, seed)
            result = [[u, v] for u, v in edge_set]
            
        cleaned = []
        for clique in result:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
    
    return results
