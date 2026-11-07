"""Rewritten program: clique edge-partition of split graphs."""
import time

def construct():
    start = time.time()
    deadline = start + 140.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique(V, adj):
        V = set(V)
        best = []
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            pool = P | X
            u = max(pool, key=lambda x: len(adj[x] & P)) if pool else None
            for v in list(P - (adj[u] if u is not None else set())):
                R.add(v)
                bk(R, P & adj[v], X & adj[v])
                R.remove(v)
                P.remove(v)
                X.add(v)
        bk(set(), V, set())
        return best

    def solve(n, k, p, seed):
        K = list(range(k))
        I = list(range(k, n))
        
        # Generate cross edges
        rng = Random(seed)
        cross = {}
        for v in I:
            nbrs = set()
            for u in K:
                if rng.random() < p:
                    nbrs.add(u)
            cross[v] = nbrs

        cliques = []
        covered_KE = set()  # Set of K-K edges (u, w) with u < w that are already covered

        # Phase 1: Cover K-I edges using cross cliques.
        # For each v in I, we must cover all edges (u, v) for u in cross[v].
        # We partition cross[v] into subsets S such that:
        # 1. S is a clique in K (always true).
        # 2. No K-K edge inside S is already in covered_KE.
        # This ensures the clique S U {v} doesn't reuse any K-K edge.
        
        for v in I:
            remaining_neighbors = set(cross[v])
            while remaining_neighbors:
                # Build a graph on remaining_neighbors where edges exist only if
                # the K-K edge is NOT covered yet.
                # Since K is a complete graph, any pair is a K-K edge.
                # We need to find a clique in the "available" graph.
                # Available graph G_avail:
                # Nodes: remaining_neighbors
                # Edges: (u, w) if (min(u,w), max(u,w)) not in covered_KE
                
                if not remaining_neighbors:
                    break
                    
                # Heuristic: Pick a node, try to form a maximal clique
                # We can use a simple greedy approach or max_clique on the available graph.
                # Given small k, max_clique is fine.
                
                # Construct adjacency for the available graph
                avail_adj = {u: set() for u in remaining_neighbors}
                for u in remaining_neighbors:
                    for w in remaining_neighbors:
                        if u == w: continue
                        edge = (min(u, w), max(u, w))
                        if edge not in covered_KE:
                            avail_adj[u].add(w)
                            
                # Find max clique in this available graph
                mc = max_clique(remaining_neighbors, avail_adj)
                
                if not mc:
                    # If no edges available, we can only pick single vertices
                    # Any single vertex u forms clique {u, v} with no internal K-K edges
                    mc = [next(iter(remaining_neighbors))]
                
                # Form the clique
                clique = sorted(mc) + [v]
                cliques.append(clique)
                
                # Update covered_KE with all K-K edges inside mc
                mcl = sorted(mc)
                for a in range(len(mcl)):
                    for b in range(a + 1, len(mcl)):
                        covered_KE.add((mcl[a], mcl[b]))
                        
                remaining_neighbors -= set(mc)

        # Phase 2: Cover remaining K-K edges with pure K-cliques.
        # All K-K edges not in covered_KE must be covered.
        remaining_edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                if (i, j) not in covered_KE:
                    remaining_edges.add((i, j))
        
        # Greedily pack remaining edges into cliques.
        # Since K is a clique, any subset of vertices is a valid clique.
        # We want to minimize the number of cliques.
        # Strategy: While there are remaining edges, find a maximal clique
        # in the graph formed by remaining edges.
        while remaining_edges:
            # Build graph of remaining edges
            rem_adj = {u: set() for u in K}
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Find a maximal clique in this graph.
            # We can use the same max_clique algorithm.
            # Vertices are K, edges are remaining_edges.
            # Note: max_clique expects adj dict for vertices in V.
            mc = max_clique(set(K), rem_adj)
            
            if not mc:
                # Should not happen if remaining_edges is not empty,
                # but just in case, pick an edge
                u, v = next(iter(remaining_edges))
                mc = [u, v]
                
            cl = sorted(mc)
            cliques.append(cl)
            
            # Remove all edges covered by this clique from remaining_edges
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    remaining_edges.discard((cl[a], cl[b]))

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        result.append(solve(n, k, p, seed))

    cleaned = [[[int(x) for x in c] for c in clist] for clist in result]
    return cleaned
