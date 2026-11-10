"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def greedy_clique_cover(adj, vertices):
        """Cover edges of graph (adj: dict vertex->set of neighbors) with minimum-ish cliques.
        Returns list of cliques (each a list of vertices)."""
        remaining_edges = set()
        for u in vertices:
            for v in adj.get(u, set()):
                if v > u:
                    remaining_edges.add((u, v))
        
        cliques = []
        while remaining_edges:
            # Find a maximal clique greedily
            # Pick an edge, build maximal clique containing it
            e = next(iter(remaining_edges))
            u, v = e
            # Start with {u, v}, try to add more vertices
            clique = {u, v}
            candidates = [x for x in vertices if x != u and x != v]
            for x in candidates:
                if x in adj.get(u, set()) and x in adj.get(v, set()):
                    # Check x is adjacent to all in clique
                    all_adj = True
                    for y in clique:
                        if y not in adj.get(x, set()):
                            all_adj = False
                            break
                    if all_adj:
                        clique.add(x)
            
            # Remove all edges within this clique
            clique_list = sorted(clique)
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    rem = (a, b) if a < b else (b, a)
                    remaining_edges.discard(rem)
            
            cliques.append(clique_list)
        
        return cliques
    
    def solve_instance(n, k, p, seed):
        # Build the split graph
        rng = Random(seed)
        # C = {0, ..., k-1}, I = {k, ..., n-1}
        # Cross edges: for each i in I, for each u in C, edge with prob p
        cross_neighbors = {}  # i -> set of neighbors in C
        for i in range(k, n):
            nbrs = set()
            for u in range(k):
                if rng.random() < p:
                    nbrs.add(u)
            cross_neighbors[i] = nbrs
        
        # Step 1: For each i in I, create clique {i} ∪ N(i)
        cliques = []
        # Track which C-C edges are covered by cross cliques
        cc_covered = set()  # set of (u,v) with u<v, u,v < k
        for i in range(k, n):
            nbrs = cross_neighbors[i]
            if len(nbrs) >= 1:
                clique = [i] + sorted(nbrs)
                cliques.append(clique)
                # Mark C-C edges within nbrs as covered
                nbrs_list = sorted(nbrs)
                for a_idx in range(len(nbrs_list)):
                    for b_idx in range(a_idx + 1, len(nbrs_list)):
                        u, v = nbrs_list[a_idx], nbrs_list[b_idx]
                        cc_covered.add((u, v))
        
        # Step 2: Find remaining C-C edges
        remaining_cc_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in cc_covered:
                    remaining_cc_edges.add((u, v))
        
        # Build adjacency for remaining C-C graph
        if remaining_cc_edges:
            adj = {u: set() for u in range(k)}
            for u, v in remaining_cc_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Cover remaining C-C edges with cliques
            c_cliques = greedy_clique_cover(adj, list(range(k)))
            for c in c_cliques:
                if len(c) >= 2:
                    cliques.append(c)
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Emergency: return trivial solution (each edge as its own clique)
            # But this would be huge; just return what we have
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    
    # If we ran out of time, fill remaining with trivial (but this shouldn't happen for small k)
    while len(results) < 3:
        spec = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(results)]
        n, k, p, seed = spec
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        for i in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, i))
        results.append([list(e) for e in edges])
    
    # Ensure all are proper Python lists of lists of ints
    final = []
    for cliques in results:
        converted = []
        for c in cliques:
            converted.append([int(x) for x in c])
        final.append(converted)
    
    return final

# EVOLVE-BLOCK-END
