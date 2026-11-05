<<<<<<< SEARCH
def _greedy_clique_partition(adj, n, order):
    """
    Greedily partition edges into cliques.
    `order` is a permutation of vertices to try as seeds (high priority first).
    Returns a list of cliques (each a list of vertex indices, size >= 2).
    """
    # remaining edges as a set of frozensets for O(1) check
    edge_set = set()
    for u in range(n):
        for v in adj[u, :]:
            pass
    # Build edge set from adjacency
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edge_set.add((u, v))

    # remaining degree per vertex (number of incident remaining edges)
    rem_deg = [0] * n
    for (u, v) in edge_set:
        rem_deg[u] += 1
        rem_deg[v] += 1

    cliques = []

    # For fast "is (u,v) still an edge" we keep edge_set
    # We'll process vertices in priority order; but a vertex may be used in
    # multiple cliques. Instead, we repeatedly pick the best seed among vertices
    # that still have remaining edges, using the given priority (centrality).

    # Precompute a priority rank: lower rank = higher priority
    rank = {v: i for i, v in enumerate(order)}

    while edge_set:
        # choose seed: vertex with remaining edges and best (lowest) rank
        seed = None
        best_rank = None
        for v in range(n):
            if rem_deg[v] > 0:
                r = rank[v]
                if best_rank is None or r < best_rank:
                    best_rank = r
                    seed = v
        if seed is None:
            break

        # Build a clique starting from seed.
        clique = [seed]
        # Candidate neighbors: vertices adjacent to all current clique members,
        # with at least one remaining edge to the clique.
        # Greedily add the candidate that maximizes remaining edges to clique.
        while True:
            # candidates
            cands = []
            for v in range(n):
                if v in clique:
                    continue
                if rem_deg[v] == 0:
                    continue
                ok = True
                for c in clique:
                    if c < v:
                        e = (c, v)
                    else:
                        e = (v, c)
                    if e not in edge_set:
                        ok = False
                        break
                if ok:
                    cands.append(v)
            if not cands:
                break
            # pick candidate with max remaining degree (tie-break by rank)
            best = None
            best_key = None
            for v in cands:
                key = (rem_deg[v], -rank[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            clique.append(best)

        # Now remove all edges among clique members that still exist
        added = False
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                e = (u, v) if u < v else (v, u)
                if e in edge_set:
                    edge_set.remove(e)
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
                    added = True
        # Only record clique if it actually covered an edge (size >= 2 guaranteed
        # because seed had rem_deg>0 and we added at least one neighbor, or if
        # no neighbor was added we must not record a single-vertex clique).
        if len(clique) >= 2 and added:
            cliques.append(clique)

    return cliques
=======
def _greedy_clique_partition(adj, n, order):
    """
    Greedily partition edges into cliques.
    `order` is a permutation of vertices to try as seeds (high priority first).
    Returns a list of cliques (each a list of vertex indices, size >= 2).
    """
    # Build edge set from adjacency
    edge_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edge_set.add((u, v))

    # remaining degree per vertex (number of incident remaining edges)
    rem_deg = [0] * n
    for (u, v) in edge_set:
        rem_deg[u] += 1
        rem_deg[v] += 1

    cliques = []

    # Precompute a priority rank: lower rank = higher priority
    rank = {v: i for i, v in enumerate(order)}

    # For fast adjacency checks, keep a set-based representation
    # adj_set[v] = set of neighbors of v that still have an edge to v
    adj_set = [set() for _ in range(n)]
    for (u, v) in edge_set:
        adj_set[u].add(v)
        adj_set[v].add(u)

    while edge_set:
        # Find the vertex that is in the largest potential clique
        # Use: for each vertex with rem_deg > 0, compute the size of its
        # common neighborhood (intersection of adj_set of all its neighbors)
        # as a proxy for max clique size through that vertex.
        
        # Heuristic: pick the vertex v with rem_deg > 0 that maximizes
        # the number of edges in the subgraph induced by N(v) (its neighborhood).
        # This approximates the max clique size.
        
        best_seed = None
        best_score = -1
        
        for v in range(n):
            if rem_deg[v] == 0:
                continue
            # Score: number of edges among neighbors of v
            # (these edges could be part of a clique containing v)
            nbrs = list(adj_set[v])
            if not nbrs:
                continue
            # Count edges among nbrs
            edge_count = 0
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    a, b = nbrs[i], nbrs[j]
                    e = (a, b) if a < b else (b, a)
                    if e in edge_set:
                        edge_count += 1
            # Also factor in centrality rank
            score = edge_count * 1000 - rank[v]
            if score > best_score:
                best_score = score
                best_seed = v
        
        if best_seed is None:
            break
        
        seed = best_seed
        
        # Build a clique starting from seed using greedy expansion
        # that maximizes the number of edges covered
        clique = [seed]
        
        # Common neighborhood: vertices adjacent to all current clique members
        common_nbrs = set(adj_set[seed])
        
        while True:
            if not common_nbrs:
                break
            
            # For each candidate in common_nbrs, compute how many edges
            # it would add to the clique (i.e., edges from this candidate
            # to existing clique members that are in edge_set)
            # All such edges exist by construction (candidate is in common_nbrs)
            # So we want to pick the candidate that maximizes the future
            # potential: the size of the intersection of adj_set[candidate]
            # with common_nbrs (excluding the candidate itself)
            
            best_cand = None
            best_potential = -1
            
            for v in common_nbrs:
                # Potential: number of vertices in common_nbrs that are
                # also neighbors of v (can be added after v)
                potential = 0
                for u in common_nbrs:
                    if u == v:
                        continue
                    if u in adj_set[v]:
                        potential += 1
                # Tie-break by rank (lower rank = better)
                pot_key = (potential, -rank[v])
                if best_cand is None:
                    best_cand = v
                    best_potential = pot_key
                else:
                    if pot_key > best_potential:
                        best_cand = v
                        best_potential = pot_key
            
            if best_cand is None:
                break
            
            clique.append(best_cand)
            
            # Update common_nbrs: intersect with adj_set[best_cand]
            common_nbrs = common_nbrs & adj_set[best_cand]
            common_nbrs.discard(best_cand)
        
        # Now remove all edges among clique members that still exist
        added = False
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                e = (u, v) if u < v else (v, u)
                if e in edge_set:
                    edge_set.remove(e)
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
                    adj_set[u].discard(v)
                    adj_set[v].discard(u)
                    added = True
        
        # Only record clique if it actually covered an edge
        if len(clique) >= 2 and added:
            cliques.append(clique)

    return cliques
>>>>>>> REPLACE

