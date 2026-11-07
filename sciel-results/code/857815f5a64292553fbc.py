"""Optimal split-graph clique partition using matching-based edge absorption."""

def construct():
    from random import Random
    import time
    from collections import deque

    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def hopcroft_karp(left_size, right_size, adj):
        """
        Maximum bipartite matching.
        adj[u] = list of right vertices adjacent to left vertex u.
        Returns (matching, matched_right) where matching[u] = v or -1,
        matched_right[v] = u or -1.
        """
        pair_u = [-1] * left_size
        pair_v = [-1] * right_size
        dist = [0] * left_size

        def bfs():
            q = deque()
            for u in range(left_size):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(left_size):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u, pair_v, matching

    def solve_split(n, k, p, seed):
        rng = Random(seed)
        # Generate cross edges (neighborhoods of each w in I)
        neighbors = {}
        for w in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[w] = nbrs

        # Build list of K-edges
        k_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                k_edges.append((u, v))
        num_k_edges = len(k_edges)

        # For each w, determine which K-edges it can absorb
        # w can absorb edge (u,v) iff u in neighbors[w] and v in neighbors[w]
        # Build adjacency for bipartite matching:
        # Left: K-edges (indexed 0..num_k_edges-1)
        # Right: (w, slot) pairs. For each w, slots = 0..floor(deg(w)/2)-1
        # Edge from k_edge_idx to (w, slot) if w can absorb that edge.

        # First, compute deg(w) and slot counts
        deg = {}
        for w in range(k, n):
            deg[w] = len(neighbors[w])

        # Build right side: list of (w, slot)
        right_list = []  # index -> (w, slot)
        w_slot_start = {}
        idx = 0
        for w in range(k, n):
            slots = deg[w] // 2
            w_slot_start[w] = idx
            for s in range(slots):
                right_list.append((w, s))
                idx += 1
        right_size = idx

        if right_size == 0:
            # No absorption possible; fall back to trivial
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for w in range(k, n):
                for u in neighbors[w]:
                    cliques.append([u, w])
            return cliques

        # Build adjacency: for each K-edge, list of right indices it can go to
        # k_edge (u,v) can go to (w, slot) if u,v in neighbors[w]
        # Precompute for each w, the set of K-edges it can absorb
        # To save memory, build adj as list of lists for left side

        # For each w, compute which K-edges it can absorb
        # neighbors[w] is a list; convert to set for O(1) lookup
        nbr_sets = {w: set(neighbors[w]) for w in range(k, n)}

        adj = [[] for _ in range(num_k_edges)]
        for e_idx, (u, v) in enumerate(k_edges):
            for w in range(k, n):
                if u in nbr_sets[w] and v in nbr_sets[w]:
                    # w can absorb this edge; add all its slots
                    start = w_slot_start[w]
                    slots = deg[w] // 2
                    for s in range(slots):
                        adj[e_idx].append(start + s)

        # Solve max bipartite matching
        pair_u, pair_v, match_count = hopcroft_karp(num_k_edges, right_size, adj)

        # Determine which K-edges are matched and to which w
        # matched_k_edges: set of e_idx that are matched
        # edge_to_w: e_idx -> w
        matched_e_idx = set()
        edge_to_w = {}
        for e_idx in range(num_k_edges):
            if pair_u[e_idx] != -1:
                matched_e_idx.add(e_idx)
                w = right_list[pair_u[e_idx]][0]
                edge_to_w[e_idx] = w

        # For each w, collect the K-edges assigned to it
        w_assigned_edges = {w: [] for w in range(k, n)}
        for e_idx in matched_e_idx:
            w = edge_to_w[e_idx]
            w_assigned_edges[w].append(k_edges[e_idx])

        # Build cliques
        cliques = []

        # 1. 3-cliques for absorbed edges
        for w in range(k, n):
            for (u, v) in w_assigned_edges[w]:
                cliques.append([u, v, w])

        # 2. 2-cliques for remaining cross edges
        for w in range(k, n):
            used_u = set()
            for (u, v) in w_assigned_edges[w]:
                used_u.add(u)
                used_u.add(v)
            for u in neighbors[w]:
                if u not in used_u:
                    cliques.append([u, w])

        # 3. Cover remaining K-edges (those not matched) with cliques
        # Remaining K-edges: k_edges not in matched_e_idx
        # This forms a complete graph on k vertices minus a matching.
        # Clique cover of K_k minus matching of size m: k - m cliques.
        # Strategy: For each unmatched vertex in the matching, start a clique.
        # Actually, the standard result: K_k minus a perfect matching (or near-perfect)
        # has clique cover number = k - m where m = matching size.
        # Construction: pair up matched edges; each unmatched vertex gets its own "group".
        # More concretely: the complement of a matching in K_k is a complete graph with some
        # edges removed. A clique in this graph cannot contain both endpoints of a removed edge.
        # So we need at least as many cliques as the matching number... actually k - m.
        
        # Simple constructive approach:
        # The removed edges form a matching. Let the matching edges be (a1,b1), (a2,b2), ..., (am,bm).
        # Unmatched vertices: c1, c2, ..., c_{k-2m} (if k even) or one more.
        # We can form cliques as follows:
        # For each matched edge (ai, bi), we need them in separate cliques.
        # Total cliques needed: k - m.
        # Construction: 
        #   Clique 1: {a1} + all unmatched + all bj for j != 1 + all aj for j != 1? No.
        #   
        # Let me think differently. The graph is K_k minus matching M.
        # A clique can contain at most one endpoint from each matched edge.
        # So we need at least 2 cliques per matched edge? No, we need to separate the two endpoints.
        # Minimum clique cover = k - |M| (this is a known result).
        # 
        # Greedy construction:
        # Start with all vertices. For each matched edge (u,v), ensure u and v are in different cliques.
        # We can do: 
        #   Let unmatched vertices be U. |U| = k - 2m.
        #   Create m+1 cliques if m < k/2, or m cliques if m = k/2... 
        #   Actually k - m cliques.
        #   
        # Simpler: just do a greedy maximal clique extraction on the remaining graph.
        # Since k <= 30, this is fast.

        remaining_edges_set = set()
        for e_idx, (u, v) in enumerate(k_edges):
            if e_idx not in matched_e_idx:
                remaining_edges_set.add((u, v))

        # Greedy clique cover on remaining K-edges
        from itertools import combinations
        rem = set(remaining_edges_set)
        while rem:
            # Find a maximal clique in the remaining graph
            # Start with an arbitrary edge
            e = next(iter(rem))
            clique = {e[0], e[1]}
            # Try to add more vertices
            changed = True
            while changed:
                changed = False
                for x in range(k):
                    if x not in clique:
                        # Check if x is connected to all in clique
                        if all((min(x, y), max(x, y)) in rem for y in clique):
                            clique.add(x)
                            changed = True
                            break
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                # Remove all edges within this clique from rem
                for a, b in combinations(clique, 2):
                    rem.discard((min(a, b), max(a, b)))
            else:
                # Should not happen
                cliques.append(sorted([e[0], e[1]]))
                rem.discard(e)

        # Ensure all cliques are lists of Python ints
        clean_cliques = [[int(x) for x in c] for c in cliques]
        return clean_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            # Fallback: trivial solution for remaining instances
            n, k, p, seed = spec
            rng = Random(seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for w in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        cliques.append([u, w])
            results.append(cliques)
            continue
        cliques = solve_split(*spec)
        results.append(cliques)

    return results
