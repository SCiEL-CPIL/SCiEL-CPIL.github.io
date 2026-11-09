import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        # For each independent set vertex v, its neighbors in C
        N_v = {v: adj[v] & set(C) for v in I}

        # We need to cover:
        # 1. All cross edges (v, u) where v in I, u in C, edge exists
        # 2. All edges within C
        #
        # Strategy: For each v in I, we form cliques that include v and some subset of N_v.
        # These cliques also cover edges within C among the chosen subset.
        # After processing all v in I, remaining edges in C must be covered by cliques entirely in C.
        #
        # Key insight: We want to maximize the number of edges in C covered by cross-cliques,
        # to minimize the remaining edges in C that need separate cliques.
        #
        # Approach:
        # For each v in I, we need to partition N_v into cliques (in the remaining graph of C).
        # The coloring approach: color the "conflict graph" where edges represent already-covered C-edges.
        # But a better approach: greedily pick large cliques in N_v that haven't been "used" yet.
        #
        # Let's use a different strategy:
        # Process vertices in I in some order. For each v, find a maximum clique in the
        # subgraph of C induced by N_v, restricted to edges not yet covered.
        # Add v to that clique. Mark all edges in the clique as covered.
        #
        # But finding max clique is NP-hard. For small k (10, 20, 30), we can use heuristics.
        #
        # Better approach: Use the fact that C is a clique. So any subset of C is a clique.
        # The constraint is that edges within C that are already covered by previous cliques
        # cannot be in a new clique (each edge must be in exactly one clique).
        #
        # So for vertex v in I, we need to find a subset S of N_v such that no edge in S
        # has been covered yet. Then S ∪ {v} is a valid clique.
        #
        # To maximize efficiency, we want S to be as large as possible.
        # This is equivalent to finding a maximum clique in the graph G_v where:
        # - Vertices are N_v
        # - Edge (u, w) exists if u-w edge in C is NOT yet covered
        #
        # Since k is small (≤30), we can use a simple greedy or even exact approach for small cases.

        # Track which edges in C are covered
        # Use a set of frozensets for covered edges in C
        covered_C_edges = set()

        # Order of processing I vertices
        # Try multiple orderings and keep the best result
        best_partition = None
        best_count = float('inf')

        # Orderings to try
        orderings = []

        # 1. Descending degree in C
        order_desc = sorted(I, key=lambda v: len(N_v[v]), reverse=True)
        orderings.append(order_desc)

        # 2. Ascending degree in C
        order_asc = sorted(I, key=lambda v: len(N_v[v]))
        orderings.append(order_asc)

        # 3. Random
        rng = Random(seed + 1000)
        order_rand = I[:]
        rng.shuffle(order_rand)
        orderings.append(order_rand)

        for order in orderings:
            # Reset covered edges
            covered_C_edges_local = set()
            partition = []

            for v in order:
                if not N_v[v]:
                    continue

                # Find a large clique in the remaining graph on N_v[v]
                # Vertices: N_v[v]
                # Edge (u, w) if (u, w) not in covered_C_edges_local
                candidates = list(N_v[v])

                # Greedy: repeatedly pick the vertex with max degree in remaining subgraph
                clique = []
                remaining = set(candidates)

                while remaining:
                    # Pick vertex with max degree in the subgraph induced by remaining
                    best_v = None
                    best_deg = -1
                    for u in remaining:
                        deg = 0
                        for w in remaining:
                            if u != w:
                                e = (min(u, w), max(u, w))
                                if e not in covered_C_edges_local:
                                    deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = u

                    if best_v is None:
                        break

                    # Check if best_v can connect to all current clique members
                    can_add = True
                    for x in clique:
                        e = (min(best_v, x), max(best_v, x))
                        if e in covered_C_edges_local:
                            can_add = False
                            break

                    if can_add:
                        clique.append(best_v)
                        remaining.discard(best_v)
                    else:
                        # Can't add to current clique, skip for now
                        # Actually, we should just not add it to this clique
                        remaining.discard(best_v)

                if len(clique) >= 1:
                    # Form clique with v
                    clique_list = sorted(clique + [v])
                    partition.append(clique_list)

                    # Mark all edges in clique (within C) as covered
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            u = clique[i]
                            w = clique[j]
                            e = (min(u, w), max(u, w))
                            covered_C_edges_local.add(e)

            # Cover remaining edges in C
            remaining_edges = []
            for u in range(k):
                for v in range(u + 1, k):
                    if (u, v) not in covered_C_edges_local:
                        remaining_edges.append((u, v))

            if remaining_edges:
                # Greedy clique cover for remaining edges
                rem_adj = [set() for _ in range(k)]
                for u, v in remaining_edges:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)

                rem_edge_set = set(remaining_edges)

                while rem_edge_set:
                    # Find a maximal clique
                    # Pick vertex with max degree in remaining graph
                    max_deg_v = -1
                    max_deg = -1
                    for u in range(k):
                        if len(rem_adj[u]) > max_deg:
                            max_deg = len(rem_adj[u])
                            max_deg_v = u

                    if max_deg_v == -1:
                        break

                    clique = {max_deg_v}
                    candidates = set(rem_adj[max_deg_v])

                    # Greedy extension
                    while candidates:
                        # Pick candidate with max degree in the subgraph induced by candidates
                        best_cand = None
                        best_cand_deg = -1
                        for c in candidates:
                            deg_in_cands = len(rem_adj[c] & candidates)
                            if deg_in_cands > best_cand_deg:
                                best_cand_deg = deg_in_cands
                                best_cand = c

                        if best_cand is None:
                            break

                        # Check if best_cand is connected to all in clique
                        can_add = True
                        for x in clique:
                            if x not in rem_adj[best_cand]:
                                can_add = False
                                break
                        if can_add:
                            clique.add(best_cand)
                            candidates.discard(best_cand)
                            candidates = candidates & rem_adj[best_cand]
                        else:
                            candidates.discard(best_cand)

                    if len(clique) >= 2:
                        partition.append(sorted(list(clique)))

                    # Remove edges of this clique
                    clique_list = list(clique)
                    for i in range(len(clique_list)):
                        u = clique_list[i]
                        for j in range(i + 1, len(clique_list)):
                            v = clique_list[j]
                            e = (min(u, v), max(u, v))
                            if e in rem_edge_set:
                                rem_edge_set.remove(e)
                                rem_adj[u].remove(v)
                                rem_adj[v].remove(u)

            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = partition

        return best_partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)

    return results
