<<<<<<< SEARCH
    # Use a much simpler and faster approach:
    # For each edge (u,v), count the number of 8-sets containing u,v that have
    # the "right" structure to potentially contain Q3 using (u,v).
    # Use a fast bitset-based check.

    # Precompute: for each vertex, its adjacency bitmask
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                adj_mask[i] |= (1 << j)

    # For counting Q3 embeddings using edge (u,v):
    # We need to find 6 other vertices such that the 8-set contains Q3 using (u,v).
    # Key insight: in Q3, if u and v are adjacent, they have specific common neighbor
    # structures. Let's use the fact that Q3 is the cube and has a very specific structure.

    # For a Q3 embedding where u and v are adjacent Q3 vertices:
    # - u has 2 other Q3-neighbors (besides v)
    # - v has 2 other Q3-neighbors (besides u)
    # - There are 4 more vertices completing the cube

    # Let me use a different strategy: instead of counting exact Q3 embeddings,
    # use a proxy that's fast to compute and correlates with Q3 count.

    # Proxy: for edge (u,v), count the number of 4-cycles containing (u,v) in G.
    # A Q3 contains many 4-cycles, so this is a reasonable proxy.

    # Actually, let me just use the original approach but with a much faster
    # Q3 detection using precomputed patterns.

    # Precompute all "Q3 edge patterns" for 8-sets:
    # For each 8-subset S (sorted), and for each pair of positions (i,j),
    # precompute the list of "completion patterns" that would make S contain Q3
    # using edge (S[i], S[j]).

    # A completion pattern is a 6x6 boolean matrix where entry (k,l) indicates
    # whether S[remaining_pos[k]] must be adjacent to S[remaining_pos[l]] in G.

    # This is getting too complex. Let me just use a simple heuristic:
    # Delete edges in order of decreasing (deg(u) * deg(v)), which correlates
    # with the number of 4-cycles and thus Q3s.

    # Start with K16
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                adj_mask[i] |= (1 << j)

    # Precompute all pairs
    all_pairs = list(itertools.combinations(range(n), 2))

    # Greedy deletion: repeatedly delete the edge that is in the most Q3s
    # Approximate Q3 count using: for edge (u,v), count 4-cycles through (u,v)
    # A 4-cycle through (u,v) is a path u-a-v-b-u where a,b are other vertices.

    def count_4cycles_through_edge(u, v, adj_mask):
        """Count 4-cycles containing edge (u,v)."""
        # A 4-cycle u-a-v-b-u requires:
        # - a adjacent to u and v
        # - b adjacent to u and v
        # - a and b not adjacent to each other (for a simple 4-cycle)
        # Actually, for counting Q3s, we don't need simple 4-cycles.
        # Let me count pairs (a,b) such that a~u, a~v, b~u, b~v, a!=b
        common = adj_mask[u] & adj_mask[v]
        common &= ~(1 << u) & ~(1 << v)
        # Number of ways to choose 2 distinct vertices from common
        deg_common = bin(common).count('1')
        return deg_common * (deg_common - 1) // 2

    # For each edge, compute the 4-cycle count and delete the max
    edges_to_delete = []
    for u, v in all_pairs:
        cnt = count_4cycles_through_edge(u, v, adj_mask)
        edges_to_delete.append((cnt, u, v))

    # Sort by count descending
    edges_to_delete.sort(reverse=True)

    # Delete edges one by one, re-evaluating after each deletion
    # But this is expensive. Let me just delete the top-k edges at once.

    # Actually, let me use a proper greedy approach:
    # At each step, find the edge with the highest Q3 count (approximated by 4-cycles),
    # delete it, and repeat.

    # To speed up, I'll batch the deletions.

    deleted = set()
    max_steps = 500  # Limit number of deletions to stay within time budget

    for step in range(max_steps):
        if time.time() > deadline - 10:
            break

        # Compute 4-cycle count for each remaining edge
        best_edge = None
        best_count = -1
        for u, v in all_pairs:
            if (u, v) in deleted or (v, u) in deleted:
                continue
            cnt = count_4cycles_through_edge(u, v, adj_mask)
            if cnt > best_count:
                best_count = cnt
                best_edge = (u, v)

        if best_edge is None:
            break

        # Delete the best edge
        u, v = best_edge
        deleted.add((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

        # If no more 4-cycles through any edge, we might be Q3-free
        # But let's continue for a few more steps to be safe

    # Build final edge list
    final_edges = []
    for u, v in all_pairs:
        if (u, v) not in deleted and (v, u) not in deleted:
            final_edges.append([u, v])

    return final_edges
=======
    # Use a proper greedy deletion with exact Q3 counting using bitsets.
    # Key optimization: precompute for each 8-subset the "Q3 signature" --
    # a set of edge-bitmasks that correspond to Q3 embeddings.

    # For an 8-subset S = (s0,...,s7), a Q3 embedding is a permutation pi of 0..7
    # such that all 12 Q3 edges map to edges in G[S].
    # The 12 edges as a bitmask over the 28 possible edges in S.

    # Precompute for each 8-subset the list of all 40320 "potential Q3 edge bitmasks"
    # (one for each permutation). Then, for the current graph, a permutation is
    # "active" if all 12 edges in its bitmask are present in G.

    # For each edge (u,v), the number of active Q3 embeddings containing it is
    # the number of active permutations where the Q3 edge corresponding to (u,v)
    # is present.

    # This is still expensive. Let me use a different approach:

    # For each 8-subset S, precompute a 40320-bit vector where bit i indicates
    # whether permutation i is a valid Q3 embedding in the current graph.
    # When we delete an edge, we update this vector for all 8-subsets containing
    # that edge.

    # But 12870 8-subsets * 40320 bits = 12870 * 5040 bytes = 64 MB. Manageable.

    # Let me implement this.

    # Start with K16
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                adj_mask[i] |= (1 << j)

    all_pairs = list(itertools.combinations(range(n), 2))

    # Precompute all 8-subsets
    all_8subsets = list(itertools.combinations(range(n), 8))

    # For each 8-subset, precompute the 40320 permutations and their edge bitmasks
    # A permutation pi maps Q3 vertex k to S[pi[k]].
    # The 12 Q3 edges become 12 pairs (S[pi[a]], S[pi[b]]) for (a,b) in q3_edges.
    # Represent as a bitmask over the 28 edges in S.

    # To save memory and time, let me precompute for each 8-subset:
    # - The list of 40320 edge bitmasks (as Python integers)
    # - For each of the 12 Q3 edges, which permutations have that Q3 edge mapping
    #   to a specific pair of S-vertices.

    # This is getting very complex. Let me just use the 4-cycle proxy approach
    # but with a better heuristic.

    # New approach: use the fact that Q3 is 3-regular and has girth 4.
    # A necessary condition for Q3-free: no 8-vertex induced subgraph is Q3.
    # But we need subgraph, not induced.

    # Let me just use the 4-cycle count as a proxy and do greedy deletion.
    # This is fast and should give a reasonable answer.

    def count_4cycles_through_edge(u, v, adj_mask):
        common = adj_mask[u] & adj_mask[v]
        common &= ~(1 << u) & ~(1 << v)
        deg_common = bin(common).count('1')
        return deg_common * (deg_common - 1) // 2

    deleted = set()
    max_steps = 1000

    for step in range(max_steps):
        if time.time() > deadline - 15:
            break

        best_edge = None
        best_count = -1
        for u, v in all_pairs:
            if (u, v) in deleted:
                continue
            cnt = count_4cycles_through_edge(u, v, adj_mask)
            if cnt > best_count:
                best_count = cnt
                best_edge = (u, v)

        if best_edge is None or best_count == 0:
            break

        u, v = best_edge
        deleted.add((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    final_edges = []
    for u, v in all_pairs:
        if (u, v) not in deleted:
            final_edges.append([u, v])

    return final_edges
>>>>>>> REPLACE

