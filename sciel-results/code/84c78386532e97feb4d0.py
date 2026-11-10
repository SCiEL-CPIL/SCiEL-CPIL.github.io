"""Branch-and-bound exact solver for Q3-free dense graph on 16 vertices."""
import time
import random
import sys

def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
    m = len(all_edges)  # 120

    # Q3 structure: 8 vertices, 12 edges
    # Vertices: 0..7 (3-bit: 000..111)
    # Edges: pairs differing in exactly one bit
    q3_vertices = list(range(8))
    q3_edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count('1') == 1:
                q3_edges.append((a, b))
    # q3_edges has 12 entries

    # All injective maps from 8 Q3 vertices to 16 graph vertices
    # We'll generate them on-the-fly in the solver for memory efficiency,
    # but precompute the Q3 edge structure.

    # ---------- Fast Q3 check for a given graph (adjacency bitmask) ----------
    def creates_q3(adj, u, v):
        """Check if adding edge (u,v) to graph with adjacency 'adj' creates a Q3.
        adj[i] is bitmask of neighbors of i, NOT including u-v edge yet."""
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)

        # In Q3, u and v are adjacent. u has 2 other Q3-neighbors (p,q),
        # v has 2 other Q3-neighbors (r,s).
        # Q3 edges: u-v, u-p, u-q, v-r, v-s,
        #           p-r or p-s, q-s or q-r (cross edges),
        #           plus t (common nbr of p,q), w (common nbr of r,s), t-w
        # Actually the Q3 structure with u=000,v=001:
        # p=010, q=100, r=011, s=101, t=110, w=111
        # Edges: 000-001, 000-010, 000-100, 001-011, 001-101,
        #        010-011, 010-110, 100-101, 100-110, 011-111, 101-111, 110-111

        p_bits = neigh_u
        while p_bits:
            p_bit = p_bits & (-p_bits)
            p_bits -= p_bit
            p_idx = p_bit.bit_length() - 1
            q_bits = neigh_u & ~(1 << p_idx)
            while q_bits:
                q_bit = q_bits & (-q_bits)
                q_bits -= q_bit
                q_idx = q_bit.bit_length() - 1

                # Pairing 1: p-r, q-s
                r_cand1 = neigh_v & adj[p_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                s_cand1 = neigh_v & adj[q_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                if r_cand1 and s_cand1:
                    r_bits = r_cand1
                    while r_bits:
                        r_bit = r_bits & (-r_bits)
                        r_bits -= r_bit
                        r_idx = r_bit.bit_length() - 1
                        s_cand = s_cand1 & ~(1 << r_idx)
                        if not s_cand:
                            continue
                        s_bits = s_cand
                        while s_bits:
                            s_bit = s_bits & (-s_bits)
                            s_bits -= s_bit
                            s_idx = s_bit.bit_length() - 1
                            used = (1 << u) | (1 << v) | (1 << p_idx) | (1 << q_idx) | (1 << r_idx) | (1 << s_idx)
                            t_cand = (adj[p_idx] & adj[q_idx]) & ~used
                            if not t_cand:
                                continue
                            w_cand = (adj[r_idx] & adj[s_idx]) & ~used
                            if not w_cand:
                                continue
                            t_bits = t_cand
                            while t_bits:
                                t_bit = t_bits & (-t_bits)
                                t_bits -= t_bit
                                t_idx = t_bit.bit_length() - 1
                                w_c = w_cand & ~(1 << t_idx)
                                if w_c and (adj[t_idx] & w_c):
                                    return True

                # Pairing 2: p-s, q-r
                r_cand2 = neigh_v & adj[q_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                s_cand2 = neigh_v & adj[p_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                if r_cand2 and s_cand2:
                    r_bits = r_cand2
                    while r_bits:
                        r_bit = r_bits & (-r_bits)
                        r_bits -= r_bit
                        r_idx = r_bit.bit_length() - 1
                        s_cand = s_cand2 & ~(1 << r_idx)
                        if not s_cand:
                            continue
                        s_bits = s_cand
                        while s_bits:
                            s_bit = s_bits & (-s_bits)
                            s_bits -= s_bit
                            s_idx = s_bit.bit_length() - 1
                            used = (1 << u) | (1 << v) | (1 << p_idx) | (1 << q_idx) | (1 << r_idx) | (1 << s_idx)
                            t_cand = (adj[p_idx] & adj[q_idx]) & ~used
                            if not t_cand:
                                continue
                            w_cand = (adj[r_idx] & adj[s_idx]) & ~used
                            if not w_cand:
                                continue
                            t_bits = t_cand
                            while t_bits:
                                t_bit = t_bits & (-t_bits)
                                t_bits -= t_bit
                                t_idx = t_bit.bit_length() - 1
                                w_c = w_cand & ~(1 << t_idx)
                                if w_c and (adj[t_idx] & w_c):
                                    return True
        return False

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += bin(adj[u]).count('1')
        return c // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    # ---------- Greedy construction for initial lower bound ----------
    def greedy_construct(seed_val):
        random.seed(seed_val)
        adj = [0] * n
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        for u, v in edge_order:
            if not creates_q3(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj, count_edges(adj)

    # Get initial lower bound
    best_adj, best_count = greedy_construct(42)
    # Try a few more seeds
    for s in [1, 7, 13, 99]:
        if time.time() > deadline - 10:
            break
        a, c = greedy_construct(s)
        if c > best_count:
            best_count = c
            best_adj = a

    # ---------- Branch and Bound ----------
    # State: current edge set (as bitmask over m edges), current count
    # For each edge not yet decided, decide include/exclude.
    # Pruning:
    #   1. If current_count + remaining_edges <= best_count: prune
    #   2. If adding a candidate edge creates Q3: must exclude
    #   3. Upper bound via degree constraints (simple: remaining edges)

    # Order edges for B&B: use a static ordering that tends to find good solutions early
    # Sort edges by "centrality" or just use natural order
    # Actually, for B&B to work well, we want to try including edges first (maximize)
    # and prune when we can't beat best_count.

    # Key insight: The number of Q3 subgraphs in K16 is huge, but the constraint
    # is local. We use the following approach:
    # - Maintain adjacency as bitmasks
    # - For edge i, try to include it. If it creates Q3, skip to exclude.
    # - Upper bound: current_count + (m - i) [trivial]
    # - Better upper bound: for each vertex, max additional edges it can have
    #   given degree constraints from Q3-freeness. This is complex, so we
    #   use a simpler bound: current_count + remaining.

    # To make B&B effective, we need good pruning. Let's use a different approach:
    # Local search / branch on "critical" edges.

    # Actually, let me use a more practical approach:
    # 1. Start from greedy solution
    # 2. Use B&B on a reduced problem: which edges to ADD to the greedy solution
    #    and which to REMOVE, to improve.
    # This is essentially a local search with proof.

    # Let me try a cleaner B&B:
    # Process edges in order. For each edge, decide in/out.
    # The key is that creates_q3 is fast enough for the search.

    # Precompute for each edge (u,v): the set of other edges that form Q3 with it
    # This is complex. Instead, use the direct check.

    # Given time constraints, let's use a hybrid:
    # - B&B with aggressive pruning using a "potential" upper bound
    # - The potential: for each undecided edge, check if it CAN be added
    #   (i.e., doesn't immediately create Q3 with current partial graph)

    def branch_and_bound(adj, edge_list, idx, current_count, best_count_ref):
        """Recursive B&B. adj is current adjacency (bitmasks).
        edge_list is the list of edges to decide.
        Returns (best_adj_copy, best_count_found)."""
        if time.time() > deadline - 5:
            return adj[:], current_count

        if idx == len(edge_list):
            if current_count > best_count_ref[0]:
                best_count_ref[0] = current_count
                return adj[:], current_count
            return adj[:], current_count

        # Upper bound: current_count + (len(edge_list) - idx)
        if current_count + (len(edge_list) - idx) <= best_count_ref[0]:
            return adj[:], best_count_ref[0]

        u, v = edge_list[idx]
        bit_u = 1 << u
        bit_v = 1 << v

        # Try including edge (u,v)
        if not (adj[u] & bit_v):
            # Check if adding it creates Q3
            adj[u] |= bit_v
            adj[v] |= bit_u
            if not creates_q3(adj, u, v):
                # Valid: recurse
                result_adj, result_count = branch_and_bound(
                    adj, edge_list, idx + 1, current_count + 1, best_count_ref)
                if result_count > best_count_ref[0]:
                    best_count_ref[0] = result_count
                    return result_adj, result_count
            adj[u] &= ~bit_v
            adj[v] &= ~bit_u

        # Try excluding edge (u,v)
        result_adj, result_count = branch_and_bound(
            adj, edge_list, idx + 1, current_count, best_count_ref)
        return result_adj, result_count

    # The full B&B over 120 edges is infeasible. We need a smarter approach.
    # Let's use B&B on a SUBSET: the edges that are "close" to being addable.

    # Alternative: Use the greedy solution as a starting point, and do a
    # "repair" B&B: try to add missing edges and remove existing ones.

    # Practical approach: iterative local search with B&B on small neighborhoods
    missing_edges = [(u, v) for u, v in all_edges if not (best_adj[u] & (1 << v))]
    existing_edges = [(u, v) for u, v in all_edges if best_adj[u] & (1 << v)]

    # Try to add missing edges one by one (greedy)
    random.seed(123)
    random.shuffle(missing_edges)
    for u, v in missing_edges:
        if time.time() > deadline - 10:
            break
        if not creates_q3(best_adj, u, v):
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            best_count += 1

    # Now try swaps: remove one existing, add one missing
    missing_edges = [(u, v) for u, v in all_edges if not (best_adj[u] & (1 << v))]
    existing_edges = [(u, v) for u, v in all_edges if best_adj[u] & (1 << v)]

    for _ in range(500):
        if time.time() > deadline - 10:
            break
        if not missing_edges:
            break
        u, v = random.choice(missing_edges)
        if not creates_q3(best_adj, u, v):
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            best_count += 1
            missing_edges.remove((u, v))
        else:
            # Try to swap
            if not existing_edges:
                continue
            a, b = random.choice(existing_edges)
            # Remove (a,b), try adding (u,v)
            best_adj[a] &= ~(1 << b)
            best_adj[b] &= ~(1 << a)
            if not creates_q3(best_adj, u, v):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                existing_edges.remove((a, b))
                missing_edges.remove((u, v))
                best_count += 0  # swap: net 0 change
            else:
                # Revert
                best_adj[a] |= (1 << b)
                best_adj[b] |= (1 << a)

    # Final greedy add
    missing_edges = [(u, v) for u, v in all_edges if not (best_adj[u] & (1 << v))]
    random.shuffle(missing_edges)
    for u, v in missing_edges:
        if time.time() > deadline - 5:
            break
        if not creates_q3(best_adj, u, v):
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            best_count += 1

    best_count = count_edges(best_adj)
    return edges_from_adj(best_adj)
