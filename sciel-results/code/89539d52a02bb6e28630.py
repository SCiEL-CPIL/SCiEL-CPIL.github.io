import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill

    n = 16

    # Q3 edges
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    # Precompute Q3 neighbor lists
    q3_neighbors = [[] for _ in range(8)]
    for u, v in q3_edges:
        q3_neighbors[u].append(v)
        q3_neighbors[v].append(u)

    # Precompute all 8-vertex subsets of 16 vertices
    # There are C(16,8) = 12870 subsets
    # For each subset, check if it can host a Q3 (i.e., the induced subgraph contains Q3)
    # Actually, we need to check if the graph induced by the subset IS a supergraph of Q3
    # i.e., contains all 12 edges of Q3 under some labeling

    # Better approach: Precompute for each 8-vertex subset S, whether the induced subgraph G[S]
    # contains a Q3 as a subgraph.
    # But this is expensive to precompute. Instead, let's use a faster checking method.

    # Alternative: Use the bitmask approach with a lookup table.
    # For a given graph (128-bit adjacency), check if Q3 exists.
    # We can enumerate all 8-vertex subsets and for each, check if Q3 is a subgraph.
    # C(16,8) = 12870 subsets. For each, we need to check 8! / 6 = 560 labelings? No,
    # the automorphism group of Q3 has size 48, so 8!/48 = 420 labelings to check.
    # 12870 * 420 = ~5.4 million checks, each checking 12 edges. That's ~65 million edge checks.
    # Too slow for repeated calls.

    # Better: Precompute a table indexed by the 8-vertex subset (as a bitmask) and the
    # induced edge pattern (12 bits for the 12 Q3 edges under a canonical labeling).
    # Actually, let's think differently.

    # Key insight from inspiration: Precompute a 2^28 lookup table? No, that's for something else.
    # Let me think about a faster Q3 check.

    # Fast Q3 check using backtracking with strong pruning:
    # Fix Q3 vertex 0 to each graph vertex g0 (16 choices).
    # Then backtrack on Q3 vertices 1-7.
    # With good ordering and pruning, this should be fast.

    def has_q3(adj):
        """
        adj: list of 16 ints, bit j set in adj[i] means edge (i,j) exists.
        Returns True if Q3 subgraph exists.
        """
        for g0 in range(n):
            assign = {0: g0}

            def backtrack(qv):
                if qv == 8:
                    return True

                used = set(assign.values())

                # Try all graph vertices not yet used
                for gv in range(n):
                    if gv in used:
                        continue

                    # Check compatibility with already assigned Q3 neighbors
                    ok = True
                    for nq in q3_neighbors[qv]:
                        if nq in assign:
                            gn = assign[nq]
                            if not (adj[gv] >> gn) & 1:
                                ok = False
                                break
                    if not ok:
                        continue

                    assign[qv] = gv
                    if backtrack(qv + 1):
                        return True
                    del assign[qv]

                return False

            if backtrack(1):
                return True

        return False

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible = len(all_edges)  # 120

    best_edges_count = 0
    best_adj = None

    # Strategy: Greedy edge addition with random restarts, plus local search improvement.

    def greedy_add(adj, edge_order, deadline):
        """Add edges in given order, skipping those that create Q3."""
        adj = adj[:]
        count = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            # Try adding edge (u,v)
            if (adj[u] >> v) & 1:
                continue  # already present
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if has_q3(adj):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                count += 1
        return adj, count

    def local_search(adj, count, deadline):
        """Try to add more edges by trying all missing edges."""
        adj = adj[:]
        count = count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if has_q3(adj):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    count += 1
                    improved = True
        return adj, count

    # Initial seed: K3,13 has 39 edges. Let's start from a dense random graph and repair.
    # Or start from empty and greedily add.

    # Try multiple random orders for greedy addition
    trials = 0
    while time.time() < deadline - 20:
        trials += 1
        random.seed(trials * 12345 + 67890)

        # Random edge order
        edge_order = all_edges[:]
        random.shuffle(edge_order)

        adj = [0] * n
        adj, count = greedy_add(adj, edge_order, deadline - 10)

        if count > best_edges_count:
            best_edges_count = count
            best_adj = adj[:]

        # If we have time, do local search on the best
        if best_adj is not None and time.time() < deadline - 5:
            adj_ls, count_ls = local_search(best_adj, best_edges_count, deadline - 2)
            if count_ls > best_edges_count:
                best_edges_count = count_ls
                best_adj = adj_ls

        # Also try starting from a dense random graph and removing Q3s
        if trials % 3 == 0 and time.time() < deadline - 15:
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.7:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Remove edges to make Q3-free
            count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            for _ in range(100):  # max 100 removals
                if time.time() > deadline - 5:
                    break
                if not has_q3(adj):
                    break
                # Find a Q3 and remove a random edge from it
                # For simplicity, just remove a random edge
                # This is inefficient but works
                # Better: find the actual Q3
                # Let's just remove random edges until Q3-free
                for u, v in all_edges:
                    if (adj[u] >> v) & 1:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        count -= 1
                        if not has_q3(adj):
                            break
                else:
                    break

            if count > best_edges_count:
                best_edges_count = count
                best_adj = adj[:]

        if trials > 100:
            break

    # Final local search on the best
    if best_adj is not None and time.time() < deadline - 1:
        adj_ls, count_ls = local_search(best_adj, best_edges_count, deadline - 0.5)
        if count_ls > best_edges_count:
            best_edges_count = count_ls
            best_adj = adj_ls

    if best_adj is None:
        return []

    # Convert adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges

construct()
