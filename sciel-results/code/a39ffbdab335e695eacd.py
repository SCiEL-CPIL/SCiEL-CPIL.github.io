"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    n = 40

    def time_ok():
        return time.time() < deadline

    def time_left():
        return deadline - time.time()

    # Bitset-based adjacency
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    def would_create_c6_fast(adj, u, v):
        """Fast bitset check: does adding edge (u,v) create a C6?
        A C6 using edge (u,v): u-a-b-c-d-v, all 6 distinct.
        Equivalent: there exist a in N(u)\\{v}, d in N(v)\\{u}, a!=d,
        and a path a-b-c-d of length 3 in G\\{u,v}.
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)

        # Iterate over a in N(u)\{v}
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a

            # B_base = N(a) excluding u, v
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)

            mask_d = Nv & ~(1 << a)  # d != a
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d

                # B = N(a) \ {u,v,d}
                B = Na_base & ~(1 << d)
                if B == 0:
                    continue

                # C = N(d) \ {u,v,a}
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue

                # For each b in B, check if N(b) & C & ~(1<<b) != 0
                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b

                    # Check if any neighbor of b is in C and is not b itself
                    if adj[b] & C & ~(1 << b):
                        return True

        return False

    # Spectral-guided initialization
    def spectral_init(seed_val=42):
        """Create a random 4-regular graph, compute eigenvector, use it to guide edge selection."""
        rng = np.random.default_rng(seed_val)

        # Create a random 4-regular graph on 40 vertices
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            candidates = list(range(n))
            candidates.remove(i)
            rng.shuffle(candidates)
            # Pick 4 neighbors (ensuring symmetry)
            for j in candidates[:4]:
                if adj_matrix[i][j] == 0:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
            # If degree < 4, fill in
            deg = adj_matrix[i].sum()
            if deg < 4:
                for j in range(n):
                    if i != j and adj_matrix[i][j] == 0:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1
                        deg += 1
                        if deg >= 4:
                            break

        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)

        # Find eigenvalue closest to sqrt(5) ≈ 2.236
        target = np.sqrt(5)
        idx = np.argmin(np.abs(eigenvalues - target))
        spec_vec = eigenvectors[:, idx]

        # Use spectral components to guide edge inclusion
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                weights[i][j] = abs(spec_vec[i] * spec_vec[j])

        # Get all potential edges sorted by weight (descending)
        potential_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                potential_edges.append((weights[i][j], i, j))
        potential_edges.sort(reverse=True)

        return potential_edges, adj_matrix

    # Greedy addition using fast C6 check
    def greedy_add(adj, edge_set, edges, candidate_edges):
        """Add all non-conflicting edges from candidate_edges."""
        added_any = True
        while added_any and time_ok():
            added_any = False
            remaining = []
            for item in candidate_edges:
                if len(item) == 2:
                    u, v = item
                else:
                    u, v = item[1], item[2]
                if (min(u, v), max(u, v)) in edge_set:
                    continue
                if would_create_c6_fast(adj, u, v):
                    remaining.append(item)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u, v), max(u, v)))
                    edges.append([u, v])
                    added_any = True
            candidate_edges = remaining
        return edges

    def local_search_with_swaps(adj, edge_set, edges, max_iters=100):
        """Local search with edge swaps to escape local maxima.
        Each iteration: try to add an edge; if stuck, try a 1-swap (remove one, add another).
        """
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        no_improve_count = 0

        for _ in range(max_iters):
            if time_left() < 5:
                break

            added = False

            # Try to add a random non-edge
            for _ in range(40):
                if time_left() < 5:
                    break
                u, v = random.choice(all_pairs)
                if adj[u] & (1 << v):
                    continue

                if not would_create_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u, v), max(u, v)))
                    edges.append([u, v])
                    added = True
                    no_improve_count = 0
                    break

            if not added:
                # Try a 1-swap: remove one edge, add another
                if no_improve_count > 5 and len(edge_set) > 0:
                    swapped = False
                    edge_list = list(edge_set)
                    random.shuffle(edge_list)
                    for (ru, rv) in edge_list[:10]:
                        if time_left() < 5:
                            break
                        # Temporarily remove (ru, rv)
                        adj[ru] &= ~(1 << rv)
                        adj[rv] &= ~(1 << ru)
                        edge_set.remove((ru, rv))
                        edges = [e for e in edges
                                 if not (min(e[0], e[1]) == min(ru, rv) and
                                         max(e[0], e[1]) == max(ru, rv))]

                        # Try to add some random non-edges
                        added_after_swap = False
                        for _ in range(20):
                            if time_left() < 5:
                                break
                            u, v = random.choice(all_pairs)
                            if (min(u, v), max(u, v)) in edge_set:
                                continue
                            if not would_create_c6_fast(adj, u, v):
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                edge_set.add((min(u, v), max(u, v)))
                                edges.append([u, v])
                                added_after_swap = True
                                break

                        if added_after_swap:
                            swapped = True
                            no_improve_count = 0
                            break
                        else:
                            # Restore the removed edge
                            adj[ru] |= (1 << rv)
                            adj[rv] |= (1 << ru)
                            edge_set.add((ru, rv))
                            edges.append([ru, rv])

                    if not swapped:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

            if no_improve_count > 50:
                break

        return edges

    random.seed(42)

    best_edges = []
    best_count = 0

    # Strategy 1: Spectral-guided initialization
    potential_edges, _ = spectral_init(seed_val=42)

    edges1 = []
    adj1 = [0] * n
    edge_set1 = set()
    greedy_add(adj1, edge_set1, edges1, potential_edges)

    if len(edges1) > best_count:
        best_count = len(edges1)
        best_edges = edges1[:]

    # Strategy 2: Standard K5-based initialization
    edges2 = []
    for k in range(8):
        edges2 += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    adj2 = make_adj(edges2)
    edge_set2 = set((min(u, v), max(u, v)) for u, v in edges2)

    all_possible = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set2]
    greedy_add(adj2, edge_set2, edges2, all_possible)

    if len(edges2) > best_count:
        best_count = len(edges2)
        best_edges = edges2[:]

    # Strategy 3: Randomized restarts with local search + swaps
    for restart in range(200):
        if not time_ok():
            break

        if restart < 10:
            # Start from best
            current_edges = best_edges[:]
        elif restart < 40:
            # Start from a random subset of best (moderate perturbation)
            current_edges = [e for e in best_edges if random.random() < 0.85]
        else:
            # Start from a more aggressive random subset
            current_edges = [e for e in best_edges if random.random() < 0.7]

        current_edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)
        current_adj = make_adj(current_edges)

        remaining = [(u, v) for u in range(n) for v in range(u + 1, n)
                     if (u, v) not in current_edge_set]
        random.shuffle(remaining)

        greedy_add(current_adj, current_edge_set, current_edges, remaining)

        # Local search with swaps to improve
        current_edges = local_search_with_swaps(current_adj, current_edge_set, current_edges,
                                                max_iters=80)

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]

    # Final polish: try to add any remaining edges
    final_edge_set = set((min(u, v), max(u, v)) for u, v in best_edges)
    final_adj = make_adj(best_edges)
    final_remaining = [(u, v) for u in range(n) for v in range(u + 1, n)
                       if (u, v) not in final_edge_set]
    greedy_add(final_adj, final_edge_set, best_edges, final_remaining)

    # Final local search with swaps (high-precision polish)
    best_edges = local_search_with_swaps(final_adj, final_edge_set, best_edges, max_iters=150)

    # One more greedy pass after swaps
    final_edge_set2 = set((min(u, v), max(u, v)) for u, v in best_edges)
    final_adj2 = make_adj(best_edges)
    final_remaining2 = [(u, v) for u in range(n) for v in range(u + 1, n)
                        if (u, v) not in final_edge_set2]
    greedy_add(final_adj2, final_edge_set2, best_edges, final_remaining2)

    return best_edges
# EVOLVE-BLOCK-END
