"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def recompute_n1_n2_n4(adj):
        """Compute N1, N2, N4 bitsets for C6 checking."""
        N1 = adj[:]
        N2 = [0] * n
        for i in range(n):
            mask = 0
            x = N1[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N1[j]
                x ^= lsb
            N2[i] = mask

        N4 = [0] * n
        for i in range(n):
            mask = 0
            x = N2[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N2[j]
                x ^= lsb
            N4[i] = mask
        return N1, N2, N4

    def is_safe(u, v, N1, N4):
        """Check if adding edge (u,v) creates no C6."""
        return (N1[u] & N4[v]) == 0 and (N1[v] & N4[u]) == 0

    def get_edges(adj):
        edges = []
        for u in range(n):
            x = adj[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                x ^= lsb
        return edges

    def count_edges(adj):
        cnt = 0
        for u in range(n):
            cnt += bin(adj[u]).count('1')
        return cnt // 2

    def make_seed_adj():
        """Create adjacency for 8 disjoint K5s."""
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def greedy_add(adj, N1, N2, N4, order, deadline):
        """Try to add edges in given order, updating N1/N2/N4 incrementally."""
        added = 0
        for u, v in order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if is_safe(u, v, N1, N4):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                added += 1
                # Recompute N1, N2, N4 after each addition
                N1, N2, N4 = recompute_n1_n2_n4(adj)
        return adj, N1, N2, N4, added

    def local_search(adj, N1, N2, N4, deadline):
        """Try to improve by removing one edge and re-adding others."""
        edges = get_edges(adj)
        best_adj = adj[:]
        best_N1, best_N2, best_N4 = N1[:], N2[:], N4[:]
        best_count = count_edges(adj)

        # Try removing each edge and see if we can add more
        for idx, (u, v) in enumerate(edges):
            if time.time() > deadline:
                break

            # Remove edge (u, v)
            test_adj = adj[:]
            test_adj[u] &= ~(1 << v)
            test_adj[v] &= ~(1 << u)

            # Recompute
            tN1, tN2, tN4 = recompute_n1_n2_n4(test_adj)

            # Try to greedily add back edges (including possibly (u,v) and others)
            candidates = []
            for a in range(n):
                for b in range(a + 1, n):
                    if not (test_adj[a] & (1 << b)):
                        candidates.append((a, b))
            random.shuffle(candidates)

            test_adj, tN1, tN2, tN4, _ = greedy_add(test_adj, tN1, tN2, tN4, candidates, deadline)
            new_count = count_edges(test_adj)

            if new_count > best_count:
                best_count = new_count
                best_adj = test_adj
                best_N1, best_N2, best_N4 = tN1, tN2, tN4

        return best_adj, best_N1, best_N2, best_N4, best_count

    best_edges = []
    best_count = 0

    # Build candidate edge list
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # Multiple strategies
    while time.time() < deadline:
        strategy = random.random()

        if strategy < 0.35:
            # Strategy 1: Greedy from seed with random order
            adj = make_seed_adj()
            N1, N2, N4 = recompute_n1_n2_n4(adj)

            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)

            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

        elif strategy < 0.55:
            # Strategy 2: Greedy from empty with random order
            adj = [0] * n
            N1, N2, N4 = recompute_n1_n2_n4(adj)

            candidates = all_pairs[:]
            random.shuffle(candidates)

            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

        elif strategy < 0.80:
            # Strategy 3: Greedy from seed + local search
            adj = make_seed_adj()
            N1, N2, N4 = recompute_n1_n2_n4(adj)

            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)

            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)

            # Local search
            adj, N1, N2, N4, count = local_search(adj, N1, N2, N4, deadline)
            edges = get_edges(adj)
            if count > best_count:
                best_count = count
                best_edges = edges

        else:
            # Strategy 4: Perturbed seed - start with 8 K5s but randomly remove some intra-K5 edges
            adj = make_seed_adj()
            # Remove 2-4 random edges from within K5s
            for _ in range(random.randint(2, 4)):
                k = random.randint(0, 7)
                a, b = random.sample(range(5), 2)
                u, v = 5 * k + a, 5 * k + b
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

            N1, N2, N4 = recompute_n1_n2_n4(adj)

            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)

            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    return best_edges
# EVOLVE-BLOCK-END
