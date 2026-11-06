import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute all triples for K3,3 checking
    triples = list(itertools.combinations(range(n), 3))

    def count_violations(adj):
        """Returns list of (u1, u2, u3, common_mask)."""
        violations = []
        for u1, u2, u3 in triples:
            common = adj[u1] & adj[u2] & adj[u3]
            if common.bit_count() >= 3:
                violations.append((u1, u2, u3, common))
        return violations

    def is_add_safe(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free.
        A K3,3 is created iff there exist b1,b2 in N(u) and a1,a2 in N(v)∩N(b1)∩N(b2)
        such that all 6 are distinct.
        Key optimization: only need to check pairs (b1,b2) where at least one of b1,b2
        is NOT already adjacent to both u and v (i.e., new common neighbors after adding u-v).
        """
        nu = adj[u]
        nv = adj[v]
        # New common neighbors: vertices adjacent to both u and v (excluding u,v themselves)
        # But we need to check: for any pair b1,b2 in N(u), if N(v)∩N(b1)∩N(b2) has >=2 new vertices
        # Actually, the key insight: adding edge (u,v) creates K3,3 iff there exist b1,b2 in N(u)
        # such that |N(v) ∩ N(b1) ∩ N(b2) \ {u,v,b1,b2}| >= 2
        # We can optimize by noting that if b1,b2 were already both in N(u) and the graph was K3,3-free,
        # then |N(v) ∩ N(b1) ∩ N(b2) \ {u,v,b1,b2}| was at most 1 BEFORE adding (u,v).
        # After adding (u,v), N(u) and N(v) don't change, but we need to recheck.
        # Actually the adjacency lists don't change, so the check is the same.
        # Let's just do the direct check but with early termination.
        nu_bits = nu
        while nu_bits:
            b1 = (nu_bits & (-nu_bits)).bit_length() - 1
            nu_bits &= nu_bits - 1
            nu_bits2 = nu & ~((1 << (b1 + 1)) - 1)
            while nu_bits2:
                b2 = (nu_bits2 & (-nu_bits2)).bit_length() - 1
                nu_bits2 &= nu_bits2 - 1
                common = nv & adj[b1] & adj[b2]
                mask = (1 << u) | (1 << v) | (1 << b1) | (1 << b2)
                common &= ~mask
                if common.bit_count() >= 2:
                    return False
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def non_edges(adj):
        """Return list of all non-edges (i, j) with i < j."""
        res = []
        for i in range(n):
            mi = adj[i]
            for j in range(i + 1, n):
                if not (mi >> j) & 1:
                    res.append((i, j))
        return res

    def all_edges(adj):
        """Return list of all edges (i, j) with i < j."""
        res = []
        for i in range(n):
            mi = adj[i]
            for j in range(i + 1, n):
                if (mi >> j) & 1:
                    res.append((i, j))
        return res

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def greedy_add(adj):
        """Repeatedly add any safe non-edge until none remain."""
        improved = True
        while improved:
            improved = False
            ne = non_edges(adj)
            random.shuffle(ne)
            for u, v in ne:
                if is_add_safe(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
        return adj

    def fix_violations(adj):
        """Remove edges to eliminate all K3,3 violations."""
        while True:
            violations = count_violations(adj)
            if not violations:
                break
            edge_counts = {}
            for u1, u2, u3, common in violations:
                for c in range(n):
                    if (common >> c) & 1:
                        for u in (u1, u2, u3):
                            key = (min(u, c), max(u, c))
                            edge_counts[key] = edge_counts.get(key, 0) + 1
            if not edge_counts:
                break
            best_edge = max(edge_counts, key=edge_counts.get)
            remove_edge(adj, best_edge[0], best_edge[1])
        return adj

    def local_search(adj_init, max_time=None):
        """Full local search: fix, add, 1-edge swaps, then 2-edge swaps."""
        if max_time is None:
            max_time = deadline
        adj = list(adj_init)

        # Phase 1: Fix violations
        adj = fix_violations(adj)

        # Phase 2: Greedy add
        adj = greedy_add(adj)

        edge_count = count_edges(adj)

        # Phase 3: 1-edge swap local search (remove 1, add 1)
        swap1_improved = True
        while swap1_improved and time.time() < max_time:
            swap1_improved = False
            edges = all_edges(adj)
            random.shuffle(edges)
            for e1 in edges:
                if time.time() >= max_time:
                    break
                remove_edge(adj, e1[0], e1[1])
                ne = non_edges(adj)
                random.shuffle(ne)
                added = None
                for u, v in ne:
                    if is_add_safe(adj, u, v):
                        add_edge(adj, u, v)
                        added = (u, v)
                        break
                if added is not None:
                    new_count = count_edges(adj)
                    if new_count > edge_count:
                        edge_count = new_count
                        swap1_improved = True
                    else:
                        remove_edge(adj, added[0], added[1])
                        add_edge(adj, e1[0], e1[1])
                else:
                    add_edge(adj, e1[0], e1[1])

        # Phase 4: 2-edge swap local search - improved version
        # Instead of trying all pairs, sample random pairs for efficiency
        swap2_improved = True
        while swap2_improved and time.time() < max_time:
            swap2_improved = False
            edges = all_edges(adj)
            num_edges = len(edges)
            # Try up to 500 random 2-edge swaps per iteration
            max_swaps = 500
            attempts = 0
            while attempts < max_swaps and time.time() < max_time:
                attempts += 1
                i = random.randrange(num_edges)
                j = random.randrange(num_edges)
                if i == j:
                    continue
                e1 = edges[i]
                e2 = edges[j]
                # Skip if edges share a vertex (can't remove both)
                if e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]:
                    continue
                remove_edge(adj, e1[0], e1[1])
                remove_edge(adj, e2[0], e2[1])
                ne = non_edges(adj)
                random.shuffle(ne)
                added = []
                for u, v in ne:
                    if is_add_safe(adj, u, v):
                        add_edge(adj, u, v)
                        added.append((u, v))
                        if len(added) == 2:
                            break
                if len(added) == 2:
                    new_count = count_edges(adj)
                    if new_count > edge_count:
                        edge_count = new_count
                        swap2_improved = True
                    else:
                        for u, v in added:
                            remove_edge(adj, u, v)
                        add_edge(adj, e1[0], e1[1])
                        add_edge(adj, e2[0], e2[1])
                else:
                    for u, v in added:
                        remove_edge(adj, u, v)
                    add_edge(adj, e1[0], e1[1])
                    add_edge(adj, e2[0], e2[1])
            adj = greedy_add(adj)
            new_count = count_edges(adj)
            if new_count > edge_count:
                edge_count = new_count

        return adj, edge_count

    def structured_init():
        """Denser structured init: 5 groups of 4, connect groups i to (i+1)%5 and (i+3)%5,
        plus intra-group edges to boost density."""
        adj = [0] * n
        groups = [[i * 4 + j for j in range(4)] for i in range(5)]
        # Connect group i to groups (i+1)%5 and (i+3)%5 (skip adjacent in cycle for more structure)
        for i in range(5):
            for j in [(i + 1) % 5, (i + 3) % 5]:
                for u in groups[i]:
                    for v in groups[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        # Add some intra-group edges (complete within each group of 4)
        for i in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = groups[i][a]
                    v = groups[i][b]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def structured_init2():
        """Alternative: 4 groups of 5, connect i to (i+1)%4 and (i+2)%4, complete within groups."""
        adj = [0] * n
        groups = [[i * 5 + j for j in range(5)] for i in range(4)]
        for i in range(4):
            for j in [(i + 1) % 4, (i + 2) % 4]:
                for u in groups[i]:
                    for v in groups[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        # Complete within each group
        for i in range(4):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = groups[i][a]
                    v = groups[i][b]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def random_init(p):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def perturb(adj, k=3):
        """Randomly flip k edges."""
        adj = list(adj)
        all_e = all_edges(adj)
        ne = non_edges(adj)
        for _ in range(k // 2):
            if all_e:
                e = random.choice(all_e)
                remove_edge(adj, e[0], e[1])
        for _ in range(k - k // 2):
            if ne:
                e = random.choice(ne)
                add_edge(adj, e[0], e[1])
        return adj

    best_edges = 0
    best_graph = None

    def update_best(adj):
        nonlocal best_edges, best_graph
        ec = count_edges(adj)
        if ec > best_edges:
            best_edges = ec
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            best_graph = edges

    # Try structured initializations
    for init_func in [structured_init, structured_init2]:
        if time.time() >= deadline:
            break
        adj_struct = init_func()
        adj_ls, ec_ls = local_search(adj_struct)
        update_best(adj_ls)

    # Try random restarts with various densities
    for p in [0.4, 0.5, 0.6, 0.7, 0.8]:
        if time.time() >= deadline:
            break
        for _ in range(10):
            if time.time() >= deadline:
                break
            adj_rand = random_init(p)
            adj_ls, ec_ls = local_search(adj_rand)
            update_best(adj_ls)

    # Try perturbations of the best found so far - more iterations
    if best_graph is not None:
        best_adj = [0] * n
        for e in best_graph:
            u, v = e
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)

        for _ in range(35):
            if time.time() >= deadline:
                break
            adj_pert = perturb(best_adj, k=random.randint(2, 6))
            adj_ls, ec_ls = local_search(adj_pert)
            update_best(adj_ls)
            # Update best_adj if we improved
            if ec_ls > best_edges:
                best_adj = [0] * n
                for e in best_graph:
                    u, v = e
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)

    # Final polish: intensive local search on best graph with tight deadline
    if best_graph is not None and time.time() < deadline - 5:
        best_adj = [0] * n
        for e in best_graph:
            u, v = e
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
        # Run more rounds of local search
        for _ in range(8):
            if time.time() >= deadline - 3:
                break
            adj_polish, ec_polish = local_search(best_adj, max_time=deadline - 3)
            if ec_polish > best_edges:
                update_best(adj_polish)
                best_adj = [0] * n
                for e in best_graph:
                    u, v = e
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)

    if best_graph is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]

    return best_graph
