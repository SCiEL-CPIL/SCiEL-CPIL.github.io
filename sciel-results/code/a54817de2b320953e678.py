import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute all triples for K3,3 checking
    triples = list(itertools.combinations(range(n), 3))

    def has_k33(adj):
        """Check if graph has K3,3 using bitmasks."""
        for u1, u2, u3 in triples:
            if (adj[u1] & adj[u2] & adj[u3]).bit_count() >= 3:
                return True
        return False

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
        """
        # For each pair (b1, b2) in N(u):
        #   common = N(v) & N(b1) & N(b2) & ~(1<<u) & ~(1<<v) & ~(1<<b1) & ~(1<<b2)
        #   if common.bit_count() >= 2: return False
        nu = adj[u]
        nv = adj[v]
        # b1, b2 must be in N(u), distinct from v (v not in N(u) since we're adding)
        # a1, a2 must be in N(v) ∩ N(b1) ∩ N(b2), distinct from u, v, b1, b2
        for b1 in range(n):
            if not (nu >> b1) & 1:
                continue
            for b2 in range(b1 + 1, n):
                if not (nu >> b2) & 1:
                    continue
                common = nv & adj[b1] & adj[b2]
                # Remove u, v, b1, b2 from consideration
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
            # Count how many violations each edge participates in
            edge_counts = {}
            for u1, u2, u3, common in violations:
                for c in range(n):
                    if (common >> c) & 1:
                        for u in (u1, u2, u3):
                            key = (min(u, c), max(u, c))
                            edge_counts[key] = edge_counts.get(key, 0) + 1
            if not edge_counts:
                break
            # Remove the edge with highest count
            best_edge = max(edge_counts, key=edge_counts.get)
            remove_edge(adj, best_edge[0], best_edge[1])
        return adj

    def local_search(adj_init):
        """Full local search: fix, add, then 2-edge swaps."""
        adj = list(adj_init)

        # Phase 1: Fix violations
        adj = fix_violations(adj)

        # Phase 2: Greedy add
        adj = greedy_add(adj)

        edge_count = count_edges(adj)

        # Phase 3: 2-edge swap local search
        # Remove 2 edges, try to add 2 different edges
        swap_improved = True
        while swap_improved and time.time() < deadline:
            swap_improved = False
            edges = all_edges(adj)
            random.shuffle(edges)
            for i in range(len(edges)):
                if time.time() >= deadline:
                    break
                e1 = edges[i]
                for j in range(i + 1, len(edges)):
                    e2 = edges[j]
                    # Temporarily remove both edges
                    remove_edge(adj, e1[0], e1[1])
                    remove_edge(adj, e2[0], e2[1])

                    # Try to add two non-edges
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
                            swap_improved = True
                        else:
                            # Revert
                            for u, v in added:
                                remove_edge(adj, u, v)
                            add_edge(adj, e1[0], e1[1])
                            add_edge(adj, e2[0], e2[1])
                    else:
                        # Revert
                        for u, v in added:
                            remove_edge(adj, u, v)
                        add_edge(adj, e1[0], e1[1])
                        add_edge(adj, e2[0], e2[1])

            # After swaps, do another greedy add
            adj = greedy_add(adj)
            new_count = count_edges(adj)
            if new_count > edge_count:
                edge_count = new_count

        return adj, edge_count

    def structured_init():
        """5x4 grid construction: partition into 5 groups of 4,
        group i connects to groups (i+1)%5 and (i+2)%5."""
        adj = [0] * n
        groups = [[i * 4 + j for j in range(4)] for i in range(5)]
        for i in range(5):
            for j in [(i + 1) % 5, (i + 2) % 5]:
                for u in groups[i]:
                    for v in groups[j]:
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
        # Remove some random edges
        for _ in range(k // 2):
            if all_e:
                e = random.choice(all_e)
                remove_edge(adj, e[0], e[1])
        # Add some random edges
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

    # Try structured initialization first
    adj_struct = structured_init()
    adj_ls, ec_ls = local_search(adj_struct)
    update_best(adj_ls)

    # Try random restarts with various densities
    for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        if time.time() >= deadline:
            break
        for _ in range(20):
            if time.time() >= deadline:
                break
            adj_rand = random_init(p)
            adj_ls, ec_ls = local_search(adj_rand)
            update_best(adj_ls)

    # Try perturbations of the best found so far
    if best_graph is not None:
        # Reconstruct best adj
        best_adj = [0] * n
        for e in best_graph:
            u, v = e
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)

        for _ in range(30):
            if time.time() >= deadline:
                break
            adj_pert = perturb(best_adj, k=random.randint(2, 5))
            adj_ls, ec_ls = local_search(adj_pert)
            update_best(adj_ls)

    if best_graph is None:
        # Fallback: sparse K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]

    return best_graph
