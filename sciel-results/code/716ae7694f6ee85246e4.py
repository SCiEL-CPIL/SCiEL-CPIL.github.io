"""Construct C6-free graph on 40 vertices with maximum edges."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    n = 40
    full_mask = (1 << n) - 1

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def has_c6_fast(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6 using bit operations.
        C6 through (u,v): u-a-b-c-d-v-u
        Need a in N(u)\{v}, d in N(v)\{u}, and path a-b-c-d in G\{u,v}.
        """
        if (adj[u] >> v) & 1:
            return True

        nu_mask = adj[u] & ~(1 << v)
        nv_mask = adj[v] & ~(1 << u)

        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a

            d = nv_mask & ~lb_a
            while d:
                lb_d = d & (-d)
                di = lb_d.bit_length() - 1
                d ^= lb_d

                # b in N(a) \ {u, v, d, a}
                b_cand = adj[ai] & ~(1 << u) & ~(1 << v) & ~lb_d & ~lb_a
                # c in N(d) \ {u, v, a, d}
                c_cand = adj[di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d

                # Check if any neighbor of b is in c_cand
                bm = b_cand
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    if adj[bi] & c_cand:
                        return True
        return False

    def try_add_edge(adj, u, v):
        if not has_c6_fast(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def local_search(adj, max_iter=50):
        """
        Local search: try to add any missing edge that doesn't create C6.
        Also try edge swaps (remove one, add another) to escape local optima.
        """
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            for u, v in all_pairs:
                if time.time() > deadline:
                    return adj
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        improved = True

        # Try edge swaps: remove an existing edge, try to add two new edges
        # This helps escape local optima where no single edge can be added
        for _ in range(20):
            if time.time() > deadline:
                return adj
            current_edges = get_edges(adj)
            if not current_edges:
                break
            # Pick a random edge to remove
            u, v = random.choice(current_edges)
            remove_edge(adj, u, v)
            # Try to add edges
            added_any = False
            for u2, v2 in all_pairs:
                if (u2, v2) == (u, v) or (u2, v2) == (v, u):
                    continue
                if not ((adj[u2] >> v2) & 1):
                    if try_add_edge(adj, u2, v2):
                        added_any = True
            # If we couldn't add enough, put the edge back
            if not added_any:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            # Run local search again
            adj = local_search(adj, max_iter=5)

        return adj

    random.seed(42)
    best_adj = None
    best_count = 0

    # Phase 1: Bipartite-inspired initialization
    # Split vertices into two sets, create dense bipartite structures
    n_restarts = 40

    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        adj = [0] * n

        # Strategy A: Two disjoint bipartite graphs
        if restart % 2 == 0:
            # Split into 20-20, create bipartite graph
            random.seed(1000 + restart)
            partition = list(range(n))
            random.shuffle(partition)
            left = set(partition[:20])
            right = set(partition[20:])

            # Add edges between left and right with probability ~0.8
            for u in left:
                for v in right:
                    if random.random() < 0.8:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

            # Add some intra-partition edges to boost density
            for u in left:
                for v in list(left):
                    if u < v and random.random() < 0.3:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            for u in right:
                for v in list(right):
                    if u < v and random.random() < 0.3:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            # Strategy B: Random dense graph
            random.seed(2000 + restart)
            for u, v in all_pairs:
                if random.random() < 0.4:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        # Prune C6s: repeatedly find and remove edges from C6s
        for prune_iter in range(500):
            if time.time() > deadline:
                break

            # Find a C6 and remove an edge from it
            c6_found = False
            # Try to find a C6 by checking pairs of non-adjacent vertices
            for u in range(n):
                if time.time() > deadline:
                    break
                nu = adj[u]
                # v must be non-adjacent to u
                nv_candidates = full_mask & ~nu & ~(1 << u)

                v_mask = nv_candidates
                while v_mask and not c6_found:
                    lb_v = v_mask & (-v_mask)
                    vi = lb_v.bit_length() - 1
                    v_mask ^= lb_v

                    # Path u-a-b-c-d-v
                    a_mask = nu
                    while a_mask and not c6_found:
                        lb_a = a_mask & (-a_mask)
                        ai = lb_a.bit_length() - 1
                        a_mask ^= lb_a

                        d_mask = adj[vi] & ~lb_a
                        while d_mask and not c6_found:
                            lb_d = d_mask & (-d_mask)
                            di = lb_d.bit_length() - 1
                            d_mask ^= lb_d

                            b_cand = adj[ai] & ~(1 << u) & ~(1 << vi) & ~lb_d & ~lb_a
                            c_cand = adj[di] & ~(1 << u) & ~(1 << vi) & ~lb_a & ~lb_d

                            bm = b_cand
                            while bm and not c6_found:
                                lb_b = bm & (-bm)
                                bi = lb_b.bit_length() - 1
                                bm ^= lb_b
                                if adj[bi] & c_cand:
                                    # Found C6: u-ai-bi-?-di-vi
                                    # Remove a random edge from the cycle
                                    # Edges: (u,ai), (ai,bi), (bi,?), (?,di), (di,vi), (vi,u)
                                    # We know u-ai, ai-bi, di-vi exist
                                    # Find ? such that bi-? and ?-di
                                    # For simplicity, remove (u,ai) or (di,vi)
                                    if random.random() < 0.5:
                                        remove_edge(adj, u, ai)
                                    else:
                                        remove_edge(adj, di, vi)
                                    c6_found = True
                                    break

            if not c6_found:
                break

        # Local search to fill in missing edges
        adj = local_search(adj)

        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]

        if best_count >= 105:
            if restart >= 15:
                break

    # Phase 2: Perturbation-based local search from best solution
    if best_adj is not None:
        for perturb_round in range(50):
            if time.time() > deadline:
                break

            adj = best_adj[:]

            # Remove k random edges to perturb
            k = random.randint(3, 10)
            current_edges = get_edges(adj)
            if len(current_edges) > k:
                removed = random.sample(current_edges, k)
                for u, v in removed:
                    remove_edge(adj, u, v)

            # Try to re-add edges greedily
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if time.time() > deadline:
                    break
                if not ((adj[u] >> v) & 1):
                    try_add_edge(adj, u, v)

            # Local search
            adj = local_search(adj)

            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]

    # Phase 3: Final refinement with more aggressive swaps
    if best_adj is not None:
        adj = best_adj[:]
        # Try more swap operations
        for _ in range(100):
            if time.time() > deadline:
                break
            current_edges = get_edges(adj)
            if not current_edges:
                break
            u, v = random.choice(current_edges)
            remove_edge(adj, u, v)
            # Try to add multiple edges
            added = 0
            for u2, v2 in all_pairs:
                if time.time() > deadline:
                    break
                if (u2, v2) == (u, v) or (u2, v2) == (v, u):
                    continue
                if not ((adj[u2] >> v2) & 1):
                    if try_add_edge(adj, u2, v2):
                        added += 1
            if added == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            adj = local_search(adj, max_iter=3)

            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]

    return get_edges(best_adj)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Edges: {len(edges)}, Time: {t1-t0:.2f}s")
