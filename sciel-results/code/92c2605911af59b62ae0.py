import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118.0

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def max_clique_in_induced(n, adj, mask, rng):
        best_clique = []

        def expand(R, P):
            nonlocal best_clique
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return

            pc = popcount(P)
            if len(R) + pc <= len(best_clique):
                return

            max_deg = -1
            candidates = []
            P_copy = P
            while P_copy:
                v_bit = P_copy & (-P_copy)
                v = v_bit.bit_length() - 1
                P_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [v]
                elif deg == max_deg:
                    candidates.append(v)

            if not candidates:
                return

            u = rng.choice(candidates)
            Q = P & ~adj[u]

            while Q:
                max_v_deg = -1
                v_candidates = []
                Q_copy = Q
                while Q_copy:
                    v_bit = Q_copy & (-Q_copy)
                    i = v_bit.bit_length() - 1
                    Q_copy ^= v_bit
                    deg = popcount(P & adj[i])
                    if deg > max_v_deg:
                        max_v_deg = deg
                        v_candidates = [i]
                    elif deg == max_v_deg:
                        v_candidates.append(i)

                if not v_candidates:
                    break

                v = rng.choice(v_candidates)

                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)

                P &= ~(1 << v)
                Q = Q & P

        expand([], mask)
        return best_clique

    def greedy_partition(n, adj, rng, deadline, mode='max', alpha=0.5):
        rem_adj = adj[:]
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            best_score = -1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                deg_u = popcount(rem_adj[u])
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    deg_v = popcount(rem_adj[v])
                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)

                    if mode == 'max':
                        score = cnt
                    elif mode == 'min':
                        score = -cnt
                    else:  # weighted
                        score = alpha * cnt + (1 - alpha) * (deg_u + deg_v) / 2

                    if score > best_score:
                        best_score = score
                        best_u, best_v = u, v

            if best_u == -1:
                break

            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def validate_partition(n, adj, partition):
        for clique in partition:
            if len(clique) < 2:
                return False
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if not (adj[u] & (1 << v)):
                        return False

        edge_count = {}
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edge_count[frozenset([u, v])] = 0

        for clique in partition:
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = frozenset([clique[i], clique[j]])
                    if e in edge_count:
                        edge_count[e] += 1
                    else:
                        return False

        for e, cnt in edge_count.items():
            if cnt != 1:
                return False

        return True

    def local_search_improve(n, adj, partition, rng, deadline, max_iter=200):
        """
        Try to improve partition by merging overlapping cliques or
        extracting larger cliques from unions of small ones.
        """
        if not partition:
            return partition

        best = partition[:]
        best_count = len(best)

        # Build edge-to-clique mapping
        def build_edge_map(part):
            emap = {}
            for idx, clique in enumerate(part):
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = frozenset([clique[i], clique[j]])
                        emap[e] = idx
            return emap

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            emap = build_edge_map(best)

            # Find pairs of cliques that share a vertex and whose union forms a clique
            improved = False
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() > deadline:
                        break

                    # Check if union is a clique
                    union = list(set(best[i]) | set(best[j]))
                    is_clique = True
                    for a in range(len(union)):
                        for b in range(a + 1, len(union)):
                            if not (adj[union[a]] & (1 << union[b])):
                                is_clique = False
                                break
                        if not is_clique:
                            break

                    if is_clique and len(union) >= 2:
                        # Check that all edges in union are covered by i or j
                        union_edges = set()
                        for a in range(len(union)):
                            for b in range(a + 1, len(union)):
                                union_edges.add(frozenset([union[a], union[b]]))

                        covered = set()
                        for k in (i, j):
                            for a in range(len(best[k])):
                                for b in range(a + 1, len(best[k])):
                                    covered.add(frozenset([best[k][a], best[k][b]]))

                        if union_edges == covered:
                            # Merge i and j
                            new_part = []
                            for k in range(len(best)):
                                if k != i and k != j:
                                    new_part.append(best[k])
                            new_part.append(union)

                            if len(new_part) < best_count:
                                best = new_part
                                best_count = len(best)
                                improved = True
                                break

                if improved:
                    break

            if not improved:
                # Try extracting a larger clique from a single small clique's neighborhood
                for i in range(len(best)):
                    if time.time() > deadline:
                        break
                    if len(best[i]) < 4:
                        continue

                    clique = best[i]
                    # Find max clique containing all vertices of this clique
                    common = (1 << n) - 1
                    for v in clique:
                        common &= adj[v]

                    larger = max_clique_in_induced(n, adj, common, rng)
                    if len(larger) > len(clique) and len(larger) >= 2:
                        # Check if larger clique's edges are subset of edges covered by clique i
                        larger_edges = set()
                        for a in range(len(larger)):
                            for b in range(a + 1, len(larger)):
                                larger_edges.add(frozenset([larger[a], larger[b]]))

                        clique_edges = set()
                        for a in range(len(clique)):
                            for b in range(a + 1, len(clique)):
                                clique_edges.add(frozenset([clique[a], clique[b]]))

                        # The larger clique must not use edges from other cliques
                        # This is complex; skip for now
                        pass

        return best

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)

        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)

        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            rng = random.Random(seed * 1000 + attempt)

            # Cycle through strategies
            strategy = attempt % 6
            if strategy == 0:
                result = greedy_partition(n, adj, rng, DEADLINE, mode='max', alpha=0.5)
            elif strategy == 1:
                result = greedy_partition(n, adj, rng, DEADLINE, mode='min', alpha=0.5)
            elif strategy == 2:
                result = greedy_partition(n, adj, rng, DEADLINE, mode='weighted', alpha=0.3)
            elif strategy == 3:
                result = greedy_partition(n, adj, rng, DEADLINE, mode='weighted', alpha=0.7)
            elif strategy == 4:
                result = greedy_partition(n, adj, rng, DEADLINE, mode='weighted', alpha=0.5)
            else:
                result = greedy_partition(n, adj, rng, DEADLINE, mode='max', alpha=0.5)

            if result is not None and validate_partition(n, adj, result):
                # Try local search improvement
                improved = local_search_improve(n, adj, result, rng, DEADLINE, max_iter=50)
                if validate_partition(n, adj, improved):
                    count = len(improved)
                    if count < best_count:
                        best_count = count
                        best_cliques = improved
                else:
                    count = len(result)
                    if count < best_count:
                        best_count = count
                        best_cliques = result

            attempt += 1

        return best_cliques

    # Time budget allocation
    t1 = 18.0
    t2 = 85.0

    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale

    result1 = solve_instance(32, 0.6, 601, t1)

    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, t2)

    return [result1, result2]
