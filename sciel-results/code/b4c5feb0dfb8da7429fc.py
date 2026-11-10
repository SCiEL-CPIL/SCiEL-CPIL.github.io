import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 150s kill

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
        """
        Find max clique in G[mask].
        adj is the global adjacency list (remaining edges).
        mask is a bitmask of vertices to consider.
        """
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

            # Pivot selection: find vertex in P with max degree in P
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
                # Pick v from Q with max degree in P
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

    def greedy_partition_max(n, adj, rng, deadline):
        """
        Greedy: repeatedly find edge with MAX common neighbors,
        extract max clique containing it, remove those edges.
        """
        rem_adj = adj[:]
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find edge (u,v) with MAX common neighbors in remaining graph
            max_common = -1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                # Iterate neighbors v > u
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt > max_common:
                        max_common = cnt
                        best_u, best_v = u, v

            if best_u == -1:
                break

            # Find max clique containing best_u, best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def greedy_partition_min(n, adj, rng, deadline):
        """
        Greedy: repeatedly find edge with MIN common neighbors,
        extract max clique containing it, remove those edges.
        """
        rem_adj = adj[:]
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find edge (u,v) with MIN common neighbors in remaining graph
            min_common = n + 1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                # Iterate neighbors v > u
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt < min_common:
                        min_common = cnt
                        best_u, best_v = u, v
                        if min_common == 0:
                            break
                if min_common == 0:
                    break

            if best_u == -1:
                break

            # Find max clique containing best_u, best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def greedy_partition_weighted(n, adj, rng, deadline, weight_alpha=0.5):
        """
        Greedy: find edge with weighted score combining common neighbors
        and edge density. weight_alpha controls tradeoff.
        """
        rem_adj = adj[:]
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find edge with best weighted score
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
                    
                    # Score: balance between common neighbors and degree
                    score = weight_alpha * cnt + (1 - weight_alpha) * (deg_u + deg_v) / 2
                    if score > best_score:
                        best_score = score
                        best_u, best_v = u, v

            if best_u == -1:
                break

            # Find max clique containing best_u, best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)

        # Baseline: each edge as its own 2-clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)

        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            rng = random.Random(seed * 1000 + attempt)
            
            # Try different strategies
            if attempt % 4 == 0:
                result = greedy_partition_max(n, adj, rng, DEADLINE)
            elif attempt % 4 == 1:
                result = greedy_partition_min(n, adj, rng, DEADLINE)
            elif attempt % 4 == 2:
                result = greedy_partition_weighted(n, adj, rng, DEADLINE, weight_alpha=0.3)
            else:
                result = greedy_partition_weighted(n, adj, rng, DEADLINE, weight_alpha=0.7)
            
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
            attempt += 1

        return best_cliques

    # Time budget allocation
    t1 = 15.0
    t2 = 90.0

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
