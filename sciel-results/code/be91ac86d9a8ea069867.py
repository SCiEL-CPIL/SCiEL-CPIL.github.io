import time
from random import Random

def construct():
    deadline = time.time() + 115  # 5s margin before 120s limit

    def generate_graph(n, p, seed):
        rng = Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def solve_instance(n, p, seed, time_budget):
        adj, edges = generate_graph(n, p, seed)
        if not edges:
            return []

        best_cliques = None
        best_count = float('inf')
        start_time = time.time()

        max_restarts = 500 if n <= 32 else 100

        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break

            rng = Random(seed * 1000 + restart)
            cur_adj = adj[:]
            cliques = []

            # Random ordering of vertices
            order = list(range(n))
            rng.shuffle(order)

            for v in order:
                if time.time() - start_time > time_budget:
                    break

                # Check if v has any remaining edges
                nbrs = cur_adj[v]
                if nbrs == 0:
                    continue

                # Greedy clique expansion from v
                clique = [v]
                candidates = nbrs & ~(1 << v)

                while candidates:
                    # Pick vertex with max degree in candidates
                    best_u = -1
                    best_deg = -1
                    tmp = candidates
                    while tmp:
                        lsb = tmp & -tmp
                        u = lsb.bit_length() - 1
                        deg = (cur_adj[u] & candidates).bit_count()
                        if deg > best_deg:
                            best_deg = deg
                            best_u = u
                        tmp ^= lsb

                    if best_u < 0:
                        break

                    clique.append(best_u)
                    candidates &= cur_adj[best_u]
                    candidates &= ~(1 << best_u)

                if len(clique) < 2:
                    lsb = nbrs & -nbrs
                    u = lsb.bit_length() - 1
                    clique = [v, u]

                # Remove all edges within the clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        cur_adj[u] &= ~(1 << w)
                        cur_adj[w] &= ~(1 << u)

                cliques.append(clique)

            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques

        if best_cliques is None:
            return [[u, v] for u, v in edges]

        return best_cliques

    t1 = 55
    t2 = 55

    result = []
    for n, p, seed, budget in ((32, 0.6, 601, t1), (64, 0.55, 602, t2)):
        result.append(solve_instance(n, p, seed, budget))
    return result
