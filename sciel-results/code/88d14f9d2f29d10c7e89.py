import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed, max_restarts=8):
        edges_orig = get_graph(n, p, seed)
        if not edges_orig:
            return []

        # Precompute original adjacency for reference
        orig_adj = [set() for _ in range(n)]
        for u, v in edges_orig:
            orig_adj[u].add(v)
            orig_adj[v].add(u)

        best_cliques = None
        best_count = float('inf')

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            edges = set(edges_orig)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)

            cliques = []
            rng = random.Random(seed + 1000 + restart)

            while edges:
                if time.time() > deadline:
                    # Fallback: add remaining edges as K2
                    for e in edges:
                        cliques.append([e[0], e[1]])
                    break

                # Find edge with minimum common neighbors (with random tie-breaking)
                best_common = float('inf')
                best_u = -1
                best_v = -1
                tied = []

                for u, v in edges:
                    common = len(adj[u] & adj[v])
                    if common < best_common:
                        best_common = common
                        tied = [(u, v)]
                    elif common == best_common:
                        tied.append((u, v))

                if best_u == -1 and tied:
                    # Random tie-breaking
                    best_u, best_v = tied[rng.randrange(len(tied))]
                elif not tied:
                    break

                # Build maximal clique starting from edge (best_u, best_v)
                C = [best_u, best_v]
                P = adj[best_u] & adj[best_v]
                P -= {best_u, best_v}

                # Greedy expansion with degree-based ordering
                while P:
                    if time.time() > deadline:
                        break
                    # Sort candidates by degree in remaining graph (desc)
                    candidates = sorted(P, key=lambda x: len(adj[x]), reverse=True)
                    best_cand = None
                    for cand in candidates:
                        # Check if cand connects to all in C
                        if all(cand in adj[c] for c in C):
                            best_cand = cand
                            break

                    if best_cand is None:
                        # Try less strict: find cand with most connections to C
                        best_score = -1
                        for cand in candidates:
                            score = sum(1 for c in C if cand in adj[c])
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                        if best_cand is None:
                            break

                    C.append(best_cand)
                    P = P & adj[best_cand]

                cliques.append(C)

                # Remove edges in C
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v2 = C[i], C[j]
                        if u > v2:
                            u, v2 = v2, u
                        if (u, v2) in edges:
                            edges.remove((u, v2))
                            adj[u].remove(v2)
                            adj[v2].remove(u)

            # Local improvement: try to merge cliques whose union is a clique
            changed = True
            while changed:
                changed = False
                new_cliques = []
                i = 0
                while i < len(cliques):
                    C1 = set(cliques[i])
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        if time.time() > deadline:
                            break
                        C2 = set(cliques[j])
                        union = C1 | C2
                        # Check if union is a clique using original adjacency
                        is_clique = True
                        for u in union:
                            for v in union:
                                if u < v and v not in orig_adj[u]:
                                    is_clique = False
                                    break
                            if not is_clique:
                                break
                        if is_clique and len(union) > max(len(C1), len(C2)):
                            new_cliques.append(list(union))
                            cliques[j] = None
                            merged = True
                            break
                    if not merged:
                        new_cliques.append(cliques[i])
                    i += 1

                new_cliques = [c for c in new_cliques if c is not None]
                if len(new_cliques) < len(cliques):
                    changed = True
                    cliques = new_cliques

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

            # If we already have a very good result, stop early
            if best_count <= 15:
                break

        return best_cliques if best_cliques is not None else []

    results = []
    results.append(solve_instance(32, 0.6, 601))

    if time.time() > deadline:
        results.append(solve_instance(64, 0.55, 602))
        return results

    results.append(solve_instance(64, 0.55, 602))

    return results
