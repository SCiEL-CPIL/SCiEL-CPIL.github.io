import time
import random
import sys

def _random_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
    return adj, edges

def _grow_clique(clique, adj, remaining_edges, cand_set, rng):
    """
    Grow a clique greedily. cand_set contains vertices that are connected to
    every member of clique in the remaining graph. We repeatedly pick the
    candidate that maximizes internal density (edges to other candidates) with
    random tie-breaking.
    """
    while cand_set:
        best_w = None
        max_score = -1
        for w in cand_set:
            score = 0
            for x in cand_set:
                if w == x:
                    continue
                e = (min(w, x), max(w, x))
                if e in remaining_edges:
                    score += 1
            if score > max_score:
                max_score = score
                best_w = w
            elif score == max_score and rng.random() < 0.5:
                best_w = w
        if best_w is None:
            break
        clique.append(best_w)
        cand_set.remove(best_w)
        # Filter: keep only candidates connected to best_w in remaining graph
        new_cand = set()
        for w in cand_set:
            e = (min(best_w, w), max(best_w, w))
            if e in remaining_edges:
                new_cand.add(w)
        cand_set = new_cand
    return clique

def _solve_instance(n, p, seed, adj, edges, deadline):
    best_cliques = [list(e) for e in edges]
    best_count = len(best_cliques)

    end_time = min(deadline, time.time() + 10.0)
    attempts = 0

    while time.time() < end_time:
        attempts += 1
        attempt_rng = random.Random(seed + attempts * 12345 + 67890)

        remaining_edges = set(edges)
        cliques = []

        while remaining_edges:
            if time.time() > deadline:
                # Fallback: add remaining as K2
                for e in remaining_edges:
                    cliques.append(list(e))
                remaining_edges = set()
                break

            edge_list = list(remaining_edges)
            attempt_rng.shuffle(edge_list)

            best_seed_clique = None
            best_size = 0

            # Try a handful of seed edges to find a good starting clique
            for seed_edge in edge_list[:12]:
                u, v = seed_edge
                current_clique = [u, v]

                # Candidates: common neighbors in remaining graph
                cand_set = set(adj[u])
                cand_set.intersection_update(adj[v])
                cand_set.discard(u)
                cand_set.discard(v)

                # Filter by remaining edges
                valid_cands = set()
                for w in cand_set:
                    if (min(w, u), max(w, u)) in remaining_edges and \
                       (min(w, v), max(w, v)) in remaining_edges:
                        valid_cands.add(w)
                cand_set = valid_cands

                _grow_clique(current_clique, adj, remaining_edges, cand_set, attempt_rng)

                if len(current_clique) > best_size:
                    best_size = len(current_clique)
                    best_seed_clique = current_clique

            if best_seed_clique is None or best_size < 2:
                u, v = next(iter(remaining_edges))
                best_seed_clique = [u, v]

            cliques.append(best_seed_clique)

            # Remove edges covered by this clique
            for i in range(len(best_seed_clique)):
                for j in range(i + 1, len(best_seed_clique)):
                    u, v = best_seed_clique[i], best_seed_clique[j]
                    e = (min(u, v), max(u, v))
                    remaining_edges.discard(e)

        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques

        # If we've essentially solved it (few cliques), keep trying to improve
        if best_count <= max(10, n // 2):
            break

    return best_cliques

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    results = []

    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            results.append([list(e) for e in edges])
            continue

        adj, edges = _random_graph(n, p, seed)
        partition = _solve_instance(n, p, seed, adj, edges, deadline)
        results.append(partition)

    return results
