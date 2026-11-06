import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        best_cliques = [list(e) for e in edges]
        best_count = len(best_cliques)

        end_time = start_time + time_budget

        def clique_edges(vertices):
            res = set()
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    res.add((min(u, v), max(u, v)))
            return res

        def is_valid_clique(vertices, remaining_edges):
            for e in clique_edges(vertices):
                if e not in remaining_edges:
                    return False
            return True

        # Line graph matching score: for a vertex w being added to clique C,
        # score = number of pairs of remaining edges (w,x) and (w,y) where x,y are both
        # in the candidate set, indicating potential for further clique growth
        def line_graph_score(w, cand_set, remaining_edges):
            score = 0
            for x in cand_set:
                if w == x:
                    continue
                e = (min(w, x), max(w, x))
                if e in remaining_edges:
                    score += 1
            return score

        attempts = 0
        while time.time() < end_time:
            attempts += 1
            attempt_rng = random.Random(seed + attempts * 12345 + 67890)

            remaining_edges = set(edges)
            cliques = []

            while remaining_edges:
                edge_list = list(remaining_edges)
                attempt_rng.shuffle(edge_list)

                best_seed_clique = None
                max_cleared = 0
                best_seed_clique_edges = None

                # Try multiple seed edges with beam search
                for seed_edge in edge_list[:20]:
                    u, v = seed_edge
                    current_clique = [u, v]

                    # Build initial candidate set
                    cand_set = set(adj[u])
                    cand_set.intersection_update(adj[v])
                    cand_set.discard(u)
                    cand_set.discard(v)

                    # Filter: must have all edges to current_clique in remaining
                    valid_cands = []
                    for w in cand_set:
                        valid = True
                        for c in current_clique:
                            e = (min(w, c), max(w, c))
                            if e not in remaining_edges:
                                valid = False
                                break
                        if valid:
                            valid_cands.append(w)
                    cand_set = set(valid_cands)

                    # Greedy growth with line graph scoring
                    while cand_set:
                        # Score each candidate using line graph matching heuristic
                        best_w = None
                        max_score = -1
                        for w in cand_set:
                            # Primary: edges to other candidates (clique potential)
                            score = line_graph_score(w, cand_set, remaining_edges)
                            # Secondary: degree in remaining graph (edge coverage)
                            deg = 0
                            for neighbor in adj[w]:
                                e = (min(w, neighbor), max(w, neighbor))
                                if e in remaining_edges:
                                    deg += 1
                            # Combine: prioritize clique potential, then coverage
                            combined_score = score * 10 + deg
                            if combined_score > max_score:
                                max_score = combined_score
                                best_w = w
                            elif combined_score == max_score:
                                if attempt_rng.random() < 0.5:
                                    best_w = w

                        if best_w is None:
                            break

                        current_clique.append(best_w)
                        cand_set.remove(best_w)

                        # Update candidates: must be connected to best_w in remaining
                        new_cand_set = set()
                        for w in cand_set:
                            e_bw = (min(w, best_w), max(w, best_w))
                            if e_bw in remaining_edges:
                                new_cand_set.add(w)
                        cand_set = new_cand_set

                    # Evaluate
                    if len(current_clique) >= 2:
                        c_edges = clique_edges(current_clique)
                        if is_valid_clique(current_clique, remaining_edges):
                            if len(c_edges) > max_cleared:
                                max_cleared = len(c_edges)
                                best_seed_clique = current_clique
                                best_seed_clique_edges = c_edges

                if not best_seed_clique:
                    e = edge_list[0]
                    best_seed_clique = list(e)
                    best_seed_clique_edges = {e}

                cliques.append(best_seed_clique)
                for e in best_seed_clique_edges:
                    remaining_edges.discard(e)

            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = [list(c) for c in cliques]

            if time.time() > end_time - 2:
                break

        return best_cliques

    t1 = time.time()
    budget1 = min(35, DEADLINE - (t1 - start_time) - 5)
    if budget1 < 1:
        budget1 = 1
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(105, DEADLINE - (t2 - start_time))
    if budget2 < 1:
        budget2 = 1
    cliques2 = solve_instance(64, 0.5, 602, budget2)

    return [cliques1, cliques2]
