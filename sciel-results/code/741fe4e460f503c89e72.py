import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)

        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Fallback: each edge is a clique
        best_cliques = [list(e) for e in edges]
        best_count = len(best_cliques)

        end_time = start_time + time_budget

        # Precompute edge lookup
        edge_set = set(edges)

        def has_all_edges_to_clique(w, clique, rem_edges):
            for c in clique:
                e = (min(w, c), max(w, c))
                if e not in rem_edges:
                    return False
            return True

        def count_remaining_neighbors(vertex, rem_edges):
            count = 0
            for neighbor in adj[vertex]:
                e = (min(vertex, neighbor), max(vertex, neighbor))
                if e in rem_edges:
                    count += 1
            return count

        def find_best_seed_edge(rem_edges, rng=None):
            best_edge = None
            max_common = -1
            candidates = []
            for u, v in rem_edges:
                # Common neighbors of u and v
                common = adj[u].intersection(adj[v])
                count = 0
                for w in common:
                    if w == u or w == v: continue
                    e_wu = (min(w, u), max(w, u))
                    e_wv = (min(w, v), max(w, v))
                    if e_wu in rem_edges and e_wv in rem_edges:
                        count += 1
                if count > max_common:
                    max_common = count
                    candidates = [(u, v)]
                elif count == max_common:
                    candidates.append((u, v))
            
            if not candidates:
                return next(iter(rem_edges))
            
            if rng:
                return rng.choice(candidates)
            return candidates[0]

        attempts = 0
        while time.time() < end_time:
            attempts += 1
            attempt_rng = random.Random(seed + attempts * 12345 + 67890)

            remaining_edges = set(edges)
            cliques = []

            while remaining_edges:
                if time.time() > end_time - 1:
                    break

                # Pick the best seed edge
                seed_edge = find_best_seed_edge(remaining_edges, attempt_rng)
                
                if not seed_edge:
                    seed_edge = next(iter(remaining_edges))

                u, v = seed_edge
                current_clique = [u, v]

                # Candidates: vertices connected to all in current_clique
                # And those edges are in remaining_edges
                cand_set = set(adj[u])
                cand_set.intersection_update(adj[v])
                cand_set.discard(u)
                cand_set.discard(v)

                # Filter candidates: must have all edges to current_clique in remaining
                cand_set = {w for w in cand_set if has_all_edges_to_clique(w, current_clique, remaining_edges)}

                while cand_set:
                    # Pick the best candidate
                    # Metric: number of edges in remaining incident to w
                    best_w = None
                    max_deg = -1
                    for w in cand_set:
                        deg = count_remaining_neighbors(w, remaining_edges)
                        if deg > max_deg:
                            max_deg = deg
                            best_w = w

                    if best_w is None:
                        break

                    current_clique.append(best_w)
                    cand_set.remove(best_w)

                    # Update candidates: must be neighbor of best_w and have edges to all clique members
                    new_cand_set = set()
                    for w in cand_set:
                        if w in adj[best_w]:
                            e_bw = (min(w, best_w), max(w, best_w))
                            if e_bw in remaining_edges:
                                # w must have edges to all other clique members
                                if has_all_edges_to_clique(w, current_clique, remaining_edges):
                                    new_cand_set.add(w)
                    cand_set = new_cand_set

                # Evaluate this clique
                clique_edges = set()
                for i in range(len(current_clique)):
                    for j in range(i+1, len(current_clique)):
                        clique_edges.add((min(current_clique[i], current_clique[j]), max(current_clique[i], current_clique[j])))

                # Check if all these edges are actually in remaining_edges
                # They should be, by construction.
                if not clique_edges:
                    # Fallback if something went wrong
                    clique_edges = {seed_edge}
                    current_clique = list(seed_edge)

                cliques.append(current_clique)
                for e in clique_edges:
                    remaining_edges.discard(e)

            # Check if this solution is better
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques

            if time.time() > end_time - 2:
                break

        return best_cliques

    # Solve both instances
    t1 = time.time()
    # Allocate more time to the larger, harder instance (n=64)
    budget1 = min(30, DEADLINE - (t1 - start_time) - 5)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(110, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
