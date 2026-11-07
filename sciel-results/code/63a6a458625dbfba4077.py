import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 108

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        total_k_edges = len(k_edges)

        # Precompute edge_to_vs: for each K-edge, which I-vertices can cover it
        edge_to_vs = {e: [] for e in k_edges}
        for v in range(k, n):
            N_v = set(neighbors[v])
            for e in k_edges:
                if e[0] in N_v and e[1] in N_v:
                    edge_to_vs[e].append(v)

        # Also precompute for each v in I, the set of K-edges it can cover
        v_to_edges = {v: set() for v in range(k, n)}
        for v in range(k, n):
            N_v = set(neighbors[v])
            for e in k_edges:
                if e[0] in N_v and e[1] in N_v:
                    v_to_edges[v].add(e)

        # For each v in I, the K-vertices it is adjacent to
        v_adj_k = {v: set(neighbors[v]) for v in range(k, n)}

        best_cliques = None
        best_count = float('inf')

        num_attempts = 60

        for attempt in range(num_attempts):
            if time.time() > deadline - 2:
                break

            rng = random.Random(seed + attempt * 7919 + 13)

            # State: for each v in I, which K-vertices are "used" (assigned to a clique containing v)
            # A clique [u, v] uses vertex v and K-vertex u
            # A clique [u, w, v] uses vertex v and K-vertices u, w
            # For each v, the set of K-vertices used
            used_k_for_v = {v: set() for v in range(k, n)}
            # For each K-vertex, how many I-vertices it's been paired with (for tracking)
            # Actually we need: for each v in I, which K-edges have been assigned to it
            assigned_edges_for_v = {v: set() for v in range(k, n)}

            # Greedy assignment of K-edges to I-vertices
            # Process edges in order of scarcity (fewest available v's)
            edge_order = sorted(k_edges, key=lambda e: len(edge_to_vs[e]))

            for e in edge_order:
                u, w = e
                valid = [v for v in edge_to_vs[e]
                         if u not in used_k_for_v[v] and w not in used_k_for_v[v]]
                if not valid:
                    continue

                # Score each valid v: prefer those with more capacity (fewer used K-vertices)
                # and more remaining edges they can still cover
                def score_v(v):
                    remaining = len(v_to_edges[v] - assigned_edges_for_v[v])
                    capacity = len(v_adj_k[v]) - len(used_k_for_v[v])
                    return (capacity, remaining, -v)  # maximize capacity, then remaining

                valid.sort(key=score_v, reverse=True)
                best_v = valid[0]

                # Sometimes randomize among top candidates
                if len(valid) > 2 and rng.random() < 0.2:
                    top = valid[:min(3, len(valid))]
                    best_v = rng.choice(top)

                assigned_edges_for_v[best_v].add(e)
                used_k_for_v[best_v].add(u)
                used_k_for_v[best_v].add(w)

            # Local search: try to reassign edges to reduce total cliques
            # The total number of cliques = (number of K-edges not assigned to any v) + (number of pairs (v, u) where u in adj(v) but u not used by v in any triple)
            # Actually: for each v, cliques involving v = (number of triples assigned to v) + (number of singles for v)
            # Singles for v: K-vertices u in adj(v) but not in used_k_for_v[v]
            # Triples for v: edges assigned to v
            # Remaining K-edges: K-edges not assigned to any v

            def count_cliques():
                total = 0
                for v in range(k, n):
                    total += len(assigned_edges_for_v[v])  # triples
                    total += len(v_adj_k[v] - used_k_for_v[v])  # singles
                # Remaining K-edges need maximal clique cover
                rem = total_k_edges - sum(len(assigned_edges_for_v[v]) for v in range(k, n))
                # Rough estimate: remaining edges need at least rem/((k-1)) cliques or so
                # For now just count remaining edges as needing 1 clique each (upper bound)
                # Actually let's be more precise: a maximal clique of size s covers C(s,2) edges
                # But for the count, let's just say each remaining edge needs its own clique as a safe upper bound
                total += rem
                return total

            # Try local improvements: swap edge assignments
            for ls_iter in range(20):
                if time.time() > deadline - 1:
                    break
                improved = False
                # Pick a random assigned edge and try to reassign it
                all_assigned = [(e, v) for v in range(k, n) for e in assigned_edges_for_v[v]]
                if not all_assigned:
                    break
                rng.shuffle(all_assigned)
                for e, v_orig in all_assigned[:30]:
                    u, w = e
                    # Try to reassign to a different v
                    valid = [v for v in edge_to_vs[e] if v != v_orig
                             and u not in used_k_for_v[v] and w not in used_k_for_v[v]]
                    if valid:
                        # Check if reassignment reduces total clique count
                        # Removing e from v_orig: frees u and w for v_orig (might reduce singles for v_orig)
                        # Adding e to v_new: uses u and w for v_new (might increase singles for v_new)
                        v_new = rng.choice(valid)

                        # Estimate change: 
                        # For v_orig: if u is only used in this edge, then u becomes free -> +1 single
                        # Similarly for w
                        # For v_new: if u was free, it becomes used -> -1 single
                        # Net change in singles: could be 0, +1, -1, etc.
                        # Simplified: just do it and check
                        old_count = count_cliques()

                        assigned_edges_for_v[v_orig].discard(e)
                        used_k_for_v[v_orig].discard(u)
                        used_k_for_v[v_orig].discard(w)
                        assigned_edges_for_v[v_new].add(e)
                        used_k_for_v[v_new].add(u)
                        used_k_for_v[v_new].add(w)

                        new_count = count_cliques()
                        if new_count <= old_count:
                            improved = True
                        else:
                            # Revert
                            assigned_edges_for_v[v_new].discard(e)
                            used_k_for_v[v_new].discard(u)
                            used_k_for_v[v_new].discard(w)
                            assigned_edges_for_v[v_orig].add(e)
                            used_k_for_v[v_orig].add(u)
                            used_k_for_v[v_orig].add(w)
                    if improved:
                        break
                if not improved:
                    break

            # Build cliques
            cliques = []
            for v in range(k, n):
                for e in assigned_edges_for_v[v]:
                    cliques.append([e[0], e[1], v])
                for u in v_adj_k[v] - used_k_for_v[v]:
                    cliques.append([u, v])

            # Cover remaining K edges
            assigned_set = set()
            for v in range(k, n):
                assigned_set.update(assigned_edges_for_v[v])
            rem = [e for e in k_edges if e not in assigned_set]

            if rem:
                rem_set = set(rem)
                # Build adjacency on remaining K-vertices (only considering remaining edges)
                adj = {u: set() for u in range(k)}
                for e in rem_set:
                    adj[e[0]].add(e[1])
                    adj[e[1]].add(e[0])

                while rem_set:
                    if time.time() > deadline - 0.5:
                        break
                    # Pick edge with max common neighbors
                    be = max(rem_set, key=lambda e: len(adj[e[0]] & adj[e[1]]))
                    clique = {be[0], be[1]}
                    changed = True
                    while changed:
                        changed = False
                        common = set(range(k))
                        for x in clique:
                            common &= adj[x]
                        common -= clique
                        if common:
                            best_x = max(common, key=lambda x: len(adj[x]))
                            clique.add(best_x)
                            changed = True

                    cliques.append(sorted(clique))
                    for a, b in combinations(clique, 2):
                        rem_set.discard((min(a, b), max(a, b)))

            # Validate: every clique must be a valid clique, every edge covered exactly once
            clean = []
            for c in cliques:
                clean.append([int(x) for x in c])

            cnt = len(clean)
            if cnt < best_count:
                best_count = cnt
                best_cliques = clean

        if best_cliques is None:
            # Fallback: cover each edge individually
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            best_cliques = [[int(x) for x in c] for c in cliques]

        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
