"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        # adjacency for fast lookups
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # uncovered edges as a set of tuples (u, v) with u < v for O(1) membership
        uncovered = set((min(u, v), max(u, v)) for u, v in edges)
        result = []

        def time_check():
            return time.time() - start_time < DEADLINE

        # Helper: get vertices that still have uncovered edges
        def active_vertices():
            verts = set()
            for u, v in uncovered:
                verts.add(u)
                verts.add(v)
            return verts

        # Greedy clique extraction: start from an edge, extend to a maximal clique
        # using a heuristic that prioritizes vertices with many uncovered edges
        def extract_clique(start_edge):
            if not time_check():
                return None
            a, b = start_edge
            # start_edge is already a canonical tuple from uncovered
            if start_edge not in uncovered:
                return None
            # Build a clique greedily
            clique = [a, b]
            # Common neighbors of all clique members that are still active
            common = adj[a] & adj[b]
            # Only keep vertices that have uncovered edges
            active = active_vertices()
            common &= active
            # We'll iteratively add vertices
            while True:
                if not time_check():
                    break
                if not common:
                    break
                # Score each candidate: number of uncovered edges it would add
                best_v = None
                best_score = -1
                for v in common:
                    if v in clique:
                        continue
                    # Count uncovered edges from v to existing clique members
                    score = 0
                    for c in clique:
                        if frozenset((v, c)) in uncovered:
                            score += 1
                    # Also consider edges from v to other potential additions (heuristic: degree in common)
                    score += 0.1 * len(adj[v] & common)
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v is None:
                    break
                # Check that adding best_v keeps it a clique (all edges to clique exist)
                valid = True
                for c in clique:
                    edge_tuple = (min(best_v, c), max(best_v, c))
                    if edge_tuple not in uncovered:
                        valid = False
                        break
                if not valid:
                    # Remove from common and try next
                    common.discard(best_v)
                    continue
                clique.append(best_v)
                # Update common: intersection of adj[v] for all v in clique
                common = common & adj[best_v]
                common &= active
                common -= set(clique)

            if len(clique) < 2:
                return None
            return clique

        # Main loop: repeatedly extract cliques
        max_iter = 100000
        for _ in range(max_iter):
            if not time_check():
                break
            if not uncovered:
                break

            # Pick a starting edge: prefer edges in dense regions
            # Heuristic: pick edge with max sum of degrees of endpoints
            best_edge = None
            best_val = -1
            # Sample a subset for speed if too many edges
            edge_list = list(uncovered)
            if len(edge_list) > 200:
                sample = random.sample(edge_list, 200)
            else:
                sample = edge_list

            for e in sample:
                u, v = e[0], e[1]
                val = len(adj[u]) + len(adj[v])
                if val > best_val:
                    best_val = val
                    best_edge = e

            if best_edge is None:
                break

            clique = extract_clique(best_edge)
            if clique is None or len(clique) < 2:
                # Fallback: just take the edge
                clique = [best_edge[0], best_edge[1]]

            # Remove all edges within this clique from uncovered
            removed = False
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    edge_tuple = (min(u, v), max(u, v))
                    if edge_tuple in uncovered:
                        uncovered.discard(edge_tuple)
                        removed = True

            if removed:
                result.append(clique)

        # Any remaining uncovered edges become 2-cliques
        for u, v in uncovered:
            if not time_check():
                break
            result.append([u, v])

        return result

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() - start_time >= DEADLINE:
            # Fallback: trivial partition
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
        else:
            results.append(solve_instance(n, p, seed))

    return results
# EVOLVE-BLOCK-END
