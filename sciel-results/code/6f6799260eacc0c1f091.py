import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 150s limit

    n = 16
    all_vertices = list(range(n))
    all_edges = list(itertools.combinations(range(n), 2))
    total_possible = len(all_edges)  # 120

    # Q3 structure - 8 vertices, 12 edges
    # Using standard 3-cube representation
    Q3_ADJ = [
        [1, 2, 4],      # 0
        [0, 3, 5],      # 1
        [0, 3, 6],      # 2
        [1, 2, 7],      # 3
        [0, 5, 6],      # 4
        [1, 4, 7],      # 5
        [2, 4, 7],      # 6
        [3, 5, 6]       # 7
    ]

    Q3_EDGES = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    full_mask = (1 << n) - 1

    def has_q3_with_edge(u, v, adj_mask):
        """Check if adding edge (u,v) to graph creates a Q3 subgraph."""
        # Try all 12 edges of Q3 that could map to (u,v)
        for i, j in Q3_EDGES:
            if _check_q3_extension(i, u, j, v, adj_mask):
                return True
            if _check_q3_extension(j, u, i, v, adj_mask):
                return True
        return False

    def _check_q3_extension(q3_u, g_u, q3_v, g_v, adj_mask):
        """Check if there's a Q3 embedding with q3_u->g_u and q3_v->g_v."""
        mapping = {q3_u: g_u, q3_v: g_v}
        available = full_mask ^ (1 << g_u) ^ (1 << g_v)

        def backtrack(avail):
            if len(mapping) == 8:
                return True

            # Pick unassigned Q3 vertex with most mapped neighbors (MRV heuristic)
            best_q3 = -1
            best_score = -1
            for x in range(8):
                if x not in mapping:
                    score = 0
                    for nb in Q3_ADJ[x]:
                        if nb in mapping:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_q3 = x

            if best_q3 == -1:
                return False

            # Filter candidates
            candidates = avail
            for nb in Q3_ADJ[best_q3]:
                if nb in mapping:
                    candidates &= adj_mask[mapping[nb]]

            while candidates:
                bit = candidates & -candidates
                g_cand = bit.bit_length() - 1
                candidates ^= bit

                mapping[best_q3] = g_cand
                new_avail = avail & ~(1 << g_cand)

                if backtrack(new_avail):
                    return True

                del mapping[best_q3]

            return False

        return backtrack(available)

    def count_edges(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2

    def get_missing_edges(adj_mask):
        missing = []
        for u, v in all_edges:
            if not ((adj_mask[u] >> v) & 1):
                missing.append((u, v))
        return missing

    def find_blocking_q3(u, v, adj_mask):
        """Find a Q3 that blocks adding edge (u,v). Returns set of Q3 edges or None."""
        for i, j in Q3_EDGES:
            result = _find_q3_mapping(i, u, j, v, adj_mask)
            if result is not None:
                return _edges_from_mapping(result)
            result = _find_q3_mapping(j, u, i, v, adj_mask)
            if result is not None:
                return _edges_from_mapping(result)
        return None

    def _find_q3_mapping(q3_u, g_u, q3_v, g_v, adj_mask):
        """Find a full Q3 mapping with q3_u->g_u, q3_v->g_v. Returns mapping dict or None."""
        mapping = {q3_u: g_u, q3_v: g_v}
        available = full_mask ^ (1 << g_u) ^ (1 << g_v)

        def backtrack(avail):
            if len(mapping) == 8:
                return True

            best_q3 = -1
            best_score = -1
            for x in range(8):
                if x not in mapping:
                    score = 0
                    for nb in Q3_ADJ[x]:
                        if nb in mapping:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_q3 = x

            if best_q3 == -1:
                return False

            candidates = avail
            for nb in Q3_ADJ[best_q3]:
                if nb in mapping:
                    candidates &= adj_mask[mapping[nb]]

            while candidates:
                bit = candidates & -candidates
                g_cand = bit.bit_length() - 1
                candidates ^= bit

                mapping[best_q3] = g_cand
                if backtrack(avail & ~(1 << g_cand)):
                    return True
                del mapping[best_q3]

            return False

        if backtrack(available):
            return dict(mapping)
        return None

    def _edges_from_mapping(mapping):
        edges = set()
        for u, v in Q3_EDGES:
            gu, gv = mapping[u], mapping[v]
            edges.add((min(gu, gv), max(gu, gv)))
        return edges

    def greedy_hitting_set(families):
        """Greedy set cover: pick elements to cover all sets."""
        if not families:
            return set()

        all_elements = set()
        for f in families:
            all_elements.update(f)

        if not all_elements:
            return set()

        uncovered = list(families)
        solution = set()

        while uncovered:
            best_elem = None
            max_count = 0
            for e in all_elements:
                count = sum(1 for f in uncovered if e in f)
                if count > max_count:
                    max_count = count
                    best_elem = e

            if best_elem is None or max_count == 0:
                break

            solution.add(best_elem)
            uncovered = [f for f in uncovered if best_elem not in f]

        return solution

    def random_greedy_init(seed):
        """Random greedy construction with given seed."""
        rng = random.Random(seed)
        adj_mask = [0] * n
        edges = list(all_edges)
        rng.shuffle(edges)

        for u, v in edges:
            if not has_q3_with_edge(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)

        return adj_mask

    def local_search(adj_mask, max_time, rng):
        """Iterative improvement local search with hitting-set kicks."""
        current_count = count_edges(adj_mask)
        best_count = current_count
        best_adj = adj_mask[:]
        stuck = 0
        max_stuck = 50

        while time.time() < max_time and stuck < max_stuck:
            missing = get_missing_edges(adj_mask)
            if not missing:
                break

            rng.shuffle(missing)

            added = False
            # Try to add edges
            for u, v in missing:
                if time.time() > max_time:
                    break
                if not has_q3_with_edge(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    current_count += 1
                    added = True
                    stuck = 0
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj_mask[:]

            if not added:
                stuck += 1

                # Hitting-set kick: find blockers for several missing edges
                blockers = []
                for u, v in missing[:20]:
                    q3_edges = find_blocking_q3(u, v, adj_mask)
                    if q3_edges:
                        blockers.append(q3_edges)

                if blockers:
                    edges_to_remove = greedy_hitting_set(blockers)
                    if edges_to_remove:
                        for u, v in edges_to_remove:
                            if (adj_mask[u] >> v) & 1:
                                adj_mask[u] &= ~(1 << v)
                                adj_mask[v] &= ~(1 << u)
                                current_count -= 1
                        stuck = 0

            # Periodic polish
            if stuck % 5 == 0:
                missing2 = get_missing_edges(adj_mask)
                for u, v in missing2:
                    if not has_q3_with_edge(u, v, adj_mask):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj_mask[:]

        return best_adj, best_count

    def simulated_annealing(adj_mask, max_time, rng):
        """SA-like approach: accept worse moves with decreasing probability."""
        current_count = count_edges(adj_mask)
        best_count = current_count
        best_adj = adj_mask[:]

        temp = 5.0
        cooling = 0.995

        while time.time() < max_time and temp > 0.5:
            missing = get_missing_edges(adj_mask)
            if not missing:
                break

            u, v = rng.choice(missing)

            if not has_q3_with_edge(u, v, adj_mask):
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj_mask[:]
            else:
                # Can't add directly; try removing a random edge and adding this one
                existing = []
                for a, b in all_edges:
                    if (adj_mask[a] >> b) & 1:
                        existing.append((a, b))

                if existing:
                    ra, rb = rng.choice(existing)
                    # Remove (ra, rb)
                    adj_mask[ra] &= ~(1 << rb)
                    adj_mask[rb] &= ~(1 << ra)
                    current_count -= 1

                    # Try to add (u, v)
                    if not has_q3_with_edge(u, v, adj_mask):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj_mask[:]
                    else:
                        # Restore (ra, rb) since we couldn't improve
                        adj_mask[ra] |= (1 << rb)
                        adj_mask[rb] |= (1 << ra)
                        current_count += 1

            temp *= cooling

        return best_adj, best_count

    # Main search
    best_adj = None
    best_count = 0
    num_restarts = 40

    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break

        adj = random_greedy_init(r * 137 + 42)
        current_count = count_edges(adj)

        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]

        # Allocate remaining time for local search
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break

        # Run local search
        adj, cnt = local_search(adj, time.time() + min(remaining * 0.7, 8), random.Random(r * 7919 + 13))
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

        # If time remains, try SA polish
        remaining = deadline - time.time() - 1
        if remaining > 3:
            adj, cnt = simulated_annealing(adj, time.time() + min(remaining * 0.5, 5), random.Random(r * 104729 + 7))
            if cnt > best_count:
                best_count = cnt
                best_adj = adj[:]

    # Final polish pass
    if best_adj is not None and time.time() < deadline - 1:
        adj = best_adj[:]
        cnt = count_edges(adj)
        missing = get_missing_edges(adj)
        for u, v in missing:
            if time.time() > deadline - 0.5:
                break
            if not has_q3_with_edge(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                cnt += 1
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    if best_adj is None:
        # Fallback: empty graph
        return []

    result = []
    for u, v in all_edges:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])

    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
