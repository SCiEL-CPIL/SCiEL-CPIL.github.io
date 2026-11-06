"""Q3-free graph on 16 vertices via tabu search with bitset Q3 detection."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_idx = {p: i for i, p in enumerate(all_pairs)}
    num_pairs = len(all_pairs)  # 120

    # Q3 structure: 8 vertices, 12 edges
    q3_vertices = [0, 1, 2, 3, 4, 5, 6, 7]
    q3_edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),
        (2,3),(2,6),
        (3,7),
        (4,5),(4,6),
        (5,7),
        (6,7)
    ]
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))

    # For each pair (u,v), precompute which 8-sets contain both u and v
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i + 1, 8):
                pair = (vs[i], vs[j])
                if pair not in pair_to_8sets:
                    pair_to_8sets[pair] = []
                pair_to_8sets[pair].append(vs)

    def contains_q3_in_8set(vs, adj_bits):
        """Check if 8 vertices vs induce a Q3 subgraph (not necessarily induced)."""
        # Build local adjacency
        local_adj = [0] * 8
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(8):
                if j != i and (mask >> vs[j]) & 1:
                    local_adj[i] |= (1 << j)

        # Each vertex in Q3 has degree exactly 3
        for i in range(8):
            if bin(local_adj[i]).count('1') != 3:
                return False

        # Backtrack to check isomorphism to Q3
        assignment = [-1] * 8
        vs_used = [False] * 8

        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for vs_i in range(8):
                if vs_used[vs_i]:
                    continue
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        if not ((local_adj[vs_i] >> assignment[nb]) & 1):
                            ok = False
                            break
                if ok:
                    # Also check that no extra edges exist to already-assigned vertices
                    # beyond what Q3 requires (for non-induced subgraph, we just need the Q3 edges to exist)
                    # Actually for "contains Q3 as subgraph", we need all Q3 edges to be present
                    # The degree check above ensures exactly 3 neighbors within the 8-set
                    assignment[q3_v] = vs_i
                    vs_used[vs_i] = True
                    if backtrack(q3_v + 1):
                        return True
                    vs_used[vs_i] = False
                    assignment[q3_v] = -1
            return False

        return backtrack(0)

    def check_edge_creates_q3(u, v, adj_bits):
        """Check if adding edge (u,v) creates a Q3. Only check 8-sets containing u and v."""
        if u > v:
            u, v = v, u
        relevant_8sets = pair_to_8sets.get((u, v), [])
        for vs in relevant_8sets:
            if contains_q3_in_8set(vs, adj_bits):
                return True
        return False

    def edges_to_adj_bits(edge_set):
        adj = [0] * 16
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_q3_free(edge_set):
        adj = edges_to_adj_bits(edge_set)
        for vs in all_8sets:
            if contains_q3_in_8set(vs, adj):
                return False
        return True

    def greedy_complete(edge_set, adj_bits):
        """Greedily add any possible edge that doesn't create Q3."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(16):
                for j in range(i + 1, 16):
                    if (i, j) in edge_set:
                        continue
                    new_adj = adj_bits[:]
                    new_adj[i] |= (1 << j)
                    new_adj[j] |= (1 << i)
                    if not check_edge_creates_q3(i, j, new_adj):
                        adj_bits = new_adj
                        edge_set.add((i, j))
                        changed = True
                        if time.time() > deadline - 5:
                            break
                if time.time() > deadline - 5:
                    break
        return edge_set, adj_bits

    # Multi-start initialization
    random.seed(42)
    best_edges = set()
    best_adj = [0] * 16
    best_count = 0

    densities = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    for attempt in range(60):
        if time.time() > deadline - 30:
            break
        density = densities[attempt % len(densities)]
        candidate = set()
        for e in all_pairs:
            if random.random() < density:
                candidate.add(e)

        # Remove edges to make Q3-free
        adj = edges_to_adj_bits(candidate)
        q3_count = 0
        for vs in all_8sets:
            if contains_q3_in_8set(vs, adj):
                q3_count += 1
                vs_list = list(vs)
                random.shuffle(vs_list)
                removed = False
                for i in range(8):
                    if removed:
                        break
                    for j in range(i + 1, 8):
                        u, v = vs_list[i], vs_list[j]
                        if (u, v) in candidate:
                            candidate.remove((u, v))
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            removed = True
                            break
                if q3_count > 15:
                    break

        if q3_count <= 15:
            # Greedy complete
            candidate, adj = greedy_complete(candidate, adj)
            if len(candidate) > best_count:
                best_count = len(candidate)
                best_edges = set(candidate)
                best_adj = adj[:]

    # Tabu search
    current_edges = set(best_edges)
    current_adj = best_adj[:]
    current_count = len(current_edges)

    # Tabu list: store (edge_to_remove, edge_to_add) moves
    tabu_list = []
    tabu_tenure = 10
    iteration = 0

    # Precompute neighbor edges for each edge (edges sharing a vertex)
    edge_neighbors = {}
    for idx, (u, v) in enumerate(all_pairs):
        neighbors = []
        for idx2, (u2, v2) in enumerate(all_pairs):
            if idx == idx2:
                continue
            if u == u2 or u == v2 or v == u2 or v == v2:
                neighbors.append(idx2)
        edge_neighbors[idx] = neighbors

    while time.time() < deadline - 5:
        iteration += 1

        # Generate candidate moves: swap one edge for another
        best_move = None
        best_score = current_count
        best_new_edges = None
        best_new_adj = None

        # Sample a subset of moves for efficiency
        num_moves_to_try = 200
        tried_moves = set()

        for _ in range(num_moves_to_try):
            # Pick a random edge to remove
            if not current_edges:
                break
            remove_e = random.choice(list(current_edges))
            ru, rv = remove_e

            # Pick a random edge to add (not already present)
            add_e = random.choice(all_pairs)
            au, av = add_e

            if add_e in current_edges:
                continue
            if (au, av) == (ru, rv) or (au, av) == (rv, ru):
                continue

            # Check tabu
            move_key = (min(ru, rv), max(ru, rv), min(au, av), max(au, av))
            if move_key in tried_moves:
                continue
            tried_moves.add(move_key)

            if move_key in tabu_list:
                # Aspiration: accept if it improves best
                pass  # We'll still consider it if it improves

            # Apply move tentatively
            new_adj = current_adj[:]
            new_adj[ru] &= ~(1 << rv)
            new_adj[rv] &= ~(1 << ru)
            new_adj[au] |= (1 << av)
            new_adj[av] |= (1 << au)

            # Check if the added edge creates a Q3
            if check_edge_creates_q3(au, av, new_adj):
                continue

            new_count = current_count  # Same number of edges (swap)
            # Actually count might change if add_e was already there (it's not)
            # So new_count = current_count

            if new_count > best_score:
                best_score = new_count
                best_move = (remove_e, add_e)
                best_new_edges = current_edges - {remove_e} | {add_e}
                best_new_adj = new_adj

        # If no improving move found, try a random non-tabu move (to escape local optima)
        if best_move is None:
            # Try random swap
            for _ in range(50):
                if not current_edges:
                    break
                remove_e = random.choice(list(current_edges))
                ru, rv = remove_e
                add_e = random.choice(all_pairs)
                au, av = add_e

                if add_e in current_edges:
                    continue
                if (au, av) == (ru, rv) or (au, av) == (rv, ru):
                    continue

                move_key = (min(ru, rv), max(ru, rv), min(au, av), max(au, av))
                if move_key in tabu_list:
                    continue

                new_adj = current_adj[:]
                new_adj[ru] &= ~(1 << rv)
                new_adj[rv] &= ~(1 << ru)
                new_adj[au] |= (1 << av)
                new_adj[av] |= (1 << au)

                if check_edge_creates_q3(au, av, new_adj):
                    continue

                best_move = (remove_e, add_e)
                best_new_edges = current_edges - {remove_e} | {add_e}
                best_new_adj = new_adj
                break

        if best_move is not None:
            remove_e, add_e = best_move
            current_edges = best_new_edges
            current_adj = best_new_adj
            current_count = len(current_edges)

            # Update tabu list
            ru, rv = remove_e
            au, av = add_e
            move_key = (min(ru, rv), max(ru, rv), min(au, av), max(au, av))
            tabu_list.append(move_key)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

            if current_count > best_count:
                best_count = current_count
                best_edges = set(current_edges)
                best_adj = current_adj[:]

            # Occasionally try to add an extra edge (increase size)
            if random.random() < 0.1 and time.time() < deadline - 5:
                # Try to add a random missing edge
                missing = [e for e in all_pairs if e not in current_edges]
                if missing:
                    add_e = random.choice(missing)
                    au, av = add_e
                    new_adj = current_adj[:]
                    new_adj[au] |= (1 << av)
                    new_adj[av] |= (1 << au)
                    if not check_edge_creates_q3(au, av, new_adj):
                        current_adj = new_adj
                        current_edges.add(add_e)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_edges = set(current_edges)
                            best_adj = current_adj[:]

        else:
            # No move found, try greedy completion
            current_edges, current_adj = greedy_complete(current_edges, current_adj)
            current_count = len(current_edges)
            if current_count > best_count:
                best_count = current_count
                best_edges = set(current_edges)
                best_adj = current_adj[:]

    # Final greedy pass
    best_edges, best_adj = greedy_complete(best_edges, best_adj)
    best_count = len(best_edges)

    result = [[u, v] for (u, v) in best_edges]
    return result
