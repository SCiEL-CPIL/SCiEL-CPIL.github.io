"""QUBO-formulated Q3-free graph maximization with tabu search."""
import time
import random
from itertools import combinations

def construct():
    t_start = time.time()
    DEADLINE = 142.0  # seconds wall-clock (150s kill, 8s margin)

    n = 16
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

    # All 8-subsets of vertices
    all_8sets = list(combinations(range(16), 8))
    num_8sets = len(all_8sets)

    # Precompute pair -> list of 8-sets containing that pair
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i + 1, 8):
                pr = (vs[i], vs[j])
                if pr not in pair_to_8sets:
                    pair_to_8sets[pr] = []
                pair_to_8sets[pr].append(vs)

    # All possible edges (120)
    all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    edge_index = {e: idx for idx, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 120

    def is_q3_in_8set(vs, adj):
        """Check if 8 vertices induce a Q3 using backtracking on isomorphism."""
        # Quick degree check: each vertex in Q3 has degree 3
        for i in range(8):
            d = 0
            u = vs[i]
            for j in range(8):
                if j != i:
                    if (adj[u] >> vs[j]) & 1:
                        d += 1
            if d < 3:
                return False

        # Backtrack isomorphism
        assignment = [-1] * 8
        used = [False] * 8

        def bt(qv):
            if qv == 8:
                return True
            for vi in range(8):
                if used[vi]:
                    continue
                ok = True
                u = vs[vi]
                for nb in q3_adj[qv]:
                    if nb < qv:
                        if not ((adj[u] >> vs[assignment[nb]]) & 1):
                            ok = False
                            break
                    else:
                        # For forward edges, just check it exists or will be checked later
                        # Actually we need to ensure the edge exists
                        pass
                if ok:
                    # Check all forward edges exist
                    for nb in q3_adj[qv]:
                        if nb > qv:
                            if not ((adj[u] >> vs[vi]) & 1):
                                ok = False
                                break
                    if ok:
                        assignment[qv] = vi
                        used[vi] = True
                        if bt(qv + 1):
                            return True
                        used[vi] = False
                        assignment[qv] = -1
            return False

        return bt(0)

    def count_q3s(adj):
        """Count all Q3 subgraphs in the graph defined by adj bitmasks."""
        cnt = 0
        for vs in all_8sets:
            if is_q3_in_8set(vs, adj):
                cnt += 1
        return cnt

    def build_adj(edge_set):
        adj = [0] * 16
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def check_edge_creates_q3(u, v, adj):
        """Check if adding edge (u,v) creates any Q3."""
        if u > v:
            u, v = v, u
        relevant = pair_to_8sets.get((u, v), [])
        # Temporarily add the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        found = False
        for vs in relevant:
            if is_q3_in_8set(vs, adj):
                found = True
                break
        # Remove the edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return found

    def greedy_complete(edge_set, adj):
        """Greedily add any edge that doesn't create a Q3."""
        changed = True
        while changed and time.time() < DEADLINE - 5:
            changed = False
            for (i, j) in all_edges:
                if (i, j) in edge_set:
                    continue
                if not check_edge_creates_q3(i, j, adj):
                    edge_set.add((i, j))
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    changed = True
        return edge_set, adj

    # Phase 1: Build initial Q3-free graph via random + repair
    random.seed(12345)
    best_set = set()
    best_adj = [0] * 16
    best_count = 0

    for attempt in range(80):
        if time.time() > DEADLINE - 40:
            break
        # Random dense graph
        cand = set()
        for e in all_edges:
            if random.random() < 0.75:
                cand.add(e)
        adj = build_adj(cand)

        # Repair: iteratively remove edges from Q3s
        max_iter = 200
        for _ in range(max_iter):
            q3_found = False
            for vs in all_8sets:
                if is_q3_in_8set(vs, adj):
                    q3_found = True
                    # Remove a random edge within this Q3 that exists
                    vs_list = list(vs)
                    random.shuffle(vs_list)
                    removed = False
                    for i in range(8):
                        for j in range(i + 1, 8):
                            u, v = vs_list[i], vs_list[j]
                            if (u, v) in cand:
                                cand.remove((u, v))
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                removed = True
                                break
                            elif (v, u) in cand:
                                cand.remove((v, u))
                                adj[v] &= ~(1 << u)
                                adj[u] &= ~(1 << v)
                                removed = True
                                break
                        if removed:
                            break
                    break
            if not q3_found:
                break

        if len(cand) > best_count:
            best_count = len(cand)
            best_set = set(cand)
            best_adj = adj[:]

    # Phase 2: Greedy completion
    best_set, best_adj = greedy_complete(best_set, best_adj)
    best_count = len(best_set)

    # Phase 3: QUBO-style tabu search
    # Objective: maximize |E| - lambda * (number of Q3s)
    # In practice we require 0 Q3s, so we work with edge swaps
    # Neighborhood: swap one edge for another (keep same size) or add/remove
    # Tabu list of recently moved edges

    current_set = set(best_set)
    current_adj = best_adj[:]
    current_count = len(current_set)
    best_set = set(current_set)
    best_adj = current_adj[:]
    best_count = current_count

    tabu_list = {}  # edge -> time when it became tabu
    tabu_tenure = 30
    iteration = 0

    while time.time() < DEADLINE - 8:
        iteration += 1

        # Choose move type: swap (remove 1, add 1) or add-only
        move_type = random.random()
        if move_type < 0.6:
            # Swap: remove one edge, add one different edge
            if not current_set:
                continue
            rem_e = random.choice(list(current_set))
            # Tabu check
            if tabu_list.get(rem_e, 0) > iteration - tabu_tenure:
                continue

            # Try to find a good edge to add
            candidates = [e for e in all_edges if e not in current_set and e != rem_e]
            random.shuffle(candidates)

            # Temporarily remove rem_e
            u0, v0 = rem_e
            current_set.remove(rem_e)
            current_adj[u0] &= ~(1 << v0)
            current_adj[v0] &= ~(1 << u0)

            found_add = False
            for add_e in candidates:
                if time.time() > DEADLINE - 8:
                    break
                u1, v1 = add_e
                if check_edge_creates_q3(u1, v1, current_adj):
                    continue
                # Accept this swap
                current_set.add(add_e)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                tabu_list[rem_e] = iteration
                tabu_list[add_e] = iteration
                found_add = True
                break

            if not found_add:
                # Revert: put rem_e back
                current_set.add(rem_e)
                current_adj[u0] |= (1 << v0)
                current_adj[v0] |= (1 << u0)

        elif move_type < 0.8:
            # Add one edge
            candidates = [e for e in all_edges if e not in current_set]
            if not candidates:
                continue
            random.shuffle(candidates)
            for add_e in candidates:
                if time.time() > DEADLINE - 8:
                    break
                u1, v1 = add_e
                if tabu_list.get(add_e, 0) > iteration - tabu_tenure:
                    continue
                if check_edge_creates_q3(u1, v1, current_adj):
                    continue
                current_set.add(add_e)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                tabu_list[add_e] = iteration
                break
        else:
            # Remove one edge (to escape local optimum, then greedily refill)
            if not current_set:
                continue
            rem_e = random.choice(list(current_set))
            u0, v0 = rem_e
            current_set.remove(rem_e)
            current_adj[u0] &= ~(1 << v0)
            current_adj[v0] &= ~(1 << u0)
            tabu_list[rem_e] = iteration

            # Greedily refill
            for fill_e in all_edges:
                if time.time() > DEADLINE - 8:
                    break
                if fill_e in current_set:
                    continue
                u1, v1 = fill_e
                if not check_edge_creates_q3(u1, v1, current_adj):
                    current_set.add(fill_e)
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)

        # Track best
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = set(current_set)
            best_adj = current_adj[:]

    # Final greedy pass on best
    final_set, final_adj = greedy_complete(set(best_set), best_adj[:])
    best_count = len(final_set)

    result = [[u, v] for (u, v) in sorted(final_set)]
    return result
