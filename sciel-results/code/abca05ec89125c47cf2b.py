import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _generate_cycle_pool(adj, n, rng, max_cycles=200, max_len=None):
    """Generate a diverse pool of simple cycles using randomized DFS."""
    if max_len is None:
        max_len = n
    pool = []
    seen = set()
    attempts = 0
    max_attempts = max(max_cycles * 10, 2000)

    while len(pool) < max_cycles and attempts < max_attempts:
        attempts += 1
        # Pick start vertex
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break
        start = rng.choice(candidates)

        # Randomized DFS to find a cycle
        path = [start]
        visited = {start}
        current = start
        found = False

        for step in range(max_len - 1):
            nbrs = [v for v in adj[current] if v not in visited]
            if not nbrs:
                # Stuck; check if we can close
                if len(path) >= 3 and start in adj[current]:
                    cycle = path[:]
                    key = frozenset(cycle)
                    if key not in seen:
                        seen.add(key)
                        pool.append(cycle)
                    found = True
                break

            # Check if we can close a cycle (and it's long enough)
            can_close = (len(path) >= 3 and start in adj[current])
            if can_close and rng.random() < 0.15:
                # Close now
                cycle = path[:]
                key = frozenset(cycle)
                if key not in seen:
                    seen.add(key)
                    pool.append(cycle)
                found = True
                break

            # Choose next: bias toward lower-degree vertices (to avoid dead ends)
            nbrs.sort(key=lambda x: (len(adj[x]), rng.random()))
            # Occasionally pick a random neighbor for diversity
            if rng.random() < 0.3:
                nxt = rng.choice(nbrs)
            else:
                nxt = nbrs[0]
            path.append(nxt)
            visited.add(nxt)
            current = nxt

        # If we reached max_len without closing, try to close
        if not found and len(path) >= 3 and start in adj[current]:
            cycle = path[:]
            key = frozenset(cycle)
            if key not in seen:
                seen.add(key)
                pool.append(cycle)

    return pool


def _solve_set_packing(cycles, edges_set, n, deadline):
    """
    Select edge-disjoint cycles from the pool to minimize total pieces.
    Pieces = num_cycles_selected + num_uncovered_edges.
    Equivalent: maximize (sum of cycle_lengths - num_cycles_selected)
    subject to edge-disjointness.
    
    Uses greedy with local search (2-opt style swaps).
    """
    # Precompute edge index for each cycle
    def cycle_edges(cyc):
        m = len(cyc)
        return frozenset((min(cyc[i], cyc[(i+1) % m]), max(cyc[i], cyc[(i+1) % m])) for i in range(m))

    cycle_edge_sets = [cycle_edges(c) for c in cycles]
    m = len(cycles)

    # Greedy: pick cycles in order of decreasing "efficiency" (length - 1 benefit per edge)
    # Actually, each cycle of length L uses L edges and saves (L-1) pieces vs singles.
    # So benefit = L - 1.
    # Sort by benefit/edge_count = (L-1)/L, which is increasing in L.
    # So prefer longer cycles.
    order = sorted(range(m), key=lambda i: len(cycles[i]), reverse=True)

    selected = []
    used_edges = set()
    for idx in order:
        ce = cycle_edge_sets[idx]
        if not ce & used_edges:
            selected.append(idx)
            used_edges |= ce

    # Local search: try to replace selected cycles with different ones
    # and try to add unselected cycles
    improved = True
    max_iters = 50
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        iters += 1
        improved = False

        # Try adding an unselected cycle
        for idx in range(m):
            if idx in selected:
                continue
            ce = cycle_edge_sets[idx]
            if not ce & used_edges:
                selected.append(idx)
                used_edges |= ce
                improved = True
                break

        if improved:
            continue

        # Try swapping: remove one selected, add one or more unselected
        for si, sidx in enumerate(selected):
            s_ce = cycle_edge_sets[sidx]
            s_len = len(cycles[sidx])
            # Try removing this and seeing if we can add better cycles
            new_used = used_edges - s_ce
            # Try adding unselected cycles to new_used
            added = []
            for idx in range(m):
                if idx == sidx or idx in selected:
                    continue
                ce = cycle_edge_sets[idx]
                if not ce & new_used:
                    added.append(idx)
                    new_used |= ce
            # Compare: old had sidx (length s_len). New has 'added' cycles.
            old_benefit = s_len - 1
            new_benefit = sum(len(cycles[a]) - 1 for a in added)
            if new_benefit > old_benefit and added:
                selected.remove(sidx)
                selected.extend(added)
                used_edges = new_used
                improved = True
                break

    # Compute result
    pieces = [cycles[i] for i in selected]
    remaining_edges = edges_set - used_edges
    for e in sorted(remaining_edges):
        pieces.append([e[0], e[1]])
    return pieces


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    edges_set = frozenset(edges)
    base_seed = 7

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Phase 1: Generate cycle pool with multiple seeds
    pool_deadline = deadline - 5.0  # Leave time for solving
    for seed in range(20):
        if time.time() > pool_deadline:
            break
        rng = random.Random(base_seed + seed)
        pool = _generate_cycle_pool(adj, n, rng, max_cycles=150)
        if not pool:
            continue
        # Solve set packing
        pieces = _solve_set_packing(pool, edges_set, n, deadline)
        cnt = len(pieces)
        if cnt < best_count:
            best_count = cnt
            best = pieces

    # Fallback: if no cycles found, return all edges as singles
    if best is None:
        best = [list(e) for e in edges]

    return best


def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    return results
