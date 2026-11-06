"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 145.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycle_from(adj, start, remaining):
    """
    Find a simple cycle where 'start' is the smallest vertex in the cycle,
    using only edges in remaining. Returns list of vertices or None.
    Uses iterative DFS with explicit stack for reliability.
    """
    # Get valid neighbors of start (v > start, edge in remaining)
    start_nbrs = []
    for v in sorted(adj[start]):
        if v > start:
            e = (start, v)
            if e in remaining:
                start_nbrs.append(v)
    if len(start_nbrs) < 2:
        return None

    # Iterative DFS
    path = [start]
    visited = {start}
    # neighbor_indices tracks position in each vertex's neighbor list
    nbr_indices = [0]
    # Cache valid neighbors for each vertex
    valid_nbrs_cache = {}

    def get_valid_nbrs(u):
        if u in valid_nbrs_cache:
            return valid_nbrs_cache[u]
        result = []
        for v in sorted(adj[u]):
            if v < start:
                continue
            e = (min(u, v), max(u, v))
            if e not in remaining:
                continue
            result.append(v)
        valid_nbrs_cache[u] = result
        return result

    valid_nbrs_cache[start] = start_nbrs

    while True:
        u = path[-1]
        if u not in valid_nbrs_cache:
            get_valid_nbrs(u)
        nbrs = valid_nbrs_cache[u]
        idx = nbr_indices[-1]

        found = False
        while idx < len(nbrs):
            v = nbrs[idx]
            nbr_indices[-1] = idx + 1
            if v == start:
                if len(path) >= 3:
                    return path[:]
            elif v not in visited:
                visited.add(v)
                path.append(v)
                nbr_indices.append(0)
                found = True
                break
            idx += 1

        if found:
            continue

        # Backtrack
        if len(path) == 1:
            return None
        visited.discard(path[-1])
        path.pop()
        nbr_indices.pop()


def _greedy_decompose(edges, n, rng, time_limit, strategy='longest'):
    """
    Greedily find edge-disjoint cycles. Returns list of cycles.
    strategy: 'longest' or 'random'
    """
    if not edges:
        return []

    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    adj = _build_adj(remaining)
    cycles = []
    start_time = time.time()

    while remaining:
        if time.time() - start_time > time_limit:
            break

        best_cycle = None
        best_len = 0

        # Determine which vertices to try as start
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break

        if strategy == 'random':
            rng.shuffle(candidates)
        else:
            # Sort by degree descending to find longer cycles first
            candidates.sort(key=lambda v: -len(adj[v]))

        for start in candidates:
            if time.time() - start_time > time_limit:
                break

            # Quick check: does start have at least 2 neighbors in remaining?
            count = 0
            for v in adj[start]:
                if v > start:
                    e = (start, v)
                    if e in remaining:
                        count += 1
                        if count >= 2:
                            break
            if count < 2:
                continue

            cycle = _find_cycle_from(adj, start, remaining)
            if cycle:
                if strategy == 'longest':
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                        if best_len >= n:
                            break
                else:  # random
                    best_cycle = cycle
                    break

        if best_cycle:
            cycles.append(best_cycle)
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            break

    return cycles


def _score_cycles(cycles):
    """Score: sum(|c| - 1) for all cycles. Higher is better."""
    return sum(len(c) - 1 for c in cycles)


def _cycles_to_pieces(cycles, edges):
    """Convert cycles to pieces, adding remaining edges as single-edge pieces."""
    pieces = []
    covered = set()
    for cycle in cycles:
        pieces.append(cycle)
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            covered.add((min(u, v), max(u, v)))

    for u, v in edges:
        e = (min(u, v), max(u, v))
        if e not in covered:
            pieces.append([u, v])
    return pieces


def _simulate_annealing(initial_cycles, edges, n, rng, time_limit):
    """
    Simulated annealing over cycle sets.
    Move: pick a random cycle, remove it, re-greedy-fill the freed edges.
    Accept if better, or with probability exp(-delta/T) if worse.
    """
    current = list(initial_cycles)
    current_score = _score_cycles(current)
    best = list(current)
    best_score = current_score

    start_time = time.time()
    T = 10.0  # Initial temperature
    cooling = 0.999

    while time.time() - start_time < time_limit:
        if not current:
            break

        # Pick a random cycle to remove
        idx = rng.randrange(len(current))
        removed = current[idx]
        temp = current[:idx] + current[idx + 1:]

        # Edges covered by temp cycles
        covered = set()
        for c in temp:
            for j in range(len(c)):
                u = c[j]
                v = c[(j + 1) % len(c)]
                covered.add((min(u, v), max(u, v)))

        # Available edges
        all_edges = set((min(u, v), max(u, v)) for u, v in edges)
        available = all_edges - covered

        if not available:
            T *= cooling
            continue

        # Re-greedy-fill with random strategy
        new_cycles = _greedy_decompose(available, n, rng, time_limit * 0.1, strategy='random')
        new_total = temp + new_cycles
        new_score = _score_cycles(new_total)

        delta = new_score - current_score
        if delta >= 0 or rng.random() < pow(2.718281828, delta / max(T, 0.001)):
            current = new_total
            current_score = new_score
            if current_score > best_score:
                best = list(current)
                best_score = current_score

        T *= cooling

    return best


def _decompose_one_graph(edges, n, time_limit):
    """Decompose a single graph, returning list of pieces."""
    if not edges:
        return []

    rng = random.Random(12345 + n)
    start_time = time.time()

    # Phase 1: Multiple greedy initializations, keep best
    best_cycles = []
    best_score = -1

    # Try different strategies and random seeds
    for strategy in ['longest', 'random']:
        for trial in range(3):
            if time.time() - start_time > time_limit * 0.3:
                break
            trial_rng = random.Random(1000 + strategy[0] * 100 + trial)
            cycles = _greedy_decompose(edges, n, trial_rng, time_limit * 0.05, strategy=strategy)
            score = _score_cycles(cycles)
            if score > best_score:
                best_score = score
                best_cycles = cycles

    # Phase 2: Simulated annealing refinement
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2:
        sa_rng = random.Random(99999 + n)
        refined = _simulate_annealing(best_cycles, edges, n, sa_rng, remaining_time)
        if _score_cycles(refined) > best_score:
            best_cycles = refined

    # Convert to pieces
    pieces = _cycles_to_pieces(best_cycles, edges)
    return pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 10:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        graph_time = min(remaining_time * 0.9, 45.0)

        pieces = _decompose_one_graph(edges, n, graph_time)

        # Validate
        used_edges = set()
        valid = True
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break

        if not valid:
            results.append([[u, v] for u, v in edges])
        else:
            results.append(pieces)

    return results
