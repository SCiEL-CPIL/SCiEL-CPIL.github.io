"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
import time
import random
from collections import defaultdict, deque

# EVOLVE-BLOCK-START

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _remove_edge(adj, remaining, u, v):
    e = (min(u, v), max(u, v))
    remaining.discard(e)
    if v in adj[u]:
        adj[u].remove(v)
    if u in adj[v]:
        adj[v].remove(u)


def _find_longest_cycle_greedy(adj, n, seed, deadline=None):
    """
    Find a long cycle using greedy longest-path extension.
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    rng.shuffle(vertices)

    # Try multiple starting vertices (top by degree + random)
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:5] + vertices[:5]))
    rng.shuffle(starts)

    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Greedy longest path from start: always extend to unvisited neighbor
        # with highest degree (keeps options open for closing the cycle)
        path = [start]
        visited = {start}

        while True:
            u = path[-1]
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            candidates.sort(reverse=True)
            # Randomize among top-3 to add diversity
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            path.append(next_v)
            visited.add(next_v)

        # Check if we can close the cycle directly
        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)

        # Look for any back-edge (path[i], path[j]) with i < j-1 to form a cycle
        # Take the longest such cycle
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j + 1]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break  # Found the longest cycle starting at i

    return best_cycle


def _try_merge_into_cycle(pieces, edges_set, deadline=None):
    """
    Try to merge single-edge pieces into cycles by finding paths in the remaining graph.
    Returns a new list of pieces with potentially fewer total pieces.
    """
    if not pieces:
        return pieces

    # Identify single-edge pieces
    single_edges = []
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            single_edges.append((i, piece[0], piece[1]))

    if len(single_edges) < 2:
        return pieces

    # Build adjacency from single edges
    single_adj = defaultdict(set)
    for _, u, v in single_edges:
        single_adj[u].add(v)
        single_adj[v].add(u)

    # Try to find cycles in the single-edge subgraph
    # A cycle of length k in single edges can replace k single-edge pieces with 1 cycle piece
    new_pieces = list(pieces)
    used_single_indices = set()
    improvements = 0

    for idx, u, v in single_edges:
        if idx in used_single_indices:
            continue

        # Try to extend from u-v to form a cycle
        # Start with path [u, v] and try to extend
        best_cycle = None

        # Try extending from both ends
        for start, end in [(u, v), (v, u)]:
            if start not in single_adj or not single_adj[start]:
                continue

            # BFS/DFS to find a path from start back to end using unused single edges
            # We want a path start -> ... -> end where all edges are single edges not yet used
            visited = {start}
            queue = deque([(start, [start])])
            found = False

            while queue and not found:
                if deadline is not None and time.time() > deadline:
                    break
                node, path = queue.popleft()
                if len(path) > 15:  # Limit path length for efficiency
                    continue
                for neighbor in single_adj[node]:
                    if neighbor == end and len(path) >= 2:
                        # Found a cycle: path + [end]
                        cycle_path = path + [end]
                        if best_cycle is None or len(cycle_path) > len(best_cycle):
                            best_cycle = cycle_path
                        found = True
                        break
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

            if found:
                break

        if best_cycle and len(best_cycle) >= 3:
            # Extract the cycle edges and mark them as used
            cycle_edges = set()
            for i in range(len(best_cycle)):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(a, b), max(a, b))
                cycle_edges.add(e)

            # Verify all these edges are in our single-edge set
            valid = True
            for e in cycle_edges:
                if e not in edges_set:
                    valid = False
                    break

            if valid:
                # Find which single-edge pieces correspond to these edges
                edges_to_remove = set()
                for si, su, sv in single_edges:
                    if si in used_single_indices:
                        continue
                    e = (min(su, sv), max(su, sv))
                    if e in cycle_edges:
                        edges_to_remove.add(si)

                if len(edges_to_remove) == len(cycle_edges):
                    # Replace all these single edges with one cycle piece
                    for si in edges_to_remove:
                        new_pieces[si] = None
                        used_single_indices.add(si)

                    # Add the cycle as a new piece
                    new_pieces.append(best_cycle)
                    improvements += len(cycle_edges) - 1

    # Remove None entries
    final_pieces = [p for p in new_pieces if p is not None]
    return final_pieces


def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []

    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    global_deadline = start_time + time_limit

    # Initial greedy decomposition to have a baseline
    def greedy_decompose(remaining_edges, n, seed_offset=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        cycle_counter = 0
        while remaining:
            # Find a long cycle
            cycle = _find_longest_cycle_greedy(adj, n, seed + seed_offset + cycle_counter, deadline=global_deadline)
            if cycle:
                # Extract cycle edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        cycle_edges.add(e)
                    else:
                        cycle = None
                        break

                if cycle:
                    # Remove from remaining and adj
                    for e in cycle_edges:
                        u, v = e
                        _remove_edge(adj, remaining, u, v)
                    pieces.append(cycle)
                    cycle_counter += 1
                else:
                    # Cycle wasn't valid; fall through to single edge
                    pass

            if not cycle or len(cycle) < 3:
                # No cycle found, take a single edge
                # Pick an edge incident to a high-degree vertex to keep graph connected?
                # Or just any edge.
                # To maximize future cycles, maybe remove edges that are "bridges" or in small components?
                # Simple: pick random edge
                e = next(iter(remaining))
                u, v = e
                _remove_edge(adj, remaining, u, v)
                pieces.append([u, v])

        return pieces

    # Multi-start search
    # We use a larger number of restarts because the greedy method is fast
    for restart in range(500):
        if time.time() > global_deadline:
            break

        pieces = greedy_decompose(edges, n, restart)
        current_count = len(pieces)

        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces

        # Try to improve by merging single edges into cycles
        if best_pieces is not None and time.time() < global_deadline - 5:
            edges_set = set(edges)
            improved = _try_merge_into_cycle(best_pieces, edges_set, deadline=global_deadline - 2)
            if len(improved) < best_count:
                best_count = len(improved)
                best_pieces = improved

    if best_pieces is None:
        # Fallback: all edges as single pieces
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=40.0)
        results.append(pieces)
    return results

# EVOLVE-BLOCK-END
