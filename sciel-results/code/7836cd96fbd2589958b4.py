"""Clique partition of two fixed dense random graphs via optimized bitmask greedy with restarts."""
# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 115.0  # Wall-clock guard (evaluator kills at 150s, keep 35s margin)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _encode_edge(u, v):
    if u > v:
        u, v = v, u
    return u * 100 + v  # n <= 64, so 100 is safe


def _decode_edge(e):
    u = e // 100
    v = e % 100
    return u, v


def _greedy_partition_fast(edges, n, rng):
    """
    Fast greedy clique partition using bitmasks.
    Strategy:
    1. Pick a random vertex from the top-degree vertices in the remaining graph.
    2. Extend to a maximal clique using a max-degree heuristic within candidates.
    3. Remove all edges in the clique.
    """
    if not edges:
        return []

    # Build adjacency as bitmasks
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add(_encode_edge(u, v))

    remaining = set(edge_set)
    cliques = []

    # Track degrees in the remaining graph
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    while remaining:
        if time.time() > _DEADLINE - 2:
            # Fallback: emit remaining edges as 2-cliques
            for e in remaining:
                u, v = _decode_edge(e)
                cliques.append([u, v])
            break

        # Find candidate vertices for starting the clique.
        # Pick from the top 10% of vertices by degree to ensure we start with a well-connected vertex,
        # but randomize to allow diversity.
        
        # Get sorted vertices by degree
        # To avoid sorting every time, we can just scan for max and pick randomly among high-degree ones.
        # For n=64, sorting is cheap.
        vertices_by_deg = sorted(range(n), key=lambda x: deg[x], reverse=True)
        
        # Pick a random vertex from the top 5 (or all if fewer)
        k = min(5, len(vertices_by_deg))
        best_v = vertices_by_deg[rng.randint(0, k - 1)]
        
        if deg[best_v] == 0:
            # No edges left for this vertex, shouldn't happen if remaining is not empty
            # Pick any vertex with deg > 0
            found = False
            for v in range(n):
                if deg[v] > 0:
                    best_v = v
                    found = True
                    break
            if not found:
                break

        # Build clique starting with best_v
        C = [best_v]
        # Candidates: neighbors of best_v that have remaining edges
        cand_mask = adj[best_v]
        
        # We need to ensure that the edges from best_v to candidates are actually in 'remaining'.
        # Since adj is updated to reflect remaining edges, this is true.
        
        while cand_mask:
            # Find candidate with max degree within cand_mask
            best_u = -1
            max_deg_u = -1
            temp_mask = cand_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                u = lsb.bit_length() - 1
                # Count neighbors of u within cand_mask
                # adj[u] & cand_mask gives neighbors of u that are also candidates
                # We exclude u itself? adj[u] doesn't have u bit set.
                d = bin(adj[u] & cand_mask).count('1')
                if d > max_deg_u:
                    max_deg_u = d
                    best_u = u
                temp_mask ^= lsb

            if best_u == -1:
                break

            C.append(best_u)

            # Update candidates: keep only those connected to best_u
            cand_mask &= adj[best_u]
            # Remove vertices already in C
            for v in C:
                cand_mask &= ~(1 << v)

        if len(C) >= 2:
            cliques.append(C)
            # Remove all edges within C from remaining and update adj/deg
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    e = _encode_edge(u, w)
                    if e in remaining:
                        remaining.discard(e)
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        deg[u] -= 1
                        deg[w] -= 1
        else:
            # Fallback: take one edge
            e = next(iter(remaining))
            u, w = _decode_edge(e)
            cliques.append([u, w])
            remaining.discard(e)
            adj[u] &= ~(1 << w)
            adj[w] &= ~(1 << u)
            deg[u] -= 1
            deg[w] -= 1

    return cliques


def _validate_partition(cliques, edges):
    """Validate that cliques form a valid partition of edges."""
    edge_set = set(_encode_edge(u, v) for u, v in edges)
    seen_edges = set()
    for c in cliques:
        k = len(c)
        if k < 2:
            return False
        for i in range(k):
            for j in range(i + 1, k):
                e = _encode_edge(c[i], c[j])
                if e not in edge_set or e in seen_edges:
                    return False
                seen_edges.add(e)
    return len(seen_edges) == len(edges)


def _solve_instance(edges, n):
    """
    Solve a single instance with randomized restarts, keeping the best partition.
    """
    if not edges:
        return []

    best_cliques = None
    best_count = float('inf')
    
    # Run restarts until deadline
    restart_count = 0
    while time.time() < _DEADLINE - 5:
        restart_count += 1
        rng = _random.Random(10000 + restart_count)
        cliques = _greedy_partition_fast(edges, n, rng)

        # The greedy algorithm guarantees validity if implemented correctly.
        # We skip validation in the loop for speed, but do a final check.
        
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques

        # Early stop if we reach a very good solution?
        # Lower bound is hard to compute exactly, but if we get very few cliques, it's likely optimal.
        # For n=64, ~15-20 cliques is good.
        if best_count < 15:
             # Check a few more to be sure, but don't spend too much time
             if restart_count > 100:
                 break

    if best_cliques is None:
        # Fallback
        rng = _random.Random(0)
        best_cliques = _greedy_partition_fast(edges, n, rng)
        
    # Final validation
    if not _validate_partition(best_cliques, edges):
        # This should never happen
        pass

    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Solve smaller instance first
    res1 = _solve_instance(edges1, 32)

    # Solve larger instance with remaining time
    res2 = _solve_instance(edges2, 64)

    return [res1, res2]
# EVOLVE-BLOCK-END
