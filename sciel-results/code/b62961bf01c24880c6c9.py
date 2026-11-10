"""Clique partition of two fixed dense random graphs via optimized bitmask greedy with local improvement."""
# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 112.0  # Wall-clock guard (evaluator kills at 150s, keep 38s margin)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _encode_edge(u, v):
    if u > v:
        u, v = v, u
    return u * 128 + v  # n <= 64, so 128 is safe


def _decode_edge(e):
    u = e >> 7
    v = e & 127
    return u, v


def _greedy_partition_fast(edges, n, rng):
    """
    Fast greedy clique partition using bitmasks.
    Strategy:
    1. Pick an edge with fewest common neighbors (hardest to cover) from a sample.
    2. Extend to a maximal clique using a max-degree heuristic within common neighbors.
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

        # Find a good edge to start from: pick an edge with fewest common neighbors
        # Sample a few edges and pick the one with minimum common neighbors
        if len(remaining) > 20:
            sample = rng.sample(list(remaining), 20)
        else:
            sample = list(remaining)

        best_edge = None
        best_common = float('inf')
        for e in sample:
            u, v = _decode_edge(e)
            common = bin(adj[u] & adj[v]).count('1')
            if common < best_common:
                best_common = common
                best_edge = e

        u, v = _decode_edge(best_edge)
        
        # Build clique starting with u and v
        C = [u, v]
        # Candidates: common neighbors of u and v that have remaining edges
        cand_mask = adj[u] & adj[v]
        cand_mask &= ~(1 << u)
        cand_mask &= ~(1 << v)
        
        while cand_mask:
            # Find candidate with max degree within cand_mask
            best_c = -1
            max_deg_c = -1
            temp_mask = cand_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                c = lsb.bit_length() - 1
                # Count neighbors of c within cand_mask
                d = bin(adj[c] & cand_mask).count('1')
                if d > max_deg_c:
                    max_deg_c = d
                    best_c = c
                temp_mask ^= lsb

            if best_c == -1:
                break

            C.append(best_c)

            # Update candidates: keep only those connected to best_c
            cand_mask &= adj[best_c]
            # Remove vertices already in C
            for w in C:
                cand_mask &= ~(1 << w)

        if len(C) >= 2:
            cliques.append(C)
            # Remove all edges within C from remaining and update adj/deg
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = _encode_edge(a, b)
                    if e in remaining:
                        remaining.discard(e)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        deg[a] -= 1
                        deg[b] -= 1
        else:
            # Fallback: take one edge
            e = next(iter(remaining))
            a, b = _decode_edge(e)
            cliques.append([a, b])
            remaining.discard(e)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            deg[a] -= 1
            deg[b] -= 1

    return cliques


def _local_improve(cliques, edges, n, rng):
    """
    Local improvement: try to merge or reassign edges between cliques to reduce count.
    """
    if len(cliques) < 2:
        return cliques

    # Build edge-to-clique mapping
    edge_to_clique = {}
    for ci, c in enumerate(cliques):
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                e = _encode_edge(c[i], c[j])
                edge_to_clique[e] = ci

    # Build adjacency for quick lookup
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    improved = True
    new_cliques = [list(c) for c in cliques]
    
    while improved and time.time() < _DEADLINE - 10:
        improved = False
        # Try merging pairs of cliques
        for i in range(len(new_cliques)):
            if time.time() > _DEADLINE - 10:
                break
            for j in range(i + 1, len(new_cliques)):
                # Check if union of cliques i and j forms a clique
                # This is expensive, so only do it for small cliques
                if len(new_cliques[i]) + len(new_cliques[j]) > 12:
                    continue
                
                union_set = set(new_cliques[i]) | set(new_cliques[j])
                is_clique = True
                for a in union_set:
                    for b in union_set:
                        if a < b:
                            if not (adj[a] & (1 << b)):
                                is_clique = False
                                break
                    if not is_clique:
                        break
                
                if is_clique:
                    new_cliques[i] = list(union_set)
                    new_cliques.pop(j)
                    improved = True
                    break
            if improved:
                break

    return new_cliques


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


def _solve_instance(edges, n, time_budget):
    """
    Solve a single instance with randomized restarts, keeping the best partition.
    """
    if not edges:
        return []

    best_cliques = None
    best_count = float('inf')
    
    # Run restarts until deadline
    restart_count = 0
    start_time = time.time()
    
    while time.time() < _DEADLINE - 5:
        restart_count += 1
        rng = _random.Random(10000 + restart_count)
        cliques = _greedy_partition_fast(edges, n, rng)
        
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques

        # Early stop if we reach a very good solution
        if best_count < 15:
            if restart_count > 100:
                break

    if best_cliques is None:
        # Fallback
        rng = _random.Random(0)
        best_cliques = _greedy_partition_fast(edges, n, rng)
    
    # Try local improvement
    rng2 = _random.Random(42)
    improved = _local_improve(best_cliques, edges, n, rng2)
    if len(improved) < len(best_cliques):
        best_cliques = improved

    # Final validation
    if not _validate_partition(best_cliques, edges):
        # Regenerate if invalid
        rng3 = _random.Random(999)
        best_cliques = _greedy_partition_fast(edges, n, rng3)
    
    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Solve smaller instance first with 40% of budget
    res1 = _solve_instance(edges1, 32, 45.0)

    # Solve larger instance with remaining time
    res2 = _solve_instance(edges2, 64, 65.0)

    return [res1, res2]
# EVOLVE-BLOCK-END
