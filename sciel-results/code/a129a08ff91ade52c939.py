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
    Improved greedy clique partition using bitmasks with triangle-first strategy.
    Strategy:
    1. First pass: greedily extract triangles (3-cliques) to maximize efficient coverage.
    2. Second pass: extend remaining edges into larger cliques where possible.
    3. Use combined heuristic: prefer edges that can form larger cliques.
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

    # Helper: score an edge for clique potential
    def _edge_score(e, adj, deg):
        u, v = _decode_edge(e)
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        common_count = bin(common).count('1')
        # Score: prefer edges with more common neighbors (can form bigger cliques)
        # But also consider if the edge is "hard" (low degree) to cover
        return common_count * 10 + min(deg[u], deg[v])

    # Pass 1: Extract triangles first
    while remaining and time.time() < _DEADLINE - 15:
        # Find best edge for triangle extraction
        if len(remaining) > 50:
            sample = rng.sample(list(remaining), 50)
        else:
            sample = list(remaining)

        best_edge = None
        best_score = -1
        best_common_count = 0
        for e in sample:
            u, v = _decode_edge(e)
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            common_count = bin(common).count('1')
            if common_count >= 1:  # Can form a triangle
                score = common_count * 10 + min(deg[u], deg[v])
                if score > best_score:
                    best_score = score
                    best_edge = e
                    best_common_count = common_count

        if best_edge is None:
            break  # No more triangles possible

        u, v = _decode_edge(best_edge)
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)

        # Pick the common neighbor that maximizes further clique potential
        best_w = -1
        best_w_score = -1
        temp_mask = common
        while temp_mask:
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            # Score: how many more common neighbors would we get
            extra = (adj[u] & adj[v] & adj[w]) & ~(1 << u) & ~(1 << v) & ~(1 << w)
            w_score = bin(extra).count('1') * 5 + deg[w]
            if w_score > best_w_score:
                best_w_score = w_score
                best_w = w
            temp_mask ^= lsb

        if best_w == -1:
            break

        # Form triangle [u, v, best_w]
        C = [u, v, best_w]
        cliques.append(C)

        # Remove all edges within C
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

    # Pass 2: Handle remaining edges with clique extension
    while remaining:
        if time.time() > _DEADLINE - 2:
            # Fallback: emit remaining edges as 2-cliques
            for e in remaining:
                u, v = _decode_edge(e)
                cliques.append([u, v])
            break

        # Find a good edge to start from
        if len(remaining) > 30:
            sample = rng.sample(list(remaining), 30)
        else:
            sample = list(remaining)

        best_edge = None
        best_score = -1
        for e in sample:
            score = _edge_score(e, adj, deg)
            if score > best_score:
                best_score = score
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
    Enhanced local improvement: merge cliques, reassign edges, and split large cliques
    to reduce total clique count.
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

    def _is_clique(vertices, adj):
        """Check if vertices form a clique."""
        vs = list(vertices)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if not (adj[vs[i]] & (1 << vs[j])):
                    return False
        return True

    new_cliques = [list(c) for c in cliques]
    improved = True
    iterations = 0
    max_iterations = 500

    while improved and time.time() < _DEADLINE - 10 and iterations < max_iterations:
        improved = False
        iterations += 1

        # Strategy 1: Try merging pairs of cliques
        for i in range(len(new_cliques)):
            if time.time() > _DEADLINE - 10:
                break
            if len(new_cliques[i]) < 2:
                continue
            for j in range(i + 1, len(new_cliques)):
                if len(new_cliques[j]) < 2:
                    continue
                # Only try merging if combined size is reasonable
                if len(new_cliques[i]) + len(new_cliques[j]) > 15:
                    continue
                
                union_set = set(new_cliques[i]) | set(new_cliques[j])
                if _is_clique(union_set, adj):
                    new_cliques[i] = list(union_set)
                    new_cliques.pop(j)
                    improved = True
                    break
            if improved:
                break

        if improved:
            continue

        # Strategy 2: Try to move a vertex from one clique to another (if it maintains clique property)
        for i in range(len(new_cliques)):
            if time.time() > _DEADLINE - 10:
                break
            if len(new_cliques[i]) < 3:
                continue
            for j in range(len(new_cliques)):
                if j == i:
                    continue
                if len(new_cliques[j]) < 2:
                    continue
                # Try moving each vertex from clique i to clique j
                for k, v in enumerate(new_cliques[i]):
                    # Check if v can be added to clique j
                    test_j = list(new_cliques[j]) + [v]
                    if _is_clique(test_j, adj):
                        # Check if removing v from clique i still leaves a valid clique
                        remaining_i = [x for x in new_cliques[i] if x != v]
                        if len(remaining_i) >= 2 and _is_clique(remaining_i, adj):
                            # Do the move
                            new_cliques[j] = test_j
                            new_cliques[i] = remaining_i
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    # Remove any empty cliques
    new_cliques = [c for c in new_cliques if len(c) >= 2]

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
    Uses multiple strategies: triangle-first greedy, then local improvement.
    """
    if not edges:
        return []

    best_cliques = None
    best_count = float('inf')
    
    # Run restarts until deadline
    restart_count = 0
    start_time = time.time()
    
    while time.time() < _DEADLINE - 8:
        restart_count += 1
        rng = _random.Random(10000 + restart_count)
        cliques = _greedy_partition_fast(edges, n, rng)
        
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques

        # Try local improvement on this solution
        if time.time() < _DEADLINE - 5:
            rng2 = _random.Random(20000 + restart_count)
            improved = _local_improve(cliques, edges, n, rng2)
            if len(improved) < best_count:
                best_count = len(improved)
                best_cliques = improved

        # Early stop if we reach a very good solution
        if best_count < 12:
            if restart_count > 50:
                break

    if best_cliques is None:
        # Fallback
        rng = _random.Random(0)
        best_cliques = _greedy_partition_fast(edges, n, rng)
    
    # Try local improvement on the best solution
    rng3 = _random.Random(42)
    improved = _local_improve(best_cliques, edges, n, rng3)
    if len(improved) < len(best_cliques):
        best_cliques = improved

    # Final validation
    if not _validate_partition(best_cliques, edges):
        # Regenerate if invalid
        rng4 = _random.Random(999)
        best_cliques = _greedy_partition_fast(edges, n, rng4)
    
    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Solve smaller instance first with 35% of budget
    res1 = _solve_instance(edges1, 32, 40.0)

    # Solve larger instance with remaining time
    res2 = _solve_instance(edges2, 64, 70.0)

    return [res1, res2]
# EVOLVE-BLOCK-END
