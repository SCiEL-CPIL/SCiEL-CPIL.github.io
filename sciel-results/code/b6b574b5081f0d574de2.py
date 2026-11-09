"""Hexagon-free graph construction on 40 vertices – block-structured approach."""
import time
import random
import numpy as np

N = 40
G = 8       # number of groups
S = 5       # size of each group
TOTAL = G * S

def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_c6_adj(adj, n):
    """Brute-force C6 check using adjacency sets."""
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                return True
    return False

def _incremental_c6_check(adj, n, added_u, added_v):
    """
    Check if adding edge (added_u, added_v) creates a C6.
    A C6 through this new edge: added_u - added_v - w - x - y - z - added_u
    i.e., a path of length 4 from added_v to added_u in the current graph.
    """
    for w in adj[added_v]:
        if w == added_u:
            continue
        for x in adj[w]:
            if x == added_u or x == added_v or x == w:
                continue
            for y in adj[x]:
                if y == added_u or y == added_v or y == w or y == x:
                    continue
                for z in adj[y]:
                    if z == added_u or z == added_v or z == w or z == x or z == y:
                        continue
                    if added_u in adj[z]:
                        return True
    return False

def _group_of(v):
    return v // S

def _local_intra_edges():
    """Complete graph within each group of S=5 vertices."""
    edges = []
    for g in range(G):
        base = g * S
        for a in range(S):
            for b in range(a + 1, S):
                edges.append([base + a, base + b])
    return edges

def _random_inter_edges(rng, density):
    """Random inter-group edges with given density, ensuring no C6 via check."""
    edges = []
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            for a in range(S):
                for b in range(S):
                    if rng.random() < density:
                        edges.append([g1 * S + a, g2 * S + b])
    return edges

def _build_and_check(intra_edges, inter_edges):
    """Build full edge list and verify C6-free."""
    all_edges = intra_edges + inter_edges
    adj = _build_adj(all_edges)
    if _has_c6_adj(adj, N):
        return None, adj
    return all_edges, adj

def _optimize_inter_edges(intra_edges, rng, time_limit, start_time):
    """
    Optimize inter-group edges via greedy add + 1-swap local search.
    Returns best (inter_edges, count) found.
    """
    # Start with a moderate-density random inter-edge set
    best_inter = []
    best_count = 0
    best_adj = None

    all_inter_pairs = []
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            for a in range(S):
                for b in range(S):
                    all_inter_pairs.append([g1 * S + a, g2 * S + b])

    # Try several initial densities
    for density in [0.2, 0.3, 0.4, 0.5, 0.6]:
        if time.time() - start_time > time_limit - 5:
            break
        rng.seed(int(time.time() * 1000) % 10000 + int(density * 100))
        inter_edges = _random_inter_edges(rng, density)
        all_edges, adj = _build_and_check(intra_edges, inter_edges)
        if all_edges is None:
            continue
        count = len(all_edges)
        if count > best_count:
            best_count = count
            best_inter = list(inter_edges)
            best_adj = adj

    if best_adj is None:
        # Fallback: no inter edges
        return [], len(intra_edges)

    # Local search: try adding/removing/swapping inter edges
    edge_set = set(tuple(e) for e in best_inter)
    current_inter = list(best_inter)
    current_count = best_count

    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        return best_inter, best_count

    # Precompute all possible inter pairs
    all_inter_set = set(tuple(e) for e in all_inter_pairs)

    no_improve = 0
    max_no_improve = 500

    while time.time() - start_time < time_limit - 5:
        if no_improve > max_no_improve:
            break

        # Try to add a random inter edge
        added = False
        for _ in range(50):
            u, v = rng.choice(all_inter_pairs)
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if _incremental_c6_check(best_adj, N, u, v):
                continue
            # Add it
            best_adj[u].add(v)
            best_adj[v].add(u)
            edge_set.add((u, v))
            current_inter.append([u, v])
            current_count += 1
            added = True
            if current_count > best_count:
                best_count = current_count
                best_inter = list(current_inter)
            no_improve = 0
            break

        if not added:
            # Try a swap: remove one inter edge, add another
            if not current_inter:
                break
            idx = rng.randint(0, len(current_inter) - 1)
            u, v = current_inter[idx]
            # Remove
            best_adj[u].discard(v)
            best_adj[v].discard(u)
            edge_set.discard((u, v))
            edge_set.discard((v, u))
            current_inter.pop(idx)
            current_count -= 1

            # Try to add a different inter edge
            swap_added = False
            for _ in range(50):
                a, b = rng.choice(all_inter_pairs)
                if (a, b) in edge_set or (b, a) in edge_set:
                    continue
                if _incremental_c6_check(best_adj, N, a, b):
                    continue
                # Add
                best_adj[a].add(b)
                best_adj[b].add(a)
                edge_set.add((a, b))
                current_inter.append([a, b])
                current_count += 1
                swap_added = True
                if current_count > best_count:
                    best_count = current_count
                    best_inter = list(current_inter)
                no_improve = 0
                break

            if not swap_added:
                # Revert
                best_adj[u].add(v)
                best_adj[v].add(u)
                edge_set.add((u, v))
                current_inter.append([u, v])
                current_count += 1
                no_improve += 1

    return best_inter, best_count

def construct():
    n = N
    deadline = 135  # Safety margin before 150s kill
    start_time = time.time()

    rng = random.Random(42)

    # Intra-group edges: 8 * C(5,2) = 8 * 10 = 80 edges
    intra_edges = _local_intra_edges()

    # Optimize inter-group edges
    inter_edges, count = _optimize_inter_edges(
        intra_edges, rng, time_limit=deadline - 10, start_time=start_time
    )

    all_edges = intra_edges + inter_edges

    # Final verification
    adj = _build_adj(all_edges)
    if _has_c6_adj(adj, n):
        # Fallback to 8-K5 seed
        fallback = _local_intra_edges()
        return fallback

    return all_edges
