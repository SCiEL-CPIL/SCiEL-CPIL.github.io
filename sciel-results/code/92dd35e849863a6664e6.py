"""Hexagon-free graph construction on 40 vertices."""
# EVOLVE-BLOCK-START
import time
import random

N = 40


def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _has_c6_adj(adj, n):
    """Check if graph with adjacency sets has a 6-cycle."""
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
    where w-x-y-z is a path of length 4 from added_v to added_u
    with all vertices distinct and not equal to added_u, added_v.
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


def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges


def _greedy_with_swap(n, seed, time_limit):
    """Greedy edge addition with periodic swaps to escape local optima."""
    random.seed(seed)
    adj = [set() for _ in range(n)]
    edge_set = set()
    start_time = time.time()
    
    # Start from 8-K5
    edges = _seed_graph_8K5()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
    
    current_count = len(edge_set)
    best_count = current_count
    best_edges = list(edge_set)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Phase 1: Greedy add
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if time.time() - start_time > time_limit * 0.4:
            break
        if (u, v) in edge_set:
            continue
        if _incremental_c6_check(adj, n, u, v):
            continue
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        current_count += 1
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
    
    # Phase 2: Local search with swaps
    remaining = time_limit - (time.time() - start_time)
    if remaining <= 0:
        return [[u, v] for u, v in best_edges], best_count
    
    # Build list of current edges for random access
    edge_list = list(edge_set)
    
    while time.time() - start_time < time_limit:
        # Try to add a random non-edge
        progress = False
        for _ in range(30):
            u, v = random.choice(all_pairs)
            if (u, v) in edge_set:
                continue
            if not _incremental_c6_check(adj, n, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                edge_list.append((u, v))
                current_count += 1
                progress = True
                if current_count > best_count:
                    best_count = current_count
                    best_edges = list(edge_set)
                break
        
        if not progress and edge_list:
            # Try a swap: remove one edge, add another
            idx = random.randint(0, len(edge_list) - 1)
            u, v = edge_list[idx]
            adj[u].discard(v)
            adj[v].discard(u)
            edge_set.discard((u, v))
            edge_list.pop(idx)
            current_count -= 1
            
            added = False
            for _ in range(30):
                a, b = random.choice(all_pairs)
                if a == b or (a, b) in edge_set:
                    continue
                if not _incremental_c6_check(adj, n, a, b):
                    adj[a].add(b)
                    adj[b].add(a)
                    edge_set.add((a, b))
                    edge_list.append((a, b))
                    current_count += 1
                    added = True
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = list(edge_set)
                    break
            
            if not added:
                # Revert
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                edge_list.append((u, v))
                current_count += 1
    
    return [[u, v] for u, v in best_edges], best_count


def _multi_start_local_search(n, time_limit):
    """Run multiple independent local searches from different random starts."""
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    # Several different seeds and strategies
    seeds = [1, 7, 13, 42, 99, 123, 256, 512, 1000, 2024]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - 5:
            break
        per_time = min(8, max(2, time_limit - 5 - (time.time() - start_time)))
        edges, count = _greedy_with_swap(n, seed, per_time)
        if count > best_count:
            best_count = count
            best_edges = edges
    
    return best_edges, best_count


def construct():
    n = N
    deadline = 135  # Safety margin before 150s kill
    
    start_time = time.time()
    
    # Multi-start local search
    edges, count = _multi_start_local_search(n, time_limit=deadline - 10)
    
    # Final verification: ensure no C6
    adj = _build_adj(edges)
    if _has_c6_adj(adj, n):
        # Fallback to seed
        return _seed_graph_8K5()
    
    return edges
# EVOLVE-BLOCK-END
