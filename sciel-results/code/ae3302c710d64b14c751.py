"""Hexagon-free graph construction on 40 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

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

def _has_c6_matrix(adj_mat, n):
    """Check for C6 using numpy matrix operations."""
    # A 6-cycle exists iff there's a walk of length 6 from u back to u
    # with all intermediate vertices distinct and u not appearing internally
    # For a proper C6: u-v-w-x-y-z-u where all 6 vertices are distinct
    
    # Use adjacency matrix A
    # Count 6-cycles: trace(A^6) counts closed walks of length 6
    # A 6-cycle contributes 12 to trace(A^6) (2 directions * 6 starting points)
    # But we also need to exclude shorter cycles that close after 6 steps
    # Actually for simple graphs with no triangles, 4-cycles, 5-cycles, 
    # trace(A^6)/12 gives the number of 6-cycles
    # But if there ARE 4-cycles or other structures, this gets complicated.
    
    # Safer: use the set-based check for verification
    # But for speed in search, use a heuristic incremental check
    
    return False  # Fallback to set-based

def _incremental_c6_check(adj, n, added_u, added_v):
    """
    Check if adding edge (added_u, added_v) creates a C6.
    A C6 through this new edge: added_u - added_v - w - x - y - z - added_u
    where w-x-y-z is a path of length 4 from added_v to added_u
    with all vertices distinct and not equal to added_u, added_v.
    """
    # Find all paths of length 4 from added_v to added_u
    # added_v -> w -> x -> y -> z -> added_u
    # w != added_u, added_v; x != added_u, added_v, w; y != added_u, added_v, w, x; z != added_u, added_v, w, x, y
    
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

def _seed_graph_blowup():
    """
    Start with a C6-free base graph and blow up vertices.
    Use the complement of a matching as base? No, that has C6.
    Use C5 blowup? C5 has 5 vertices, no C6. Blow up each vertex to a set.
    If we blow up C5 with parts of sizes s1..s5, total vertices = sum(si).
    Edges between part i and part j if i and j adjacent in C5.
    This is a complete 5-partite graph with parts connected in a cycle.
    No C6 because any cycle in a C5-blowup projects to a closed walk in C5,
    and C5 has no closed walk of length 6 that visits 6 distinct projected vertices
    (since there are only 5 parts, by pigeonhole some part repeats, creating a chord or shorter cycle).
    
    Actually wait: a 6-cycle in the blowup projects to a closed walk of length 6 in C5.
    C5 has closed walks of length 6: e.g., 1-2-3-4-5-1-2... no that's not a closed walk of length 6 ending at start.
    Closed walks of length 6 in C5: the number is trace(A^6) where A is C5 adjacency.
    C5 eigenvalues: 2cos(2πk/5) for k=0..4 = 2, 0.618, 0.618, -1.618, -1.618
    trace(A^6) = 2^6 + 2*(0.618)^6 + 2*(-1.618)^6 ≈ 64 + 2*0.0557 + 2*17.944 ≈ 64 + 0.111 + 35.888 ≈ 99.999 ≈ 100
    So there are 100/12 ≈ 8.33... that's not integer, so I'm computing wrong.
    
    Actually for C5: A^6 diagonal entries. Let me just check: does C5-blowup have C6?
    A 6-cycle in the blowup: v1-v2-v3-v4-v5-v6-v1. Each vi is in some part Pi.
    P1-P2-P3-P4-P5-P6-P1 is a closed walk in C5 of length 6.
    In C5, a closed walk of length 6 must visit a vertex at least twice (only 5 vertices).
    If Pi = Pj for i≠j with |i-j| not ≡ 0 mod 5, then in the blowup, vertices in the same part are NOT connected (it's a complete multipartite where parts are independent sets).
    Wait, in a blowup of C5, vertices in the same part form an independent set. So if P1 = P3, then v1 and v3 are in the same independent set, so they're NOT adjacent. But in the 6-cycle, v1 and v3 are not adjacent (they're distance 2 apart). So that's fine.
    
    But if P1 = P4, then v1 and v4 are in the same part, not adjacent. In the 6-cycle, v1 and v4 are distance 3 apart, not adjacent. Fine.
    
    The issue is: can we have a 6-cycle where all 6 projected parts form a valid closed walk in C5?
    Closed walks of length 6 in C5: e.g., 1-2-1-2-3-4-? no, must return to 1.
    1-2-1-2-1-2-? no.
    Let me enumerate: starting at 1, after 6 steps return to 1.
    Step sequences (±1 mod 5): need sum ≡ 0 mod 5, 6 steps.
    Possible: three +1 and three -1: sum=0. E.g., 1,2,3,2,1,0,1? That's 6 steps: 1→2→3→2→1→0→1. Parts: 1,2,3,2,1,0,1. P1=P5=1, P2=P6=2. In the blowup, v1 and v5 are both in part 1, independent, not adjacent - but in the 6-cycle they're not adjacent. v2 and v6 both in part 2, not adjacent in cycle. So this is a valid 6-cycle in the blowup!
    
    So C5-blowup CAN have C6. Bad seed.
    
    Let me try a different approach: use the 8-K5 seed and do structured local search.
    """
    return _seed_graph_8K5()

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
