"""Hexagon-free graph construction on 40 vertices using Spectral Heuristic."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 40
DEADLINE = 135.0  # Safety margin before 150s kill

def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_c6_adj(adj, n):
    """Check if graph with adjacency sets has a 6-cycle."""
    # Optimized check: for each edge (u,v), check if there is a path of length 4 from v to u
    # avoiding u, v
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # Find all vertices at distance 2 from v (excluding u)
            # dist2 = { w | exists x: v-x-w, x!=u, w!=u, w!=v }
            # Actually, we need a path v-w-x-y-z-u.
            # This is equivalent to: (A^4)[v][u] > 0 in the graph with edge (u,v) removed?
            # No, C6 is a simple cycle.
            # Let's stick to the explicit search but optimized with sets.
            
            # Path: u - v - w - x - y - z - u
            # w is neighbor of v, w != u
            # x is neighbor of w, x != u, x != v
            # y is neighbor of x, y != u, y != v, y != w
            # z is neighbor of y, z != u, z != v, z != w, z != x
            # Check if z is neighbor of u
            
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
    The new cycle must use the new edge.
    Cycle: added_u - added_v - w - x - y - z - added_u
    So we need a path of length 4 from added_v to added_u in the graph WITHOUT the new edge.
    Path: added_v - w - x - y - z - added_u
    w in adj[added_v], w != added_u
    x in adj[w], x != added_u, x != added_v
    y in adj[x], y != added_u, y != added_v, y != w
    z in adj[y], z != added_u, z != added_v, z != w, z != x
    Check if added_u in adj[z]
    """
    for w in adj[added_v]:
        if w == added_u:
            continue
        for x in adj[w]:
            if x == added_u or x == added_v:
                continue
            for y in adj[x]:
                if y == added_u or y == added_v or y == w:
                    continue
                for z in adj[y]:
                    if z == added_u or z == added_v or z == w or z == x:
                        continue
                    if added_u in adj[z]:
                        return True
    return False

def _spectral_score(adj, n):
    """
    Heuristic score to guide the search.
    We want to maximize edges while avoiding C6.
    C6 existence is related to the number of 4-paths.
    Let P4 be the number of paths of length 4.
    Minimizing P4 for a given number of edges helps avoid C6?
    Actually, ex(n, C6) graphs tend to be quasi-random or structured.
    Let's use a simpler proxy: 
    Score = num_edges - lambda * (number of C4s + number of C5s) ?
    Or better: Just use a random walk based heuristic?
    
    Alternative: 
    Use the fact that if the graph is C6-free, it cannot be too dense.
    Let's just use a "local density" penalty.
    
    For the Simulated Annealing, we need a cost function.
    Let's define Cost = -num_edges + penalty_for_c6_risk.
    C6 risk can be estimated by the number of pairs of vertices at distance 3 that are non-adjacent?
    If u, v are at distance 3 (path u-a-b-v) and not adjacent, adding edge (u,v) creates a C4? No.
    If u, v are at distance 5 (path u-a-b-c-d-v) and not adjacent, adding edge (u,v) creates a C6.
    
    So, a good heuristic is to count the number of pairs of non-adjacent vertices that are at distance 5.
    If this count is 0, the graph is "saturated" with respect to C6 addition?
    Actually, if there is a path of length 5 between u and v, and u-v is not an edge, adding it creates a C6.
    So, to be C6-free, there must be NO path of length 5 between any non-adjacent pair.
    
    Let's calculate the number of "forbidden" pairs: pairs (u,v) not in E such that dist(u,v) = 5.
    If this count is > 0, we are "safe" to add more edges? No, if we add any of those edges, we create a C6.
    So, if the graph is C6-free, the number of such pairs MUST be 0?
    Wait. If there is a path of length 5 between u and v, and u-v is NOT an edge, the graph is currently C6-free.
    Adding the edge (u,v) WOULD create a C6.
    So, a C6-free graph can have many such pairs. It just means we *can't* add those edges.
    
    To maximize edges, we want to minimize the number of "blocked" edges?
    Or rather, we want to find a graph where as many edges as possible can be added without creating a C6.
    
    Let's try a different spectral approach:
    Random Regular Graphs are often C6-free if degree is low enough.
    Start with a random 8-regular graph.
    Use Simulated Annealing to swap edges to increase density or change structure.
    
    Cost Function for SA:
    We want to maximize edges.
    Let's try to start with a sparse graph and add edges.
    But SA works better on fixed-size or swap-based.
    
    Let's stick to the "Greedy + Swap" but improve the heuristic for choosing swaps.
    Current program uses random swaps.
    
    New Strategy: 
    1. Start with 8 K5s (100 edges).
    2. Identify "bottlenecks": edges whose removal allows many new edges to be added.
    3. Use a "Lookahead" greedy: 
       For each non-edge (u,v), check if it can be added.
       If not, it's because of a path of length 4 between u and v.
       Let P be the set of such paths.
       If we remove an edge e from P, can we add (u,v)?
       If yes, and if the net gain is positive (e.g., we can add 2 new edges for every 1 removed), do it.
    """
    pass

def _construct_spectral():
    start_time = time.time()
    n = N
    
    # Initialize with 8 K5s
    edges = set()
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.add((base + a, base + b))
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    current_count = len(edges)
    best_edges = set(edges)
    best_count = current_count
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Phase 1: Aggressive Greedy with Backtracking/Lookahead
    # We will try to add edges. If blocked, we look for a "cheap" edge to remove.
    
    # Precompute "blockers" for each non-edge? Too expensive to update dynamically.
    # Instead, when we fail to add (u,v), we find one path of length 4.
    # We pick a random edge from that path to remove, add (u,v), and then try to re-add the removed edge or others.
    
    def get_c6_path(u, v):
        """Find one path of length 4 between u and v in current graph."""
        for w in adj[v]:
            if w == u: continue
            for x in adj[w]:
                if x == u or x == v: continue
                for y in adj[x]:
                    if y == u or y == v or y == w: continue
                    for z in adj[y]:
                        if z == u or z == v or z == w or z == x: continue
                        if u in adj[z]:
                            return [u, v, w, x, y, z]
        return None

    temperature = 10.0
    cooling = 0.99
    
    while time.time() - start_time < DEADLINE - 5:
        # Try to add a random non-edge
        random.shuffle(all_pairs)
        progress = False
        for u, v in all_pairs:
            if (u, v) in edges:
                continue
            if not _incremental_c6_check(adj, n, u, v):
                # Can add
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                current_count += 1
                progress = True
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(edges)
                break
        
        if not progress:
            # Stuck. Try a swap.
            # Pick a random edge to remove
            if not edges:
                break
            rem_u, rem_v = random.choice(list(edges))
            
            # Remove it
            edges.discard((rem_u, rem_v))
            adj[rem_u].discard(rem_v)
            adj[rem_v].discard(rem_u)
            current_count -= 1
            
            # Try to add several edges now
            added_any = False
            # Try to re-add the removed edge? No, that's a revert.
            # Try to add other edges
            random.shuffle(all_pairs)
            for a, b in all_pairs:
                if (a, b) in edges:
                    continue
                if not _incremental_c6_check(adj, n, a, b):
                    edges.add((a, b))
                    adj[a].add(b)
                    adj[b].add(a)
                    current_count += 1
                    added_any = True
                    break
            
            # If we didn't add anything, revert
            if not added_any:
                edges.add((rem_u, rem_v))
                adj[rem_u].add(rem_v)
                adj[rem_v].add(rem_u)
                current_count += 1
            else:
                # If we added one, count is same. 
                # Let's try to add MORE edges to make it worth it.
                # Loop a few times to try to gain net +1 or +2
                for _ in range(5):
                    random.shuffle(all_pairs)
                    for a, b in all_pairs:
                        if (a, b) in edges:
                            continue
                        if not _incremental_c6_check(adj, n, a, b):
                            edges.add((a, b))
                            adj[a].add(b)
                            adj[b].add(a)
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                                best_edges = set(edges)
                            break

    return list(best_edges), best_count

def construct():
    start_time = time.time()
    
    # Run the spectral/greedy heuristic
    edges, count = _construct_spectral()
    
    # Verify
    adj = _build_adj(edges)
    if _has_c6_adj(adj, N):
        # Fallback to seed
        edges = []
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([base + a, base + b])
        return edges
    
    return edges
# EVOLVE-BLOCK-END
