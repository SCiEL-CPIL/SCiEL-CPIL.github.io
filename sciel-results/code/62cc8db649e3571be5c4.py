"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Strategy:
    1. Try fast Greedy Random Construction with incremental forbidden-set updates.
    2. Fallback to LZW Algebraic Construction.
    """
    deadline = time.time() + 110  # Safety margin
    
    # Try greedy construction for smaller n first
    # The Moore bound for k=5, g=9 is 426.
    # Best known is 1116.
    # We try to find something smaller than the LZW fallback (1562).
    target_ns = [1100, 1150, 1200, 1300, 1400, 1500]
    
    for n in target_ns:
        if time.time() > deadline:
            break
        result = _greedy_construct(k, g, n, deadline)
        if result is not None:
            return result
            
    # Fallback to LZW
    return _lzw_construct(k, g)

def _greedy_construct(k, g, n, deadline):
    """
    Fast greedy construction of a k-regular graph with girth >= g.
    Maintains forbidden[u] = set of vertices within distance < g-1 from u.
    """
    max_restarts = 3
    for attempt in range(max_restarts):
        if time.time() > deadline:
            return None
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        # forbidden[u] contains all vertices v such that dist(u,v) < g-1 in current graph
        # Initially, each vertex only forbids itself
        forbidden = [set([i]) for i in range(n)]
        
        edges_added = 0
        target_edges = n * k // 2
        
        # Track vertices that still need edges
        # Use a list of (vertex, deficit) for efficient selection
        pending = list(range(n))
        random.shuffle(pending)
        
        while edges_added < target_edges:
            if time.time() > deadline:
                return None
            
            # Pick a vertex u that needs an edge
            # Filter pending to those with deg < k
            candidates_u = [u for u in pending if deg[u] < k]
            if not candidates_u:
                break
            
            # Pick u with fewest available options (most constrained)
            # Available options = n - 1 - len(forbidden[u]) - (k - deg[u] - 1) ... 
            # Actually: valid v are those not in forbidden[u], deg[v] < k, v != u
            # Estimate: n - len(forbidden[u]) - deg[u] (rough)
            # For speed, just pick random or pick the one with smallest available count
            
            # For efficiency, pick a random candidate but weight toward constrained ones
            # Simple: random pick
            u = random.choice(candidates_u)
            
            # Find valid v: v not in forbidden[u], deg[v] < k, v != u
            # forbidden[u] already includes u and all vertices within dist < g-1
            # So we just need: v not in forbidden[u] and deg[v] < k
            
            # Build list of valid candidates
            valid_vs = []
            for v in range(n):
                if v in forbidden[u]:
                    continue
                if deg[v] < k:
                    valid_vs.append(v)
            
            if not valid_vs:
                # Stuck
                return None
            
            # Pick v - prefer vertices with more remaining capacity to reduce future constraints
            # Weighted random: weight = (k - deg[v])
            weights = [k - deg[v] for v in valid_vs]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            v = valid_vs[0]
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    v = valid_vs[i]
                    break
            
            # Add edge (u, v)
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
            edges_added += 1
            
            # Update forbidden sets
            # When adding edge (u,v), the neighborhoods expand.
            # For any x, if dist(x,u) < g-2, then v becomes within dist < g-1 of x
            # Similarly for y, if dist(y,v) < g-2, then u becomes within dist < g-1 of y
            
            # More precisely: after adding edge (u,v), for any vertex x:
            #   if dist(x, u) <= g-3, then dist(x, v) <= dist(x,u)+1 <= g-2 < g-1, so add v to forbidden[x]
            #   if dist(x, v) <= g-3, then dist(x, u) <= dist(x,v)+1 <= g-2 < g-1, so add u to forbidden[x]
            # Also, for any pair (x,y) where x was within dist < g-1 of u and y was within dist < g-1 of v,
            # the new edge may create shorter paths, but this is complex. 
            # 
            # Simpler correct approach: forbidden[x] should contain all y with dist(x,y) < g-1.
            # After adding edge (u,v), we need to update. The key new connections are:
            # - v is now adjacent to u, so for any x with dist(x,u) <= g-3, dist(x,v) <= g-2, so v in forbidden[x]
            # - u is now adjacent to v, so for any x with dist(x,v) <= g-3, dist(x,u) <= g-2, so u in forbidden[x]
            # - Also, for any x with dist(x,u) <= g-2 and y with dist(y,v) <= g-2, 
            #   dist(x,y) <= dist(x,u) + 1 + dist(v,y) <= (g-2)+1+(g-2) = 2g-3 which may be >= g-1, so no issue
            # - But for x with dist(x,u) <= g-3 and y with dist(y,v) <= g-3: dist(x,y) <= (g-3)+1+(g-3) = 2g-5
            #   For g=9: 2*9-5=13 >= 8=g-1, so no issue
            # - For x with dist(x,u) <= g-4 and y with dist(y,v) <= g-4: dist(x,y) <= (g-4)+1+(g-4) = 2g-7
            #   For g=9: 2*9-7=11 >= 8, no issue
            # 
            # The critical updates are:
            # 1. For each x with dist(x,u) <= g-3: add v to forbidden[x]
            # 2. For each x with dist(x,v) <= g-3: add u to forbidden[x]
            # 3. Additionally, for pairs (x,y) where x in N_{g-2}(u) and y in N_{g-2}(v):
            #    dist(x,y) <= dist(x,u)+1+dist(v,y) <= (g-2)+1+(g-2) = 2g-3
            #    We need 2g-3 >= g-1 => g >= 2, always true. So no new forbidden pairs from this.
            #    Wait, we need dist(x,y) < g-1 to add to forbidden. 2g-3 < g-1 => g < 2, false.
            #    So no additional cross-forbidden updates needed beyond the direct ones.
            #
            # But wait: we also need to update forbidden for u and v themselves, and for neighbors.
            # Let me think more carefully. The forbidden set for vertex x is {y : dist(x,y) < g-1}.
            # Before adding edge, this was computed. After adding edge (u,v):
            # - For x = u: forbidden[u] gains all y with dist(y,v) <= g-3 (since dist(u,y) <= 1 + dist(v,y) <= g-2 < g-1)
            #   But also y=v is now at dist 1 < g-1, so v in forbidden[u] (already handled by rule 2 for x=u, dist(u,v)=1<=g-3 for g>=4)
            # - Similarly for x = v.
            #
            # Let me just do a BFS-based update for efficiency. Since g is small (9), the neighborhood 
            # of radius g-3=6 is small.
            
            # Find all vertices within distance <= g-3 from u
            # BFS from u up to depth g-3
            depth = g - 3  # For g=9, depth=6
            if depth >= 0:
                # BFS from u
                seen_u = {u}
                frontier = [u]
                for d in range(depth):
                    next_frontier = []
                    for node in frontier:
                        for nb in adj[node]:
                            if nb not in seen_u:
                                seen_u.add(nb)
                                next_frontier.append(nb)
                    frontier = next_frontier
                
                # For all x in seen_u, add v to forbidden[x]
                for x in seen_u:
                    forbidden[x].add(v)
            
            # BFS from v
            if depth >= 0:
                seen_v = {v}
                frontier = [v]
                for d in range(depth):
                    next_frontier = []
                    for node in frontier:
                        for nb in adj[node]:
                            if nb not in seen_v:
                                seen_v.add(nb)
                                next_frontier.append(nb)
                    frontier = next_frontier
                
                # For all x in seen_v, add u to forbidden[x]
                for x in seen_v:
                    forbidden[x].add(u)
            
            # Also need to handle the fact that adding edge (u,v) may create new short paths
            # between vertices that were previously far apart. Specifically, for any x with dist(x,u) <= g-3
            # and any y with dist(y,v) <= g-3, dist(x,y) <= dist(x,u)+1+dist(v,y) <= (g-3)+1+(g-3) = 2g-5.
            # For g=9, 2g-5=13 >= 8=g-1, so no issue.
            # For x with dist(x,u) <= g-4 and y with dist(y,v) <= g-4: dist(x,y) <= 2g-7. For g=9, 11>=8.
            # So the only updates needed are the direct ones above.
            
            # But we also need to ensure that forbidden[u] and forbidden[v] are correct.
            # After adding edge, u and v are adjacent, so v in forbidden[u] and u in forbidden[v].
            # This is handled by the BFS above since dist(u,v)=1 <= g-3 for g>=4.
            
        # Check if all degrees are k
        if all(d == k for d in deg):
            # Final verification
            if _verify_girth(n, edges, g):
                return (n, edges)
                
    return None

def _verify_girth(n, edges, g):
    """
    Verify the graph has girth >= g using non-backtracking walks.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # For each vertex, do BFS tracking parent to avoid backtracking
    # Find shortest cycle
    
    min_girth = float('inf')
    for start in range(n):
        # BFS with parent tracking
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for node in queue:
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    # Found a back edge
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
        
        if min_girth < g:
            return False
            
    return True

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
