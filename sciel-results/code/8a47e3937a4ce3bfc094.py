import math
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g, minimizing n.
    Uses a multi-strategy approach:
    1. Try known explicit constructions (Lazebnik-Ustimenko-Woldar, Moore bound-based)
    2. Try random graph + local optimization
    3. Return best valid graph found within time limit
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_result = None
    
    # Strategy 1: Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    if q >= 3:
        d_min = g - 5
        if d_min % 2 == 0:
            d_min += 1
        d_min = max(3, d_min)
        
        for d in range(d_min, min(d_min + 10, 20)):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = _luw_graph(q, d)
                if n > 0 and _verify_graph(k, g, n, edges, time_limit=5):
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
            except Exception:
                pass
    
    # Strategy 2: Random graph with local search optimization
    if time.time() < deadline - 5:
        try:
            n_random, edges_random = _random_regular_with_girth_optimized(k, g, time_limit=deadline - time.time() - 5)
            if n_random > 0 and _verify_graph(k, g, n_random, edges_random, time_limit=5):
                if best_result is None or n_random < best_result[0]:
                    best_result = (n_random, edges_random)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = _luw_graph(k, d)
        if n > 0:
            best_result = (n, edges)
        else:
            if k == 1:
                best_result = (2, [(0, 1)])
            elif k == 2:
                best_result = (g, [(i, (i+1) % g) for i in range(g)])
            else:
                best_result = (0, [])
    
    return best_result

def _luw_graph(q, d):
    # ... (keep as is) ...
    pass # Placeholder for SEARCH/REPLACE

def _random_regular_with_girth_optimized(k, g, time_limit=10):
    """
    Construct k-regular graph with girth >= g using random init + local search.
    """
    start_time = time.time()
    
    # Moore bound
    if g % 2 == 0:
        h = g // 2 - 1
        moore = 2 * k * (k - 1) ** h if h >= 0 else 2 * k
    else:
        h = (g - 1) // 2
        moore = 2 * (k - 1) ** h + k * (k - 1) ** (h - 1) if h >= 1 else 2 * k
    
    moore = max(moore, 2 * k)
    
    # Try increasing n
    # For k=5, g=9, moore=426.
    # We try a few values.
    for n in range(moore, moore + 100):
        if time.time() - start_time > time_limit:
            break
        
        # Ensure n*k is even
        if (n * k) % 2 != 0:
            continue
            
        # Try a few random initializations
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
                
            try:
                adj = _random_k_regular_adj(n, k)
                if adj is None:
                    continue
                
                # Local search to reduce short cycles
                adj = _local_search_girth(adj, g, time_limit=time_limit - (time.time() - start_time))
                
                # Check if girth is sufficient
                if _has_girth_at_least_fast(adj, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
            except Exception:
                pass
                
    return 0, []

def _random_k_regular_adj(n, k):
    """
    Generate random k-regular graph adjacency list.
    """
    if (n * k) % 2 != 0:
        return None
        
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
        
    return adj

def _count_short_cycles(adj, g):
    """
    Count cycles of length < g.
    Efficient for small g.
    """
    n = len(adj)
    count = 0
    
    # Count 4-cycles
    # For each edge (u,v), count common neighbors
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj[u].intersection(adj[v])
                count += len(common) * (len(common) - 1) // 2
                
    # If we already have 4-cycles, girth is 4 < g (if g > 4)
    # We can just return a large number or specific count
    # For optimization, we want to minimize this count.
    
    # Count 6-cycles? Too expensive to do exactly in O(1) per step.
    # We can sample or just rely on 4-cycle reduction leading to girth improvement.
    
    return count

def _local_search_girth(adj, g, time_limit=10):
    """
    Local search to eliminate short cycles.
    """
    start_time = time.time()
    n = len(adj)
    
    # Convert sets to lists for easier manipulation if needed, but sets are good for membership.
    # We need to be careful with swaps.
    
    # Initial cost
    current_cost = _count_short_cycles(adj, g)
    
    while time.time() - start_time < time_limit:
        if current_cost == 0:
             # Check if girth is actually >= g
             # _count_short_cycles only checks 4-cycles.
             # If no 4-cycles, we need to check 5, 6, 7, 8.
             if _has_girth_at_least_fast(adj, g):
                 break
        
        # Pick two random edges
        # Get all edges
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        if not edges:
            break
            
        # Pick two distinct non-adjacent edges
        # To keep it simple, pick random pairs and check validity
        found_swap = False
        for _ in range(100): # Try a few swaps per iteration
            (u, v) = random.choice(edges)
            (x, y) = random.choice(edges)
            
            # Check if edges are distinct and disjoint
            if (u == x or u == y or v == x or v == y):
                continue
                
            # Try swap: remove (u,v), (x,y); add (u,x), (v,y)
            # Check simplicity:
            if x in adj[u] or y in adj[v] or y in adj[u] or x in adj[v]:
                 # This specific swap creates multi-edge or loop?
                 # Loop: u==x? No, checked disjoint.
                 # Multi-edge: u-x exists?
                 if x in adj[u]: continue
                 if y in adj[v]: continue
                 # What about (u,y) and (v,x)? We are trying (u,x) and (v,y).
                 # We must ensure (u,x) and (v,y) are not already edges.
                 # And we must ensure we don't create loops (u!=x, v!=y).
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            new_cost = _count_short_cycles(adj, g)
            
            if new_cost < current_cost:
                current_cost = new_cost
                found_swap = True
                # Accept
            else:
                # Reject: revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj[u].add(v)
                adj[v].add(u)
                adj[x].add(y)
                adj[y].add(x)
                
        if not found_swap:
             # Stuck? Maybe try a different strategy or break
             # For now, break to avoid wasting time
             break
             
    return adj

def _has_girth_at_least_fast(adj, g):
    """
    Check girth using BFS.
    """
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _verify_graph(k, g, n, edges, time_limit=10):
    # ... (keep as is) ...
    pass

def _luw_graph(q, d):
    # ... (keep as is) ...
    pass
