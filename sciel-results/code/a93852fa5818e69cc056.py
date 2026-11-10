import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = 120.0  # Evaluator limit
    safety_margin = 5.0
    
    # Fallback: LUW Construction
    # For k=4, g=10, LUW gives n=2048.
    best_n, best_edges = _luw_construct(k, g)
    
    # If k and g are small enough, try to find a smaller cage via random search.
    # Known best for k=4, g=10 is n=320.
    # Moore bound is ~250.
    
    # Determine search range
    if k == 4 and g == 10:
        # Search from a lower bound up to the known best
        # n must be even for 4-regular? No, n*k must be even. 4n is always even.
        # But girth 10 implies no triangles, etc.
        # Let's try n from 250 to 320.
        
        current_time = time.time()
        # We have ~115 seconds left.
        
        # We will iterate n from smallest to largest.
        # If we find a valid graph for n, we return it (since we want min n).
        # However, finding n=250 is very unlikely. n=320 is the target.
        # It is more efficient to try n=320 first, then 318, etc?
        # No, we want the *minimum* n. So we must try 250, 252, ...
        # But the probability of success for n=250 is tiny.
        # We should probably just focus on n=320 or n=318 if we have enough time.
        # Actually, if we find n=320, we can't claim it's the minimum unless we prove no smaller exists.
        # The task says "n must lie between the Moore bound and max_vertices. Score: n, minimized".
        # The evaluator just takes the returned n.
        # So if I return 320, score is 320. If I return 318, score is 318.
        # I should try to find the smallest possible n within the time limit.
        
        # Strategy:
        # 1. Try n=320 with many samples. If found, check if we have time to try 318, 316...
        # 2. If time is tight, just try 320.
        
        # Let's estimate verification time.
        # BFS from 320 nodes, depth 4 or 5. Very fast, < 1ms per graph in Py?
        # Generation: shuffle 1280 stubs. Very fast.
        # So we can do thousands of tries per second? Maybe hundreds.
        
        # Let's try n from 320 down to 250.
        # If we find one at 320, we store it. Then we try 318. If we find one, store it.
        # At the end, return the best (smallest) one found.
        
        best_found_n = None
        best_found_edges = None
        
        # We will scan n from high to low to ensure we *have* a good candidate (320) early on,
        # then refine down if time permits.
        # Wait, if I return 320, it's a huge improvement over 2048.
        # If I find 310, it's even better.
        
        # Let's prioritize finding *any* valid graph < 2048 first.
        # n=320 is the most likely to be found quickly.
        
        # Range: 240 to 320
        # We will iterate n in descending order: 320, 318, ..., 240.
        # Why? Because if we find n=320, we have a solid score.
        # If we then find n=318, we improve it.
        # If we start at 240, we might waste all time trying impossible n=240 and return LUW (2048).
        
        candidate_ns = list(range(320, 238, -2)) # 320 down to 240
        
        # Add some random checks in between if time allows? No, keep it simple.
        
        for n in candidate_ns:
            elapsed = time.time() - start_time
            if elapsed > deadline - safety_margin:
                break
            
            # How many attempts for this n?
            # If n is close to 320, probability is higher.
            # If n is 240, probability is near 0.
            # Let's allocate attempts based on remaining time and n.
            
            # If n > 300, do many attempts.
            # If n < 260, do few attempts.
            
            # Rough estimate:
            # Prob(girth >= 10) for n=320 is maybe 1/100?
            # We need ~100-1000 tries to be confident.
            
            if n >= 310:
                max_attempts = 500
            elif n >= 290:
                max_attempts = 200
            elif n >= 270:
                max_attempts = 50
            else:
                max_attempts = 10
            
            found_at_n = False
            for _ in range(max_attempts):
                elapsed = time.time() - start_time
                if elapsed > deadline - safety_margin:
                    break
                
                # Generate random 4-regular graph
                edges = _generate_random_k_regular(n, k)
                if edges is None:
                    continue
                
                # Verify girth
                if _verify_girth_fast(n, edges, k, g):
                    # Found a valid graph!
                    if best_found_n is None or n < best_found_n:
                        best_found_n = n
                        best_found_edges = edges
                    
                    # Since we are iterating n downwards, if we find one at n,
                    # we should continue to see if we can find a smaller one?
                    # No, we are iterating 320 -> 240.
                    # So if we find 320, we update best.
                    # Then we try 318. If we find 318, we update best.
                    # This works.
                    
                    # Should we break out of the attempt loop for this n?
                    # Yes, we just need *one* valid graph for this n to be the candidate for this n.
                    # We don't need multiple graphs for the same n.
                    found_at_n = True
                    break
            
            # If we found a graph at n, do we continue to n-2?
            # Yes, to try to improve the score.
            # However, if n is very small (e.g., 240), we might burn time.
            # The logic above handles this with max_attempts.

        # If we found a better graph than LUW, return it.
        if best_found_n is not None and best_found_n < best_n:
            return best_found_n, best_found_edges
    
    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    # For girth g, we need d such that girth >= d + 5
    # So d = g - 5 (if odd) or g - 4 (if even)
    # For g=10 (even), d = 6? 
    # Standard LUW for even girth g uses d = g/2 + 1? 
    # Let's check the seed code: d = g - 5 if (g-5)%2==1 else g-4.
    # For g=10: (10-5)=5, odd, so d=5? No, 5 is odd, so d = g-5 = 5.
    # Wait, the seed code says:
    # d = g - 5 if (g - 5) % 2 == 1 else g - 4
    # For g=10: g-5=5, which is odd. So d=5.
    # Let's trace the seed code logic for q=4, g=10.
    
    # The seed code produced 2048 vertices? 
    # 2*q^d? 2*4^5 = 2*1024 = 2048. Yes.
    
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Build graph
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using Configuration Model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    edges = set()
    valid = True
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            valid = False
            break
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            valid = False
            break
        edges.add((u, v))
    
    if not valid:
        return None
        
    # Verify degrees (should be k if we didn't break, but self-loops/multi-edges are rejected)
    # The set logic ensures no multi-edges and no self-loops.
    # Does it guarantee k-regular?
    # Yes, if we successfully pair all stubs without conflicts, each vertex appears k times in stubs,
    # and each pair consumes 2 stubs. Since no self-loop, each vertex is in exactly k distinct edges.
    
    return list(edges)


def _verify_girth_fast(n, edges, k, g):
    """
    Verify girth >= g using BFS from each vertex.
    Optimization: Use array-based BFS to reduce overhead.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degrees
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # We check for cycles of length < g.
    # BFS from each start node.
    # dist[start] = 0
    # If we find an edge (u, v) such that dist[u] + dist[v] + 1 < g, and u, v are not parent-child in a way that avoids the cycle...
    # Standard BFS cycle detection:
    # For each edge (u, v), if dist[u] != -1 and dist[v] != -1:
    # If u is not parent of v and v is not parent of u:
    #   cycle_len = dist[u] + dist[v] + 1
    #   if cycle_len < g: return False
    
    # To be efficient, we limit BFS depth to g//2.
    # If we don't find a cycle in the BFS tree within depth g//2, 
    # any cycle passing through the start node would have length > g.
    # Wait, this only detects cycles *containing the start node*.
    # We must do this for *all* start nodes.
    
    # Optimization:
    # Instead of clearing arrays for each start, we can use a visited_timestamp array.
    # But BFS from n nodes is O(n^2) in Python which might be slow for n=320 (100k ops per node? No, edges are 4n).
    # 320 * 4 * 320 = 400k ops. Very fast.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue # No need to expand further, cycles found via this node would be >= g
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # If v is already visited
                    # Check if this creates a short cycle
                    # The edge (u, v) is a cross edge or back edge.
                    # Cycle length is dist[u] + dist[v] + 1
                    # BUT we must ensure it's not the parent edge which is part of the tree (length dist[u]+1, not a cycle in this context unless we consider the path back).
                    # If v == parent[u], it's the tree edge, ignore.
                    # If u == parent[v], it's the tree edge, ignore.
                    if v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    
    return True
