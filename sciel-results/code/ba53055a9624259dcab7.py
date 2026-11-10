import time
import random
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
    best_n, best_edges = _luw_construct(k, g)
    
    # If k and g match the specific hard case, try to find a better solution
    # The known best for k=4, g=10 is n=320.
    # We will try to find a graph with n <= 320.
    
    if k == 4 and g == 10:
        # We search for n from 320 down to 243 (Moore bound approx 243)
        # We prioritize finding a solution at n=320 first, then try to reduce n.
        
        best_found_n = None
        best_found_edges = None
        
        # List of candidate n values to try.
        # We go from 320 down to 243.
        # We will allocate time dynamically.
        
        # To be efficient, we first try n=320 with a decent number of attempts.
        # If we find it, we move to 319, etc.
        
        # Note: For k=4, n can be any integer.
        # However, the graph must be simple.
        
        # We will iterate n from 320 down to 243.
        # We give more attempts to larger n (closer to 320) initially?
        # No, we want the *smallest* n.
        # But probability of success is higher for larger n (closer to 320).
        # So we should try 320, 319, 318... and if we find one at 320, we record it.
        # Then we try 319. If we find one at 319, we record it (better score).
        # If we don't find one at 319, we keep the 320 one.
        
        # Strategy:
        # 1. Scan n from 320 down to 243.
        # 2. For each n, try a number of random circulant graphs.
        # 3. If a valid graph is found, update best_found.
        # 4. Stop if time is up.
        
        # How many attempts per n?
        # If we have 110 seconds, and ~80 values of n.
        # We can afford ~1000 attempts per n if each is fast.
        # Verification is ~1-5 ms. Generation is ~10 us.
        # So ~100-500 checks per second.
        # 110s * 200 checks/s = 22,000 checks total.
        # Spread over 80 n-values: ~275 checks per n.
        
        # However, if we find a solution at 320, we might want to spend more time on 319, 318...
        # But if we don't find 320, we should spend more time on 320.
        
        # Let's just do a fixed number of attempts per n, say 200, and iterate.
        # If we find a solution, we continue to smaller n.
        
        # To improve the chance of finding the *global* minimum, we should probably
        # spend time proportional to the likelihood of success.
        # But we don't know the likelihood.
        
        # Let's try a "sweep" approach:
        # Pass 1: Try n=320 with 500 attempts. If found, store.
        # Pass 2: Try n=319 with 500 attempts. If found, store (overwrites).
        # ...
        # This is O(N_range * Attempts).
        
        # Better:
        # Iterate n from 320 down to 243.
        # For each n, check if we have time.
        # Try a batch of random circulant graphs.
        # If found, update best.
        
        # We will use a "time-slicing" approach.
        # Total time budget: 110s.
        # We will iterate n from 320 down to 243.
        # We will give each n a slice of time, but we want to favor larger n (more likely to succeed) if we haven't found anything yet?
        # No, if we haven't found anything at 320, we should try 320 more.
        
        # Let's stick to:
        # 1. Try n=320 for 10 seconds. If found, store.
        # 2. Try n=319 for 10 seconds. If found, store.
        # ...
        # This is risky if 320 is hard.
        
        # Alternative:
        # Interleave.
        # Try 100 random graphs for n=320.
        # Try 100 random graphs for n=319.
        # ...
        # This ensures we cover the range.
        
        # Let's do this:
        # Create a list of n values: 320, 319, ..., 243.
        # Shuffle? No, order matters for "best found so far".
        # We want to return the *smallest* n found.
        # So if we find 320, we hold it. If we later find 310, we update to 310.
        
        # Let's iterate n from 320 down to 243.
        # For each n, we will try `attempts_per_n` random graphs.
        # We will adjust `attempts_per_n` based on remaining time.
        
        # Initial estimate: 200 attempts per n.
        
        candidate_ns = list(range(320, 242, -1))
        
        # We will process these in order.
        # If we find a graph for n, we store it.
        
        for n in candidate_ns:
            elapsed = time.time() - start_time
            if elapsed > deadline - safety_margin:
                break
            
            # How many attempts?
            # If we have plenty of time, do more.
            # If we have little time, do fewer.
            # Let's say we want to finish all n's in the remaining time.
            # Remaining n's: len(candidate_ns) - index
            # But we don't know the index easily in this loop structure without pre-calc.
            
            # Simple heuristic:
            # If n > 300, do 300 attempts.
            # If n > 280, do 100 attempts.
            # Else, do 20 attempts.
            
            if n >= 310:
                max_attempts = 500
            elif n >= 290:
                max_attempts = 100
            else:
                max_attempts = 20
            
            # Check time inside the loop
            for _ in range(max_attempts):
                elapsed = time.time() - start_time
                if elapsed > deadline - safety_margin:
                    break
                
                # Generate random circulant graph
                edges = _generate_random_circulant(n, k)
                if edges is None:
                    continue
                
                # Verify girth
                if _verify_girth_fast(n, edges, k, g):
                    # Found a valid graph
                    if best_found_n is None or n < best_found_n:
                        best_found_n = n
                        best_found_edges = edges
                    
                    # Since we are iterating n downwards (320 -> 243),
                    # if we find one at n, it's the best for this n.
                    # We break out of the attempt loop for this n?
                    # Yes, we just need *one* valid graph for this n to be the candidate.
                    # We don't need multiple graphs for the same n.
                    break
            
            # If we found a graph at n, do we continue to n-1?
            # Yes, to try to improve the score (find smaller n).
            
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

def _generate_random_circulant(n, k):
    """
    Generate a random k-regular circulant graph on n vertices.
    For k=4, we use 2 generators a, b.
    Edges: (i, i+a mod n) and (i, i+b mod n).
    We must ensure no self-loops and no multi-edges.
    Conditions:
    a != 0, b != 0
    a != n-a (unless n=2a, which is allowed if we treat it as a single undirected edge? 
    Actually, if a = n/2, then i+a = i-n/2, so the edge is (i, i+n/2). 
    The reverse edge is (i+n/2, i+n/2+n/2) = (i+n/2, i). 
    So it's a valid simple edge.
    However, we must ensure a != b and a != n-b.
    If a == b, we have multi-edges.
    If a == n-b, then i+a = i-(n-b) = i-b, so the edge (i, i+a) is the same as (i, i-b) which is the reverse of (i-b, i). 
    Wait, (i, i+a) and (i, i-b). If a = n-b, then i+a = i-b mod n. 
    So the edge set would contain (i, i+a) and (i, i-a) [since b = n-a].
    Actually, the generators are usually taken from 1 to n//2.
    Let's pick a, b from 1 to n//2.
    Ensure a != b.
    If a == n/2, it's a single edge type.
    If a != n/2, it represents two directed edges, but in undirected graph, it's one edge type.
    For k=4, we need 4 neighbors.
    If we use generators a, b, the neighbors of i are i+a, i-a, i+b, i-b.
    This gives degree 4, provided a != b, a != n-b, a != 0, b != 0.
    Also a != n-a implies 2a != 0 mod n. If n is even and a=n/2, then i+a = i-a, so we only get 3 distinct neighbors?
    No, if a=n/2, neighbors are i+n/2, i-n/2 (same), i+b, i-b.
    So we get 3 distinct neighbors if b != a and b != n-a.
    We need 4 distinct neighbors.
    So we must avoid a = n/2 if it reduces the degree?
    If a = n/2, neighbors are {i+a, i+b, i-b}. That's 3.
    So we need to pick a, b such that the set {a, -a, b, -b} has size 4.
    This means a != b, a != -b, a != -a (i.e., 2a != 0), b != -b (i.e., 2b != 0).
    So a, b should not be n/2.
    And a != b, a != n-b.
    """
    if n < 5:
        return None
    
    # Pick a, b from 1 to n-1
    # Ensure they are distinct and not inverses
    attempts = 0
    while attempts < 100:
        a = random.randint(1, n - 1)
        b = random.randint(1, n - 1)
        
        # Check conditions
        if a == b:
            attempts += 1
            continue
        if a == n - b:
            attempts += 1
            continue
        if 2 * a % n == 0: # a = n/2
            attempts += 1
            continue
        if 2 * b % n == 0: # b = n/2
            attempts += 1
            continue
            
        # Construct edges
        edges = set()
        valid = True
        for i in range(n):
            # Neighbors: i+a, i-a, i+b, i-b
            for delta in (a, -a, b, -b):
                j = (i + delta) % n
                if i > j:
                    i, j = j, i
                # Check for self-loop (should be handled by delta != 0)
                # Check for multi-edge (handled by set)
                edges.add((i, j))
        
        # Verify it's k-regular
        # Since we generated it from circulant rules, it should be.
        # But let's double check if any collisions happened.
        # The set `edges` will have size n*k/2 if all are distinct.
        if len(edges) != n * k // 2:
            attempts += 1
            continue
            
        return list(edges)
        
    return None

def _verify_girth_fast(n, edges, k, g):
    """
    Verify girth >= g using BFS from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degrees
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check for cycles of length < g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    
    return True
