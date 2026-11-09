import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 115s to be safe against 120s limit
    
    # Moore bound for odd girth g = 2r + 1
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        # Even girth case is different, but task specifies girth 9 (odd)
        # Fallback for even girth
        r = g // 2
        moore = k * sum((k - 1) ** i for i in range(r - 1))
        
    # Start slightly above Moore bound
    start_n = max(50, moore + 10)
    
    # We will try n values increasing from start_n
    # Step size can be small to find the minimum
    step_n = 10 
    
    # To avoid getting stuck on a very small n that requires many retries,
    # we limit the number of attempts per n based on remaining time.
    
    current_n = start_n
    max_n = 5000 # Hard cap to prevent infinite loop if something goes wrong
    
    while current_n <= max_n:
        # Check time
        if time.time() > deadline - 10: # 10s buffer for final check
            break
            
        # Estimate attempts allowed for this n
        # If n is small, probability of success is low, so we might need many tries.
        # If n is large, probability is high, fewer tries needed.
        # Let's just try a fixed number of times per n, or until a small time slice is used.
        attempts_for_n = 0
        max_attempts_for_n = 2000 # Cap attempts per n to avoid stalling
        
        while attempts_for_n < max_attempts_for_n:
            if time.time() > deadline - 5:
                break
                
            # Generate random k-regular graph with current_n vertices
            graph, is_simple = generate_random_regular_graph(current_n, k)
            attempts_for_n += 1
            
            if is_simple:
                # Check girth
                girth = compute_girth(graph, current_n)
                if girth >= g:
                    # Return edges
                    edges = []
                    for u in range(current_n):
                        for v in graph[u]:
                            if u < v:
                                edges.append((u, v))
                    return current_n, edges
            # If not simple, discard and try again
            
        # If we exhausted attempts for this n, increase n
        current_n += step_n
        
    # If we ran out of time or n range, return a valid graph from the last known good n?
    # Actually, the loop above returns as soon as a valid one is found.
    # If we exit the loop without returning, we must return *something* valid.
    # This implies we failed to find one in the time budget.
    # We should probably fall back to a known valid construction (like the seed) or the last generated simple graph (even if girth < g) is INVALID.
    # The task requires a VALID graph.
    # If we didn't find one, we are in trouble.
    # However, random regular graphs of size ~1000-2000 usually have girth > 9.
    # Let's ensure we don't exit without a valid graph.
    
    # Fallback: Use the seed construction if we failed to find a random one in time?
    # The seed construction is deterministic and valid.
    # But it's large.
    # Let's try to force one more attempt with a larger n if time permits.
    # Or just use the seed logic as a last resort.
    
    # Re-implementing the seed logic briefly as fallback is risky/verbose.
    # Better: Ensure the random search is robust.
    # For k=5, g=9, n=1000 should be very likely to work.
    # If we reach this point, we failed.
    # Let's return a simple cycle? No, must be k-regular.
    # Let's return the seed graph.
    
    # Calling the seed logic (copied from prompt)
    return _seed_construct(k, g)

def generate_random_regular_graph(n, k):
    """
    Generates a random k-regular graph on n vertices using the configuration model.
    Returns (adj_list, is_simple).
    adj_list is a list of lists.
    """
    if (n * k) % 2 != 0:
        return None, False
        
    # Create stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    # Pair stubs
    edges = set()
    adj = [set() for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        
        # Check for loops
        if u == v:
            return None, False
            
        # Check for multi-edges
        if v in adj[u]:
            return None, False
            
        adj[u].add(v)
        adj[v].add(u)
        edges.add((min(u, v), max(u, v)))
        
    # Convert sets to lists for faster iteration later? 
    # Sets are fine for girth check if we convert to list of lists for the return.
    # But for girth check, sets allow O(1) neighbor lookup? No, we iterate neighbors.
    # List is faster for iteration.
    adj_list = [list(s) for s in adj]
    
    return adj_list, True

def compute_girth(adj_list, n):
    """
    Computes the girth of a graph using BFS from each vertex.
    """
    min_girth = float('inf')
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        found_cycle = False
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # We need to ensure this is the shortest cycle involving u and v?
                    # Actually, BFS finds shortest paths.
                    # If we see a visited node v that is not the parent,
                    # we have a cycle.
                    # Is it the girth?
                    # The shortest cycle in the graph is the minimum over all such back-edges.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    # We can break early if min_girth is 3? No, we want the global min.
                    # But if we find a cycle of length 3, that's the absolute minimum possible.
                    # If min_girth == 3, we can return 3 immediately.
                    if min_girth == 3:
                        return 3
        # Note: This BFS approach finds the shortest cycle *reachable* from start that is "discovered" by this specific BFS tree structure.
        # It is correct for finding the girth if we take the min over all starts.
        # However, a standard BFS from a node finds the shortest cycle *containing* that node?
        # Actually, it finds the shortest cycle in the graph?
        # Let's verify.
        # The standard algorithm for girth using BFS:
        # For each vertex v, run BFS. When an edge (u, w) is encountered where w is visited and w != parent[u],
        # the cycle length is dist[u] + dist[w] + 1.
        # The minimum of these values over all v and all such edges is the girth.
        
    return min_girth if min_girth != float('inf') else n # Fallback

def _seed_construct(k, g):
    """
    Fallback to the original Lazebnik-Ustimenko-Woldar construction.
    """
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
