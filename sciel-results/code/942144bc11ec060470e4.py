"""
Constructs a 5-regular graph of girth 9 with as few vertices as possible.
Uses a combination of the Lazebnik-Ustimenko-Woldar (LUW) construction and
random search for smaller valid graphs.
"""
import random
import time
import numpy as np

def construct(k, g):
    # We focus on the task's specific parameters: k=5, g=9.
    # For other parameters, fall back to LUW.
    if k != 5 or g != 9:
        return _luw_construct(k, g)

    deadline = time.time() + 115  # 115 seconds max (safety margin)
    
    # 1. Get the LUW construction as a baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Try to find a smaller graph using random search
    # The Moore bound for (5,9) is 428.
    # We search downwards from n_luw - 1 to 428.
    
    # To generate a random 5-regular graph:
    # We use the configuration model (pairing stubs).
    
    def generate_random_5reg(n):
        """Generates a random simple 5-regular graph on n vertices."""
        if n < 5:
            return None
        # Each vertex has 5 stubs
        stubs = []
        for i in range(n):
            for _ in range(5):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        edge_set = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None # Loop
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return None # Multi-edge
            edge_set.add((u, v))
            edges.append((u, v))
        
        return edges

    def has_girth_at_least_9(n, edges):
        """Checks if the graph has girth >= 9."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, check if the ball of radius 4 is a tree
        # Number of vertices in ball of radius 4 in a 5-regular tree:
        # 1 (center) + 5 (dist 1) + 5*4 (dist 2) + 5*4*4 (dist 3) + 5*4*4*4 (dist 4)
        # = 1 + 5 + 20 + 80 + 320 = 426
        target_count = 426
        
        for start in range(n):
            # BFS from start up to depth 4
            visited = [-1] * n # -1 for unvisited, else distance
            parent = [-1] * n
            queue = [(start, 0)]
            visited[start] = 0
            count = 1
            
            while queue:
                u, d = queue.pop(0)
                if d == 4:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = d + 1
                        parent[v] = u
                        count += 1
                        queue.append((v, d + 1))
                    elif v != parent[u]:
                        # Found a back-edge to a non-parent
                        # This implies a cycle.
                        # Cycle length = d + 1 + visited[v]
                        # If this cycle length < 9, we fail.
                        # Actually, if we found a back-edge, the cycle length is d + 1 + visited[v].
                        # We need to ensure NO cycle < 9.
                        # If visited[v] <= 3 and d <= 3, cycle could be short.
                        # Specifically, if we encounter a visited node v (not parent) at depth d,
                        # the cycle length is d + 1 + visited[v].
                        # We need d + 1 + visited[v] >= 9.
                        cycle_len = d + 1 + visited[v]
                        if cycle_len < 9:
                            return False
            
            if count != target_count:
                # If the count is not 426, either there's a cycle (shorter path)
                # or the graph is not 5-regular (shouldn't happen) or n is too small.
                # If n < 426, it's impossible.
                if n < 426:
                    return False
                # If count < 426, there was a cycle causing a shorter path.
                # If count > 426, impossible for 5-reg tree.
                # So if count != 426, it's not a tree of radius 4.
                # This implies girth < 9.
                return False
                
        return True

    # Search for smaller n
    # Start from n_luw - 1 down to 428
    # To save time, we can skip some values or limit attempts per n.
    
    # Heuristic: Start with larger n (closer to n_luw) as they are more likely to be valid?
    # Actually, probability of girth >= 9 increases with n.
    # So we should try n close to n_luw first?
    # Or just iterate down?
    # Let's iterate down from n_luw - 1 to 428.
    
    # Optimization: If n < 426, it's impossible.
    min_n = max(428, 426)
    
    # We will try each n from n_luw - 1 down to min_n.
    # For each n, we try a limited number of random graphs.
    # The time limit is 115s.
    # Checking a graph takes ~ n * 426 operations. For n=2000, ~1M ops. Very fast.
    # Generating a graph takes ~ n log n.
    # So we can try many graphs.
    
    # Let's try to be smart:
    # We know random graphs of size ~1000 have girth ~10.
    # So we should find one quickly around 1000-1500.
    # We can start from n_luw - 1 and go down.
    
    current_time = time.time()
    
    for n in range(n_luw - 1, min_n - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Limit attempts per n to avoid spending too much time on large n
        # that are likely valid but we want smaller ones.
        # Actually, if we find a valid one at n, we stop.
        # So we just need to find ANY valid n < n_luw.
        # The first one we find (starting from n_luw - 1) will be the smallest?
        # No, we are going down from n_luw - 1.
        # So the first valid one found will be close to n_luw.
        # To get the *smallest*, we should go up from 428?
        # But 428 is unlikely to have a valid graph.
        # It's a trade-off.
        # Given the 120s limit, going down from n_luw is safer to find *a* valid graph.
        # But the score is minimized n.
        # So we want the smallest n.
        
        # Let's try a binary search or linear search from the bottom?
        # If we go from 428 up, we might spend a lot of time on impossible sizes.
        # If we go from n_luw down, we find a valid one quickly, but it might be large.
        
        # Compromise:
        # We know n=1000 is likely valid.
        # Let's try n from 1000 to n_luw-1?
        # Or just go down from n_luw-1 and if we find one, keep going down for a bit?
        
        # Let's just go down from n_luw - 1.
        # If we find a valid graph at n, we can update best.
        # But we must return the best (smallest) found.
        
        # To find the *smallest*, we should search from 428 upwards.
        # But that's risky if 428-900 are all invalid (likely).
        
        # Let's try a range: 428 to n_luw.
        # We will try a few random graphs for each n.
        
        # Estimate time:
        # 115 seconds.
        # If we try 100 graphs per n, and n ~ 1000, check takes ~0.01s.
        # 100 * 0.01 = 1s per n.
        # 2000 values of n -> 2000s. Too slow.
        
        # We need to be selective.
        # Let's try n in steps of 10? Or just try a few specific n values?
        
        # Actually, the LUW graph is 2500.
        # Let's try n = 2400, 2300, ..., 1000.
        # If we find one, we can then try to go smaller?
        
        # Better strategy:
        # 1. Find any valid graph with n < n_luw.
        # 2. Then try to find smaller ones.
        
        # Let's just iterate n from n_luw - 1 down to 428.
        # For each n, try 50 random graphs.
        # If found, update best_n and best_edges.
        # Continue searching for smaller n.
        # Stop when time is up.
        
        # To speed up, we can skip n if it's not a multiple of something?
        # No, n can be any integer >= 428.
        
        # Let's limit the number of attempts per n to 20.
        # 20 * (gen + check).
        # Gen: O(n). Check: O(n * 426).
        # For n=2000, Check is ~1M ops. ~0.05s.
        # 20 * 0.05 = 1s per n.
        # 2000 n's -> 2000s. Still too slow.
        
        # We must reduce the range or attempts.
        # Let's try n in steps of 50?
        # Or just try a few hundred n's?
        
        # Let's try n from n_luw - 1 down to 1000, step 10.
        # And for each, try 5 graphs.
        
        # Actually, let's just try to find the smallest n in a targeted range.
        # We suspect the minimum is around 1000-1500.
        
        # Let's try n from 428 to n_luw - 1, but only test a subset.
        # Or, we can use the fact that if a graph of size n is valid,
        # a graph of size n+1 might be valid.
        
        # Let's just do a linear search from n_luw - 1 down to 428,
        # but with a very low number of attempts (e.g., 5) per n.
        # And break if we find one? No, we want the smallest.
        
        # Re-evaluating:
        # If we find a valid graph at n=1200, can we find one at n=1100?
        # Likely yes.
        # So we should search downwards.
        
        # To fit in time, we will only test every 10th n value,
        # and if we find a valid one, we will test the 9 smaller values in detail.
        
        pass # Placeholder for logic

    # Implementing the search logic
    # We will iterate n from n_luw - 1 down to 428.
    # We will use a step of 1 initially, but if it's too slow, we'll adapt.
    # Actually, let's just try to find *any* valid graph smaller than n_luw.
    # If we find one at n, we set best_n = n.
    # Then we continue searching for n-1, n-2...
    
    # To make it fast, we will only try 5 random graphs per n.
    # And we will stop if time is up.
    
    # We start from n_luw - 1.
    # If we find a valid one, we update best.
    # We continue down.
    
    # To ensure we don't spend all time on large n, we can skip some.
    # But large n are more likely to be valid.
    
    # Let's try this:
    # 1. Try n = n_luw - 1, n_luw - 2, ..., n_luw - 100. (100 values)
    #    With 5 attempts each. 500 graphs.
    #    If n=2400, check is fast.
    # 2. If found, update best.
    # 3. Then try n = best_n - 1, ..., best_n - 100.
    #    Repeat.
    
    # This "jump" strategy allows us to descend quickly.
    
    # Let's implement this.
    
    current_n = n_luw - 1
    
    while current_n >= min_n and time.time() < deadline - 10:
        found_in_block = False
        # Try a block of 100 values
        end_n = max(min_n, current_n - 99)
        
        for n in range(current_n, end_n - 1, -1):
            if time.time() > deadline - 10:
                break
                
            # Try 5 random graphs
            for _ in range(5):
                edges = generate_random_5reg(n)
                if edges is None:
                    continue
                if has_girth_at_least_9(n, edges):
                    # Found a valid graph
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    found_in_block = True
                    break # Break inner loop, move to n-1?
                    # Actually, if we found one at n, we should try n-1 next.
                    # So we break the inner loop, and the outer loop continues to n-1.
            
            if found_in_block:
                # We found a valid graph in this block.
                # We can break out of the block loop and restart from best_n - 1.
                # But we are already iterating down.
                # So we just continue.
                pass
        
        # Move to the next block
        current_n = end_n - 1
        
    return best_n, best_edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
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
