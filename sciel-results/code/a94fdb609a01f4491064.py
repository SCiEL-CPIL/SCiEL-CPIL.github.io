import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 110
    
    # Target N. The problem states best known is 1116.
    # We will try to construct a graph with N=1116.
    N = 1116
    
    def generate_random_regular_graph(N, k, seed):
        random.seed(seed)
        adj = [[] for _ in range(N)]
        # Configuration model
        # Create list of stubs: [0,0,0,0,0, 1,1,1,1,1, ...]
        stubs = []
        for i in range(N):
            for _ in range(k):
                stubs.append(i)
        
        # We need to pair stubs to form edges.
        # Simple method: Shuffle stubs, then pair (0,1), (2,3)...
        # This creates a multigraph. We need to remove loops and multi-edges.
        
        while True:
            random.shuffle(stubs)
            adj = [[] for _ in range(N)]
            valid = True
            for i in range(0, N * k, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                return adj
        
    def calculate_girth(adj, N, max_depth):
        # Returns the length of the shortest cycle found, or max_depth + 1 if none found < max_depth + 1
        # We want to know if girth < g. So we check up to depth g-1.
        # If we find a cycle of length L < g, return L.
        # If no cycle < g, return g.
        
        for start in range(N):
            # BFS from start
            # visited[node] = (distance, parent)
            visited = {start: (0, -1)}
            queue = deque([(start, 0, -1)])
            
            while queue:
                curr, dist, parent = queue.popleft()
                if dist >= max_depth:
                    continue
                
                for neighbor in adj[curr]:
                    if neighbor == parent:
                        continue
                    if neighbor not in visited:
                        visited[neighbor] = (dist + 1, curr)
                        queue.append((neighbor, dist + 1, curr))
                    else:
                        # Found a cycle
                        # The cycle length is dist + 1 + dist_to_neighbor
                        # Wait, BFS visits nodes in increasing distance.
                        # If neighbor is already visited, the cycle length is dist + 1 + visited[neighbor][0]
                        # But we must ensure the path back doesn't go through parent?
                        # Standard BFS girth calculation:
                        # When we encounter a visited node 'neighbor' from 'curr',
                        # the cycle length is dist(curr) + 1 + dist(neighbor).
                        # However, this counts the cycle twice if we're not careful, or we just take the min.
                        cycle_len = dist + 1 + visited[neighbor][0]
                        if cycle_len < max_depth:
                            return cycle_len
        
        return max_depth

    def get_short_cycle(adj, N, limit):
        # Returns a list of vertices forming a cycle of length < limit, or None
        for start in range(N):
            visited = {start: (0, -1)}
            queue = deque([(start, 0, -1)])
            
            while queue:
                curr, dist, parent = queue.popleft()
                if dist >= limit - 1:
                    continue
                
                for neighbor in adj[curr]:
                    if neighbor == parent:
                        continue
                    if neighbor not in visited:
                        visited[neighbor] = (dist + 1, curr)
                        queue.append((neighbor, dist + 1, curr))
                    else:
                        if neighbor != start: # Avoid trivial backtracking
                            # Check if this forms a cycle
                            # We need to reconstruct the path
                            # This is slightly more complex. For now, let's just return the length
                            # If we need the cycle for rewiring, we can find it later.
                            pass
        return None # Simplified for now, we will just use girth value

    # Main Loop
    best_girth = 0
    best_adj = None
    
    # Try a few seeds for initialization
    for seed in range(20):
        if time.time() - start_time > time_limit:
            break
            
        adj = generate_random_regular_graph(N, k, seed)
        
        # Calculate initial girth
        # We check if girth < g. If girth >= g, we are done.
        # To save time, we can check girth incrementally or just run the check.
        
        # Simulated Annealing
        current_temp = 1.0
        cooling = 0.99
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                return N, [] # Timeout, return best found or fallback
                
            # Calculate current girth
            # Optimization: Only check if girth might be < g
            # A full BFS is expensive.
            
            # Let's check girth
            girth = calculate_girth(adj, N, g)
            
            if girth >= g:
                # Success!
                edges = []
                for u in range(N):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return N, edges
            
            # If girth < g, try to improve
            # Find a short cycle and rewire
            # For simplicity in this implementation, we will do random rewiring and hope it improves
            # A better method: Pick a random edge (u,v) and a random other edge (a,b)
            # and swap if it doesn't create loops/multi-edges.
            
            # To make it directed, we need to find a cycle.
            # Let's just do random 2-opt swaps.
            
            # Pick random u, v from adj
            u = random.randint(0, N - 1)
            if not adj[u]:
                continue
            v = random.choice(adj[u])
            
            # Pick random a, b
            a = random.randint(0, N - 1)
            if not adj[a]:
                continue
            b = random.choice(adj[a])
            
            # Ensure (a,b) is not (u,v) or (v,u)
            if (a == u and b == v) or (a == v and b == u):
                continue
            
            # Check validity of swap:
            # Remove (u,v) and (a,b), add (u,a) and (v,b)
            # Must not create loops: u != a, v != b
            # Must not create multi-edges: a not in adj[u] (after removal), b not in adj[v]
            
            if u == a or v == b:
                continue
            if a in adj[u] or b in adj[v]:
                continue
            # Also check (u,b) and (v,a) if we swap the other way
            # Let's try swap 1: (u,a) and (v,b)
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[a].remove(b)
            adj[b].remove(a)
            
            adj[u].append(a)
            adj[a].append(u)
            adj[v].append(b)
            adj[b].append(v)
            
            # Check new girth
            new_girth = calculate_girth(adj, N, g)
            
            # Acceptance criterion
            if new_girth >= girth:
                pass # Accept
            else:
                # Reject if worse, unless random probability (annealing)
                # delta = new_girth - girth (negative)
                # prob = exp(delta / temp)
                if random.random() > np.exp((new_girth - girth) / current_temp):
                    # Revert
                    adj[u].remove(a)
                    adj[a].remove(u)
                    adj[v].remove(b)
                    adj[b].remove(v)
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[a].append(b)
                    adj[b].append(a)
            
            current_temp *= cooling
            if current_temp < 0.01:
                break
        
        # If we didn't find a valid graph in this seed, try next seed
        # We don't keep state between seeds in this simple version, 
        # but we could keep the best_adj found so far.
        
    # Fallback if all seeds fail
    # Return the seed's known construction or a trivial one?
    # The task requires a valid graph.
    # If we fail, we must return *something* valid.
    # The seed program has a fallback. We should include a robust fallback.
    
    # Re-implement the fallback from the seed to guarantee validity
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
