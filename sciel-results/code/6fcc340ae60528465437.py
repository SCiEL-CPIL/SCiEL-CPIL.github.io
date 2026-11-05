import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # 10 seconds safety margin
    
    # Fallback: The LUW construction from the seed
    def luw_fallback(k, g):
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

    # Try to find a smaller graph using randomized backtracking
    best_n, best_edges = luw_fallback(k, g)
    
    # Target sizes to try, descending from best_n - 1 down to Moore bound
    # Moore bound for k=3, g=16 is 512
    moore_bound = 512
    # We will try a subset of sizes to spread the effort
    # If we find a graph of size N, we can stop or try smaller.
    # Given time constraints, we'll try a few sizes in random order or descending.
    
    # To maximize chance of finding a smaller graph, we try sizes close to the current best first?
    # No, smaller is better. But harder.
    # Let's try a mix.
    
    # Heuristic: Try sizes: 935, 930, 900, 800, 700, 600, 512
    # But we must ensure we don't spend all time on one.
    
    target_sizes = list(range(best_n - 1, moore_bound - 1, -10))
    # Shuffle a bit to avoid bias? No, descending is logical for "improvement".
    # Actually, finding a graph of size 935 is easier than 512.
    # So we should try 935, 934... until we find one?
    # If we find 935, score is 935.
    # If we find 512, score is 512.
    # We want the minimum.
    # So we should try smaller sizes if we have time.
    # But if we don't find any, we return 936.
    
    # Let's try a batch of sizes.
    # We'll iterate through a list of candidate Ns.
    candidates = [best_n - 1, best_n - 5, best_n - 20, 900, 850, 800, 750, 700, 650, 600, 550, 512]
    # Filter valid candidates
    candidates = [n for n in candidates if moore_bound <= n < best_n]
    
    for N in candidates:
        if time.time() > deadline:
            break
            
        # Try to find a graph of size N
        # Use randomized backtracking with a time limit per N
        n_found, edges_found = try_find_graph(k, g, N, time_limit=5.0)
        if n_found is not None:
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found
            # If we found a graph, we can continue to try smaller ones if time permits
            # But if we found one close to the bound, we might be happy?
            # No, we want to minimize. Keep trying smaller if time allows.
            # However, if we found 512, we can stop.
            if best_n == moore_bound:
                break
                
    return best_n, best_edges

def try_find_graph(k, g, N, time_limit=5.0):
    start_time = time.time()
    max_time = time_limit
    
    # Randomized Backtracking
    # We will try a few random restarts
    for restart in range(100):
        if time.time() - start_time > max_time:
            break
        
        # Initialize
        adj = [[] for _ in range(N)]
        edges = []
        
        # Randomly order vertices to break symmetry?
        # We'll pick vertices to connect in a random order.
        # But we need to ensure degree k.
        
        # Strategy:
        # 1. Create a list of "sockets" (vertex, degree_slot).
        # 2. Randomly pair them? No, that's random graph, likely short cycles.
        # 3. Backtracking:
        #    - Pick a vertex u with deg < k.
        #    - Pick a vertex v with deg < k.
        #    - Check if edge (u,v) is valid (dist >= g-1).
        #    - If yes, add, recurse.
        
        # To speed up, we use a recursive function with a node limit.
        node_limit = 5000
        nodes_visited = [0]
        
        def solve(idx, adj, edges):
            if time.time() - start_time > max_time:
                return False
            if nodes_visited[0] > node_limit:
                return False
            
            nodes_visited[0] += 1
            
            # Find a vertex with degree < k
            # Prefer vertices with higher degree (more constrained) or random?
            # Let's pick a random vertex with deg < k to maintain randomness
            candidates_u = [i for i in range(N) if len(adj[i]) < k]
            if not candidates_u:
                return True # Success
            
            u = random.choice(candidates_u)
            
            # Find candidates for v
            candidates_v = [i for i in range(N) if len(adj[i]) < k and i != u and i not in adj[u]]
            if not candidates_v:
                return False
            
            # Shuffle candidates_v
            random.shuffle(candidates_v)
            
            for v in candidates_v:
                # Check girth constraint
                # dist(u, v) must be >= g - 1
                if is_valid_edge(adj, u, v, g):
                    # Add edge
                    adj[u].append(v)
                    adj[v].append(u)
                    edges.append((u, v))
                    
                    if solve(idx+1, adj, edges):
                        return True
                    
                    # Backtrack
                    adj[u].pop()
                    adj[v].pop()
                    edges.pop()
            
            return False
        
        if solve(0, adj, edges):
            # Verify the graph
            if verify_graph(N, adj, edges, k, g):
                return N, edges
        
    return None, None

def is_valid_edge(adj, u, v, g):
    # BFS to find shortest path from u to v
    if u == v:
        return False
    if v in adj[u]:
        return False # Already connected, would create cycle of length 2 (if multigraph) or 3?
                     # Simple graph, so no repeated edges.
    
    # BFS
    visited = {u: 0}
    queue = deque([(u, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= g - 1:
            return True # Distance is large enough
        for neighbor in adj[node]:
            if neighbor not in visited:
                if neighbor == v:
                    # Found v at dist + 1
                    return (dist + 1) >= g - 1
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    # If v is not reachable, distance is infinity, which is valid
    return True

def verify_graph(N, adj, edges, k, g):
    # Check degree
    for i in range(N):
        if len(adj[i]) != k:
            return False
    
    # Check girth
    # For each edge (u,v), check if there is a path of length < g-1 between u and v
    # Actually, check all pairs? Too slow.
    # Check cycles:
    # A graph has girth >= g iff for every edge (u,v), dist(u,v) >= g-1 in G - (u,v).
    # We can check this for all edges.
    for u, v in edges:
        # Remove edge temporarily?
        # We can just check if there is a path of length < g-1 in the graph *excluding* the direct edge.
        # Since it's a simple graph, the direct edge is the only one between u and v.
        # So we check if dist(u, v) in G \ {(u,v)} >= g - 1.
        
        # BFS from u, avoiding direct edge to v?
        # Actually, if there is a path of length L < g-1, then u and v are connected by a path of length L.
        # Adding the edge (u,v) creates a cycle of length L+1.
        # We need L+1 >= g => L >= g-1.
        
        visited = {u: 0}
        queue = deque([(u, 0)])
        found_short = False
        while queue:
            node, dist = queue.popleft()
            if dist >= g - 1:
                break
            for neighbor in adj[node]:
                if neighbor == u and node == v:
                    continue # Avoid the edge itself? No, we start at u.
                # We need to avoid traversing the edge (u,v) in the reverse direction if we started at u?
                # Actually, we just want to know if there is a path from u to v of length < g-1.
                # The path cannot use the edge (u,v) directly as the first step?
                # No, if we go u->v, that's length 1.
                # We want to know if there is a *different* path.
                # So we should not allow the direct edge (u,v) to be used.
                
                if neighbor == v and dist == 0:
                    continue # Don't take the direct edge u-v
                if neighbor not in visited:
                    if neighbor == v:
                        if dist + 1 < g - 1:
                            found_short = True
                            break
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
            if found_short:
                break
        
        if found_short:
            return False
            
    return True

# EVOLVE-BLOCK-END
