import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Target n for 4-regular girth 10 is 320
    # Moore bound for (4,10) is 1 + 4*sum_{i=0}^{4} 3^i = 1 + 4*(1+3+9+27+81) = 1 + 4*121 = 485? 
    # Wait, Moore bound formula: n >= 1 + k * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i
    # For g=10, floor(9/2) = 4. Sum i=0 to 3? No, radius is g/2 - 1 = 4.
    # n >= 1 + k * sum_{i=0}^{g/2 - 2} (k-1)^i ?
    # Let's check: n >= 1 + k * (1 + (k-1) + ... + (k-1)^{g/2-2})
    # For k=4, g=10: n >= 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4*40 = 161.
    # Wait, the prompt says best lower bound is 243.
    # So n=320 is a candidate.
    
    n_target = 320
    
    def generate_random_k_regular(n, k):
        """Generate a random k-regular graph on n vertices."""
        # Configuration model
        # 1. Create a list of 2n stubs, each vertex i has k stubs.
        # 2. Shuffle the stubs.
        # 3. Pair them up to form edges.
        # 4. Check for loops and multi-edges. If found, retry.
        
        while True:
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            edges = []
            valid = True
            edge_set = set()
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edge_set or (v, u) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
                edges.append((u, v))
            
            if valid:
                # Convert to adjacency list
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                return adj
    
    def check_girth(adj, n, g):
        """Check if the graph has girth at least g. Returns True if yes."""
        for start in range(n):
            # BFS from start
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] >= g // 2: 
                    # If we reach depth g/2, we don't need to go deeper for girth < g
                    # Actually, if we find a cycle of length < g, we fail.
                    # A cycle of length L is found if we visit a neighbor v of u 
                    # such that dist[v] != -1 and v is not the parent of u.
                    # If dist[u] + dist[v] + 1 < g, we have a short cycle.
                    pass
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def find_short_cycle(adj, n, g):
        """Find a cycle shorter than g. Returns the cycle as a list of vertices."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = [u]
                            curr = u
                            while curr != v:
                                curr = parent[curr]
                                cycle.append(curr)
                            # Add path from v to start? No, the cycle is u-v-parent[v]...
                            # The cycle is u, v, parent[v], ..., start?
                            # Actually, the cycle is the path from u to v via parents + edge (u,v).
                            # Path u -> ... -> start? No.
                            # The cycle is formed by the edge (u,v) and the paths from u and v to their LCA.
                            # Since we are doing BFS from start, the path is u->...->start and v->...->start?
                            # No, if v is already visited, it might not be on the path to start from u.
                            # Standard BFS cycle detection:
                            # If v is visited and v is not parent[u], then we have a cycle.
                            # The cycle length is dist[u] + dist[v] + 1.
                            # To reconstruct, we need to find the LCA of u and v.
                            # This is complex. For now, just return existence or a dummy.
                            # We just need to break the cycle. We can pick a random edge in the graph 
                            # and try to swap it.
                            return None 
        return None

    # Local Search
    best_adj = None
    
    # Try to find a graph with n=320
    attempts = 0
    while time.time() < deadline:
        if attempts % 100 == 0:
            print(f"Attempt {attempts}") # Debug
        attempts += 1
        
        adj = generate_random_k_regular(n_target, k)
        
        # Quick check: if girth is already >= g, we are done
        if check_girth(adj, n_target, g):
            best_adj = adj
            break
        
        # Local search: try to improve girth
        # This is a simple heuristic: try random edge swaps
        # We don't track girth precisely, we just check if it's valid at the end.
        # To speed up, we can just do random swaps and check validity periodically.
        
        for _ in range(100): # 100 swaps per attempt
            if time.time() > deadline:
                break
            
            # Pick two random edges
            # Flatten edges
            edges = []
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            if len(edges) < 2:
                continue
                
            i = random.randint(0, len(edges)-1)
            j = random.randint(0, len(edges)-1)
            if i == j:
                continue
            
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            
            # Check if edges are disjoint
            if len({u1, v1, u2, v2}) < 4:
                continue
            
            # Try swap: (u1, u2) and (v1, v2)
            # Check if new edges exist
            if u2 in adj[u1] or v2 in adj[v1]:
                continue
            
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].add(u2)
            adj[u2].add(u1)
            adj[v1].add(v2)
            adj[v2].add(v1)
            
            # Check if we improved or maintained validity
            # We don't want to accept swaps that create short cycles if we were valid before.
            # But we start invalid.
            # Let's just keep swapping.
            
        # Check if this configuration is valid
        if check_girth(adj, n_target, g):
            best_adj = adj
            break

    if best_adj:
        edges = []
        for u in range(n_target):
            for v in best_adj[u]:
                if u < v:
                    edges.append((u, v))
        return n_target, edges

    # Fallback to LUW construction if local search fails
    # Re-implement the LUW logic from the original prompt to ensure it works as a fallback
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
