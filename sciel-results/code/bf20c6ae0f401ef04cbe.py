import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    # Moore bound for k=4, g=10
    # n >= 2 * sum_{i=0}^{4} (k-1)^i = 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242
    # So n must be at least 242.
    # The cage is 320.
    
    def generate_random_k_regular(n, k):
        """
        Generate a random simple k-regular graph on n vertices.
        Returns adjacency list or None if failed.
        """
        if n * k < 2:
            return None
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
            
        # Configuration model
        # Create 2n stubs? No, n*k stubs.
        # Each vertex i has k stubs.
        # Pair stubs randomly.
        
        # To avoid self-loops and multi-edges efficiently, we can use a rejection sampling approach
        # or a more complex pairing algorithm. For n=320, rejection might be okay if we retry.
        # However, simple random pairing has a low probability of being simple for large n?
        # Actually, for fixed k, as n->inf, the probability of being simple goes to 1.
        # For n=320, k=4, it should be decent.
        
        max_attempts = 50
        for _ in range(max_attempts):
            # Create list of vertex ids, each repeated k times
            vertices = [i for i in range(n) for _ in range(k)]
            random.shuffle(vertices)
            
            # Pair up
            adj = [[] for _ in range(n)]
            edges = set()
            valid = True
            
            for i in range(0, len(vertices), 2):
                u = vertices[i]
                v = vertices[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                continue
            
            # Check connectivity
            if not is_connected(n, adj):
                continue
                
            return adj, list(edges)
            
        return None

    def is_connected(n, adj):
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        return count == n

    def find_short_cycle(n, adj, max_len):
        """
        Find a cycle of length <= max_len.
        Returns the list of vertices in the cycle, or None.
        """
        # BFS from each vertex up to depth max_len/2
        # To be efficient, we can just check if any cycle < max_len exists
        # We want the shortest one to break it.
        
        # For small girth targets, BFS from each node is okay if n is small.
        # n=320, max_len=9. Depth 4.
        
        for start in range(n):
            # dist, parent
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # If v is not parent, and v is not start (unless we are closing a cycle)
                        # We found a cycle if dist[u] + 1 + dist[v] < max_len
                        # But we need to ensure we don't count the tree edge back.
                        if v != parent[u]:
                            cycle_len = dist[u] + 1 + dist[v]
                            if cycle_len < max_len:
                                # Reconstruct cycle
                                cycle = []
                                # Path from u to LCA?
                                # Actually, BFS tree gives a path from start to u and start to v.
                                # The cycle is path(start->u) + edge(u,v) + path(v->start) reversed.
                                # This is complex to reconstruct exactly from just dist/parent if they don't meet at LCA.
                                # However, for the purpose of breaking, we just need to know a cycle exists.
                                # Let's return a dummy indicator or the edge (u,v) that closes it.
                                return (u, v, cycle_len)
                                
        return None

    def break_cycle(n, adj, edges_set, u, v, cycle_len):
        """
        Attempt to break a cycle by swapping edges.
        This is a heuristic. We pick an edge in the cycle and an edge outside, and swap.
        Or, simpler: Find two edges (a,b) and (c,d) such that swapping to (a,c) and (b,d) breaks the cycle.
        """
        # This is complex to implement robustly in short time.
        # Alternative: Just regenerate the graph.
        # If we find a short cycle, discard and try next random graph.
        # For n=320, the fraction of 4-regular graphs with girth >= 10 might be small, 
        # but we can try many.
        pass

    # Search for the smallest n
    # We know n >= 242.
    # Let's try n from 242 upwards.
    # But 320 is the known cage.
    # We will try to find a valid graph for n=320 first, as it's the target.
    # If we find one, we return it.
    # If we fail, we try 322, 324, etc.
    # Or we can try to improve a graph of size 320.
    
    # Given the difficulty of generating a specific high-girth graph randomly,
    # and the time limit, let's try a few n values.
    
    # The LUW graph from the seed is a safe fallback.
    # Let's keep the LUW generator as a fallback.
    
    def luw_graph():
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

    # Try to find a graph with n vertices
    # We will try n = 320, 322, 324, 326...
    # And also check if we can find one for smaller n (unlikely but possible if cage is smaller? No, 320 is known smallest).
    # So we aim for 320.
    
    best_n = float('inf')
    best_edges = None
    
    # Try n=320 first
    for n in range(320, 340, 2): # k-regular implies n*k even. k=4, so n can be any? 4n is always even.
        if time.time() > deadline - 5:
            break
            
        # Try to generate a random 4-regular graph of size n
        # and check girth
        success = False
        trials = 200 # Limit trials per n
        
        for _ in range(trials):
            if time.time() > deadline - 2:
                break
                
            res = generate_random_k_regular(n, k)
            if res is None:
                continue
                
            adj, edges = res
            
            # Check girth
            # If girth >= g, we found one!
            # We need to verify girth properly.
            # find_short_cycle returns None if no cycle < g
            if find_short_cycle(n, adj, g) is None:
                best_n = n
                best_edges = edges
                success = True
                break
                
        if success:
            break
            
    if best_edges is None:
        # Fallback to LUW
        best_n, best_edges = luw_graph()
        
    return best_n, best_edges
