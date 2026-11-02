import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Moore bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2 - 1
        moore = k * sum((k - 1) ** i for i in range(r))
        
    # Ensure moore is at least k+1 or similar
    if moore < k + 1:
        moore = k + 1

    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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

    def check_girth_fast(n, a, b, target_g):
        # Check if circulant C_n(a, b, n/2) has girth >= target_g
        # We check if balls of radius (target_g-1)//2 are trees
        # For g=9, radius 4.
        radius = (target_g - 1) // 2
        
        # Precompute neighbors for speed? No, n is small.
        # But we do this for many (a,b).
        # We can just run BFS.
        
        for v in range(n):
            # BFS to depth 'radius'
            dist = [-1] * n
            dist[v] = 0
            q = deque([v])
            node_count = 0
            edge_count = 0
            
            while q:
                u = q.popleft()
                node_count += 1
                if dist[u] >= radius:
                    continue
                # Neighbors of u in C_n(a, b, n/2)
                # u + a, u - a, u + b, u - b, u + n/2
                for delta in (a, -a, b, -b, n // 2):
                    w = (u + delta) % n
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        q.append(w)
                    else:
                        # If w is already visited, it's an edge in the ball.
                        # We count each edge once?
                        # In a tree, edges = nodes - 1.
                        # We need to count edges in the induced subgraph.
                        # This BFS approach doesn't easily count edges without double counting.
                        pass
            
            # This BFS is not counting edges correctly for the "tree" check.
            # Alternative: Just check if we find a cycle.
            # A cycle exists if we visit a node that is already visited and is not the parent.
            # But we need to know the length.
            # If we find a back edge, the cycle length is dist[u] + dist[w] + 1? No.
            # Let's just use the property: If the ball of radius 4 is a tree, girth >= 9.
            # To check if it's a tree, we can just count nodes and edges.
            # Let's collect the nodes in the ball.
            
            ball_nodes = set()
            dist2 = [-1] * n
            dist2[v] = 0
            q2 = deque([v])
            while q2:
                u = q2.popleft()
                ball_nodes.add(u)
                if dist2[u] >= radius:
                    continue
                for delta in (a, -a, b, -b, n // 2):
                    w = (u + delta) % n
                    if dist2[w] == -1:
                        dist2[w] = dist2[u] + 1
                        q2.append(w)
            
            # Count edges in induced subgraph on ball_nodes
            # Edges are (u, w) where w in ball_nodes and w in neighbors(u)
            # To avoid double counting, only count if u < w.
            edges_in_ball = 0
            for u in ball_nodes:
                for delta in (a, b, n // 2): # Only positive deltas to avoid double counting?
                    w = (u + delta) % n
                    if w in ball_nodes and u < w:
                        edges_in_ball += 1
                # Also check u - a, u - b?
                # If we iterate all nodes and all 5 neighbors, and count u<w, it works.
            
            # Simpler:
            edges_in_ball = 0
            for u in ball_nodes:
                for delta in (a, -a, b, -b, n // 2):
                    w = (u + delta) % n
                    if w in ball_nodes and u < w:
                        edges_in_ball += 1
                        
            if edges_in_ball >= len(ball_nodes):
                return False # Cycle found
            
        return True

    # Search for Circulant
    # n must be even for 5-regular circulant with n/2 generator
    n_start = moore
    if n_start % 2 != 0:
        n_start += 1
        
    # Try a few n values
    # We want to find the smallest n.
    # We will iterate n from n_start to n_start + 200 (or until deadline)
    # For each n, try random (a,b)
    
    max_n_to_try = min(1200, n_start + 300)
    
    for n in range(n_start, max_n_to_try + 1):
        if time.time() > deadline - 5:
            break
            
        # Random search for a, b
        found = False
        # Try 50 random pairs
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            a = random.randint(1, n // 2 - 1)
            b = random.randint(1, n // 2 - 1)
            if a == b:
                continue
            # Check distinctness of generators
            # Generators: a, -a, b, -b, n/2
            # We need {a, n-a, b, n-b, n/2} to be distinct.
            # This implies a != b, a != n-b, a != n/2, etc.
            # Since 1 <= a,b < n/2, a != n/2 and b != n/2 are satisfied.
            # a != b is checked.
            # a != n-b => a+b != n.
            if a + b == n:
                continue
            # a != -b is same as a+b != n.
            # So we just need a != b and a+b != n.
            
            if check_girth_fast(n, a, b, g):
                # Construct edges
                edges = []
                for i in range(n):
                    for delta in (a, b, n // 2):
                        j = (i + delta) % n
                        if i < j:
                            edges.append((i, j))
                # Verify simplicity and regularity (should be guaranteed)
                return n, edges
                
    # Fallback to LUW
    return luw_construct(k, g)
