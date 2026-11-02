"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Fallback: LUW graph (2048 vertices)
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

    # Heuristic Search for smaller graph
    def try_find_small_graph(k, g, max_n_attempts=50):
        # Target n based on Moore bound + small overhead
        # Moore bound for k=4, g=10 is 242.
        # Known cage is 320.
        target_n = 320
        
        def generate_random_regular_graph(n, k):
            # Generate a random k-regular graph using configuration model + correction
            # Simple method: Start with random permutation of half-edges
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            random.shuffle(half_edges)
            
            adj = [[] for _ in range(n)]
            used = [False] * len(half_edges)
            
            # Greedy pairing to avoid self-loops and multi-edges
            # This is a simple heuristic, might fail to produce simple graph
            # We will use a retry loop
            for attempt in range(100):
                random.shuffle(half_edges)
                adj = [[] for _ in range(n)]
                valid = True
                edges = set()
                for i in range(0, len(half_edges), 2):
                    u = half_edges[i]
                    v = half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    if (min(u,v), max(u,v)) in edges:
                        valid = False
                        break
                    adj[u].append(v)
                    adj[v].append(u)
                    edges.add((min(u,v), max(u,v)))
                if valid:
                    return adj, list(edges)
            return None, None

        def get_girth(adj, n, max_depth):
            # Returns shortest cycle length found, or infinity if none <= max_depth
            # BFS from each node
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                queue = deque()
                queue.append(start)
                dist[start] = 0
                while queue:
                    u = queue.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            return cycle_len
            return float('inf')

        def swap_edges(adj, edges, n, k):
            # Find two edges (u,v) and (x,y) that can be swapped to (u,x) and (v,y) or (u,y) and (v,x)
            # such that no self-loop or multi-edge is created.
            # We pick a random edge, then a random other edge.
            if len(edges) < 2:
                return False
            idx1 = random.randint(0, len(edges)-1)
            u1, v1 = edges[idx1]
            
            # Try a few random swaps
            for _ in range(10):
                idx2 = random.randint(0, len(edges)-1)
                if idx1 == idx2:
                    continue
                u2, v2 = edges[idx2]
                
                # Ensure disjoint
                if u1 in (u2, v2) or v1 in (u2, v2):
                    continue
                
                # Option 1: (u1, u2) and (v1, v2)
                if u1 != u2 and v1 != v2:
                    # Check if edges exist
                    e1 = (min(u1, u2), max(u1, u2))
                    e2 = (min(v1, v2), max(v1, v2))
                    if e1 not in adj[u1] and e2 not in adj[v1]: # simplified check
                         # Actually need to check set
                         pass
                
                # This implementation is getting complex and slow.
                # Let's use a simpler "cycle breaking" strategy.
                # If we find a cycle of length L < g, we can try to "cut" it.
                # But swapping edges in a regular graph to increase girth is non-trivial.
                return False
            return False

        # The swap logic above is weak. 
        # Instead, we will rely on the fact that for n=320, random graphs might occasionally have high girth?
        # Probability is low.
        
        # Better: Use the LUW graph but check if we can reduce d?
        # For k=4, g=10, LUW d=5 gives 2048.
        # If we use d=4, girth is ~8.
        
        # Given the difficulty, I will return the LUW graph.
        # However, I will try one specific known smaller graph if I can construct it.
        # The (3,9) cage is 504.
        # The (4,10) cage is 320.
        
        # I will stick to LUW to ensure validity.
        return None

    # Attempt search
    if k == 4 and g == 10:
        # Try to find a 320 vertex graph
        # If successful, return it.
        # Since the heuristic is weak, we likely won't find it in time.
        # We return LUW.
        pass

    return luw_construct(k, g)

# EVOLVE-BLOCK-END
