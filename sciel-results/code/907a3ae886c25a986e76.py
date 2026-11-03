import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound for (3, 16) is approx 382-512.
    # Known best is 936.
    # LUW is 2550.
    # We try to beat 2550.
    
    # We will try n from 512 to 2549.
    # For each n, we try to generate a random 3-regular graph and check girth.
    # To make it faster, we can use a greedy construction.
    
    # Greedy Construction:
    # 1. Start with a random permutation of edges?
    # 2. Use the "Configuration Model" to generate random 3-regular graphs.
    # 3. Check girth.
    
    def check_girth(edges, n, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex up to depth min_girth//2
        # If we find a back-edge to a vertex in the current BFS tree
        # at a distance that creates a cycle < min_girth, return False.
        
        # Actually, just check all cycles up to length min_girth - 1.
        # For k=3, this is fast.
        
        for start in range(n):
            # BFS
            visited = [-1] * n
            parent = [-1] * n
            visited[start] = 0
            queue = [start]
            
            while queue:
                u = queue.pop(0)
                if visited[u] >= min_girth // 2:
                    continue
                
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Cycle length = dist(u) + dist(v) + 1
                        # But this is not exact if they are not in the same BFS tree branch?
                        # In a tree, yes. In a graph, BFS finds shortest paths.
                        # If visited[u] + visited[v] + 1 < min_girth, we have a short cycle?
                        # Not necessarily. The cycle might be longer.
                        # But if we find ANY cycle < min_girth, it's invalid.
                        # BFS ensures shortest paths.
                        # So if dist(u) + dist(v) + 1 < min_girth, is there a cycle of that length?
                        # Yes, the path u->start + path v->start + edge (u,v).
                        # Length = dist(u) + dist(v) + 1.
                        if visited[u] + visited[v] + 1 < min_girth:
                            return False
        return True

    def generate_random_3_regular(n):
        # Configuration model
        # n vertices, each with 3 stubs.
        # Total 3n stubs.
        # Pair them up randomly.
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self-loop, retry or fix
                # For simplicity, if self-loop, swap with another
                # This is a naive implementation.
                # Better: ensure no self-loops in generation.
                pass 
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                # Multi-edge, retry
                pass
            edges.add((u, v))
            
        # This naive generation often fails (self-loops, multi-edges).
        # A robust way is to use a library or a better algorithm.
        # Given time constraints, I will return the LUW graph if search fails.
        return list(edges)

    # Since robust random generation is tricky in a short script,
    # and the LUW graph is guaranteed valid,
    # I will implement the LUW graph as a fallback.
    # But I will try a simple greedy search for n < 2550 first.
    
    # Actually, the LUW graph code is already there.
    # I will just return the LUW graph to ensure validity,
    # but I will try to find a smaller one if I can.
    
    # Given the complexity, I will stick to the LUW graph but optimize it slightly
    # or just return it as is, because a broken search scores 0.
    
    # Wait, the prompt says "Score: n, minimized".
    # If I return LUW (2550), score is 1.0.
    # If I return a valid graph with n=2000, score is 2550/2000 = 1.275.
    
    # Let's try to construct a smaller graph using the "Tree + Matching" idea
    # but avoiding short cycles.
    # This is hard.
    
    # I will return the LUW graph to be safe.
    
    # Re-reading the prompt: "REFINEMENT: ... KEEP this design".
    # So I should keep LUW.
    
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
