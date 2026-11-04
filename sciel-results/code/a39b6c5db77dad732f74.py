import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for final check/return
    
    # Fallback: LUW construction from seed
    def luw_construction():
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

    # Girth Checker
    def get_girth(adj):
        n = len(adj)
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            if min_girth < g:
                return min_girth
        return min_girth if min_girth != float('inf') else 2 * n

    # Random K-Regular Graph Generator (Corrected)
    def gen_random_k_reg(n, k):
        # Use a backtracking or repair method
        # For simplicity and speed, use a configuration model with repair
        # But configuration model is prone to multi-edges.
        # Let's try a simple random generation and repair.
        
        # Initialize adjacency list
        adj = [[] for _ in range(n)]
        
        # Create stubs
        stubs = [i for i in range(n) for _ in range(k)]
        
        # Shuffle
        random.shuffle(stubs)
        
        # Pair stubs
        # We need to ensure no loops and no multi-edges.
        # This is tricky with simple shuffling.
        # Alternative: Generate a random permutation and pair (2i, 2i+1).
        # Then repair.
        
        # Let's try a different approach: 
        # Start with an empty graph and add edges randomly, ensuring validity.
        # This is slow for large n.
        
        # Better: Use a known algorithm for random regular graphs.
        # Given the time constraint, I will use a heuristic:
        # Generate a random graph, check validity, if invalid, restart.
        # This is inefficient but might work for small n.
        
        for _ in range(1000):
            adj = [[] for _ in range(n)]
            valid = True
            # Create list of all possible edges? No, too many.
            
            # Use stubs again, but with a repair step.
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid and all(len(a) == k for a in adj):
                return adj
        
        return None

    # Optimization Loop
    def optimize(n, k, g, time_limit):
        adj = gen_random_k_reg(n, k)
        if not adj:
            return None
            
        best_adj = None
        best_penalty = float('inf')
        
        # Penalty: count short cycles
        def calc_penalty(adj):
            pen = 0
            # Triangles
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        common = set(adj[u]) & set(adj[v])
                        pen += len(common)
            # 4-cycles
            # Approximation
            return pen
            
        current_penalty = calc_penalty(adj)
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # 2-Switch
            u = random.randint(0, n-1)
            v = random.choice(adj[u])
            w = random.randint(0, n-1)
            if w == u or w == v:
                continue
            x = random.choice(adj[w])
            if x == u or x == v:
                continue
            
            if w in adj[u] or x in adj[v] or w == x:
                continue
                
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            new_penalty = calc_penalty(adj)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty == 0:
                     girth = get_girth(adj)
                     if girth >= g:
                         edges = []
                         for i in range(n):
                             for j in adj[i]:
                                 if i < j:
                                     edges.append((i, j))
                         return (n, edges)
            else:
                # Reject
                adj[u].remove(w)
                adj[w].remove(u)
                adj[v].remove(x)
                adj[x].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[w].append(x)
                adj[x].append(w)
                
        return None

    # Try several N values
    candidates = [600, 800, 1000, 1200]
    
    for n in candidates:
        if time.time() > deadline - 10:
            break
        time_budget = min(30, deadline - time.time() - 5)
        if time_budget < 5:
            break
            
        result = optimize(n, k, g, time_budget)
        if result:
            return result

    # Fallback
    return luw_construction()
