import time
import random
import numpy as np

def construct(k, g):
    # 1. Fallback: Lazebnik-Ustimenko-Woldar graph
    # This is the seed program's logic, optimized slightly for speed.
    def get_luw_graph(k, g):
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

    n_luw, edges_luw = get_luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # 2. Search for smaller graphs
    # Known lower bound is ~243, known upper bound 320.
    # We try to find a valid graph with n < n_luw.
    # We use a randomized local search for each candidate n.
    
    deadline = time.time() + 110  # 10s safety margin
    
    def calculate_girth(n, adj):
        # Non-backtracking BFS from every node to find shortest cycle
        min_girth = float('inf')
        # Optimization: Only need to check if girth >= g. 
        # We can stop BFS at depth g/2.
        max_depth = g // 2
        for start in range(n):
            # BFS state: (node, prev, depth)
            queue = [(start, -1, 0)]
            visited = {} # (node, prev) -> depth
            while queue:
                node, prev, depth = queue.pop(0)
                key = (node, prev)
                if key in visited:
                    continue
                visited[key] = depth
                
                if depth == max_depth:
                    # If we reached max_depth, no cycle shorter than 2*max_depth+1?
                    # Actually, a cycle of length L means a path of length L-1 back to start?
                    # Non-backtracking: shortest cycle length is 2 * (min depth where neighbor of start is reached) ?
                    # Simpler: If we haven't found a cycle < g, we are good.
                    pass
                
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start and depth + 1 < g:
                        return depth + 1
                    if depth + 1 < max_depth:
                        queue.append((nb, node, depth + 1))
        return min_girth

    def is_valid_graph(n, edges, k, g):
        # Check degree
        if len(edges) != n * k // 2:
            return False
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v: return False
            if v in adj[u]: return False # Duplicate
            adj[u].add(v)
            adj[v].add(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth
        # Fast girth check using BFS up to depth g//2
        for start in range(n):
            # (node, prev, depth)
            queue = [(start, -1, 0)]
            visited = set()
            found_cycle = False
            while queue:
                node, prev, depth = queue.pop(0)
                key = (node, prev)
                if key in visited:
                    continue
                visited.add(key)
                
                if depth == g // 2:
                    continue # No need to go deeper for girth check < g
                
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start:
                        if depth + 1 < g:
                            return False # Cycle found
                        else:
                            # Cycle length >= g is ok, but we need to ensure NO cycle < g
                            # If depth+1 >= g, it's not a violation of "no cycle shorter than g"
                            pass
                    else:
                        if depth + 1 < g // 2:
                            queue.append((nb, node, depth + 1))
            # If we didn't find a short cycle from this start, continue
        return True

    def generate_random_k_regular(n, k):
        # Random 4-regular graph generation
        # Simple approach: Random permutation of half-edges
        # Half-edges: n*k / 2 edges
        # List of nodes with required degree
        # This is tricky to ensure simple graph.
        # Use configuration model:
        # List of 2n stubs? No, n*k stubs.
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((u, v))
        if not valid or len(edges) != n * k // 2:
            return None
        return list(edges)

    # Search loop
    # We try n values from lower bound up to n_luw
    # Moore bound for k=4, g=10 is 242.
    # Start from 243? No, start from 320 (known smallest) downwards? 
    # Or just try a few small n values.
    
    # Since 320 is the known best, we try to verify if we can find it or something close.
    # However, random search for n=320 might fail.
    # We try a few sizes.
    
    candidate_ns = []
    # If n_luw is large, we definitely want to beat it.
    # Try sizes near the known optimum.
    if n_luw > 400:
        candidate_ns = [320, 330, 340, 350, 360]
        # Also try slightly smaller if possible, but 320 is the cage.
    else:
        candidate_ns = [n_luw - 10, n_luw - 20]

    # To increase chance, we run search for the smallest n first.
    # If we find a valid graph, we return it immediately (it's the best).
    # If not, we keep trying larger n (which are easier to satisfy girth? No, larger n makes girth easier to maintain? 
    # Actually, for fixed k, larger n allows for larger girth more easily. 
    # So finding a girth-10 graph at n=1000 is EASIER than at n=320.
    # So we should search LARGE n first? 
    # But we want to MINIMIZE n.
    # So we want the SMALLEST n that works.
    # If we search n=1000, we find a graph, score is bad (n=1000).
    # If we search n=320 and fail, we return n=2048 (LUW).
    # So we should search n=320, 330, ... in increasing order? 
    # But n=320 is hardest.
    
    # Strategy: 
    # 1. Try n=320 with limited time.
    # 2. If fail, try n=400?
    # 3. If fail, return LUW.
    
    # Given time limits, let's try a few iterations for n=320.
    
    for n in sorted(candidate_ns):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph of size n
        found = False
        max_iters = 200  # Adjust based on time
        
        # We can also try to "repair" a LUW graph? No, different n.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            edges = generate_random_k_regular(n, k)
            if edges is None:
                continue
                
            if is_valid_graph(n, edges, k, g):
                best_n = n
                best_edges = edges
                found = True
                break
            else:
                # Local search: Swap edges to increase girth
                # This is complex, so we just rely on random restarts for now
                pass
        
        if found:
            return best_n, best_edges

    return best_n, best_edges
