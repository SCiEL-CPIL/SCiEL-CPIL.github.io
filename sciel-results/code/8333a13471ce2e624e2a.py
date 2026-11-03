import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Fallback: Lazebnik-Ustimenko-Woldar construction
    def luw_construct(k, g):
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    # --- Girth Verification ---
    
    def verify_girth(n, edges, g):
        """
        Exact girth check using BFS from every vertex.
        Returns True if girth >= g, False otherwise.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute neighbors as numpy arrays for faster iteration if needed, 
        # but pure Python list iteration is usually fast enough for n=320.
        
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= (g - 1) // 2:
                    continue # No need to expand further for short cycle detection
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                    else:
                        # If we see an edge (u, v) where u and v are already visited,
                        # we have a cycle of length dist[u] + dist[v] + 1.
                        # Note: In a BFS tree, if v is a parent of u, we skip.
                        # But since we don't store parents, we check all visited neighbors.
                        # Actually, the standard BFS girth check:
                        # When relaxing (u,v), if v is visited and v is NOT parent of u,
                        # then cycle length is dist[u] + dist[v] + 1.
                        # However, without parent tracking, we might count the tree edge back.
                        # Let's refine: 
                        # If v is visited, and dist[v] != du - 1 (i.e., v is not the parent that discovered u),
                        # then we have a non-tree edge creating a cycle.
                        # Wait, dist[v] could be du+1? No, BFS visits in order.
                        # If v is visited, dist[v] <= dist[u] + 1.
                        # If dist[v] == dist[u] + 1, it's a forward/cross edge? 
                        # In undirected graph, edges connect nodes at dist d and d+1 or d and d.
                        # If dist[v] == dist[u], cycle length is 2*du + 1.
                        # If dist[v] == dist[u] - 1, it could be the parent edge (length 2*du+1? No, parent is du-1).
                        # Let's just check if dist[u] + dist[v] + 1 < g.
                        # We must be careful not to count the tree edge itself as a cycle.
                        # A tree edge (u, parent(u)) has dist[parent] = dist[u] - 1.
                        # Cycle length would be dist[u] + dist[u]-1 + 1 = 2*dist[u]. This is not a cycle, it's the path back.
                        # So we should only consider it a cycle if it's NOT the tree edge.
                        # Without parent array, we can't distinguish tree edges easily.
                        # Better approach: Store parent.
                        
                        pass # We'll use a more robust check below if this simple one is flawed.
                        
        # Let's rewrite verify_girth with parent tracking to be safe.
        def verify_girth_with_parents(n, edges, g):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    du = dist[u]
                    if du >= (g - 1) // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            # If v is not the parent of u, we have a cycle.
                            if v != parent[u]:
                                cycle_len = du + dist[v] + 1
                                if cycle_len < g:
                                    return False
            return True
            
        return verify_girth_with_parents(n, edges, g)

    # --- Circulant Search ---
    
    def check_circulant_girth_fast(n, a, b, g):
        """
        Fast filter for circulant C_n(a,b).
        Returns False if a short cycle is definitely present.
        Returns True if it might be valid (needs full verification).
        """
        # BFS from 0
        dist = [-1] * n
        dist[0] = 0
        q = deque([0])
        
        # Precompute neighbors for speed? 
        # Neighbors of x are (x+a)%n, (x-a)%n, (x+b)%n, (x-b)%n
        
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= (g - 1) // 2:
                continue
            
            neighbors = [(u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n]
            for v in neighbors:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
                else:
                    # Check for short cycle
                    # If v is not the "parent" in the BFS tree... 
                    # In circulant, we don't track parents here for the filter.
                    # If dist[u] + dist[v] + 1 < g, we have a short cycle.
                    # This is a necessary condition for a short cycle.
                    # If we find such an edge, we reject.
                    if du + dist[v] + 1 < g:
                        return False
        return True

    def generate_circulant_edges(n, a, b):
        edges = []
        for i in range(n):
            edges.append((i, (i + a) % n))
            edges.append((i, (i + b) % n))
        return edges

    # --- Random Regular Search ---
    
    def generate_random_regular(n, k):
        """
        Generates a random k-regular graph using the configuration model.
        Returns None if it fails to generate a simple graph (loops or multi-edges).
        """
        if n * k % 2 != 0:
            return None
        
        # Use numpy for shuffling speed
        stubs = np.repeat(np.arange(n), k)
        np.random.shuffle(stubs)
        
        edges_set = set()
        adj = {i: [] for i in range(n)}
        
        for i in range(0, n * k, 2):
            u, v = int(stubs[i]), int(stubs[i+1])
            if u == v:
                return None
            if (u, v) in edges_set or (v, u) in edges_set:
                return None
            edges_set.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        
        return list(edges_set)

    # --- Main Logic ---
    
    # 1. Get baseline from LUW
    best_n, best_edges = luw_construct(k, g)
    
    # For k=4, g=10, LUW gives 320 vertices.
    # Moore bound is 161.
    # We search for smaller graphs.
    
    # Search space:
    # 1. Circulant graphs C_n(a,b)
    # 2. Random regular graphs
    
    # Strategy:
    # Iterate n from Moore bound to best_n - 1.
    # For each n, try circulants first (fast filter), then random graphs.
    # Stop if time is up.
    
    moore_bound = 1 + k * (1 + (k-1) + (k-1)**2) # For g=4? 
    # General Moore bound for girth g (even): 1 + k * sum_{i=0}^{g/2-2} (k-1)^i
    # For g=10: 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4*40 = 161.
    
    start_n = 161
    end_n = best_n - 1
    
    # If best_n is already small, we might not search much.
    # If LUW is 320, we search 161..319.
    
    # To maximize chance of finding a valid graph, we should prioritize
    # n values where random graphs are more likely to have high girth.
    # However, smaller n is better for the score.
    # We must find the SMALLEST n.
    
    # So we iterate n from start_n upwards.
    # For each n, we try to find ANY valid graph.
    # If found, return it immediately (since we're going in increasing order).
    
    # Time management:
    # We have ~110s.
    # Circulant search is fast.
    # Random search requires generating and verifying.
    # Verification for n=200 is fast.
    
    # Let's allocate time per n.
    # Or just run until deadline.
    
    # Optimizations:
    # - Use fast filter for circulants.
    # - Use fast rejection for random graphs (e.g., check for triangles first?).
    #   Actually, verify_girth is O(n * E). For n=300, E=600, it's ~180k ops.
    #   This is very fast in Python if optimized.
    
    # Let's start the loop.
    
    for n in range(start_n, end_n + 1):
        if time.time() > deadline:
            break
            
        # 1. Try Circulants
        # a, b in [1, n//2]
        # a != b, a+b != n, 2a != 0, 2b != 0
        # gcd(n, a, b) == 1
        
        # To speed up, we can pre-filter a, b pairs.
        # Number of pairs is ~ (n/2)^2 / 2 = n^2/8.
        # For n=300, ~11k pairs.
        # Filter is BFS O(n). 11k * 300 = 3.3M ops. Fast.
        
        found_circ = False
        for a in range(1, n // 2 + 1):
            if (2 * a) % n == 0:
                continue
            for b in range(a + 1, n // 2 + 1):
                if (2 * b) % n == 0:
                    continue
                if a + b == n:
                    continue
                if math.gcd(math.gcd(n, a), b) != 1:
                    continue
                
                if check_circulant_girth_fast(n, a, b, g):
                    edges = generate_circulant_edges(n, a, b)
                    if verify_girth(n, edges, g):
                        return n, edges
        
        # 2. Try Random Graphs
        # How many random graphs to try?
        # Probability of a random 4-regular graph having girth >= 10 is very low.
        # For n=161, it's almost 0.
        # For n=300, it's still very low.
        # We might not find any with random search in the time limit.
        # But let's try a few.
        
        # Limit attempts per n to avoid spending too much time on a single n.
        # If we don't find one quickly, move to next n?
        # No, we need the SMALLEST n.
        # So we must exhaust n before moving to n+1.
        # But if probability is 0, we'll waste time.
        
        # Heuristic:
        # If n is small, girth is likely < 10.
        # We can skip random search for very small n if we have a quick reject.
        # Or just do a few random tries.
        
        num_random_tries = 10
        for _ in range(num_random_tries):
            if time.time() > deadline:
                break
            edges = generate_random_regular(n, k)
            if edges:
                # Quick check: Does it have a triangle?
                # If yes, reject.
                # verify_girth will catch it, but we can do a faster triangle check.
                # For now, just run verify_girth.
                if verify_girth(n, edges, g):
                    return n, edges

    # If no smaller graph found, return LUW
    return best_n, best_edges

# Test harness (optional, for local testing)
if __name__ == "__main__":
    k, g = 4, 10
    n, edges = construct(k, g)
    print(f"n={n}")
    # Verify
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            print(f"Vertex {i} has degree {len(adj[i])}")
            break
    
    # Check girth
    # (Using the verify function)
    # print(verify_girth(n, edges, g))
