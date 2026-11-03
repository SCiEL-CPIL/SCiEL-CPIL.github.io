"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Target n based on known cage sizes or Moore bound
    # For k=4, g=10, known min is 320.
    # We try to find a graph with n=320. If it fails, we try slightly larger n.
    # If all fail, we use LUW.
    
    if k != 4 or g != 10:
        # If parameters differ, just use LUW as it's generic
        return luw_construct(k, g)
    
    target_n = 320
    
    def generate_random_4_reg(n):
        """Generates a random 4-regular graph on n vertices."""
        # Create list of stubs
        stubs = [i for i in range(n) for _ in range(4)]
        random.shuffle(stubs)
        edges = set()
        adj = [set() for _ in range(n)]
        
        # Pair stubs
        # This simple pairing might create multi-edges or loops.
        # We need to ensure simple graph.
        # A better way: Configuration model with rejection or repair.
        
        # Let's use a simple repair loop
        max_attempts = 1000
        for _ in range(max_attempts):
            stubs = [i for i in range(n) for _ in range(4)]
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                return adj
        
        # If failed, fallback to a deterministic construction (e.g., circulant)
        # Circulant graph C_n(1, n-1, 2, n-2) is 4-regular if n is large enough and no short cycles.
        # For n=320, C_320(1, 319, 2, 318) has girth?
        # Cycle 1-2-3-4-1? No, edges are i~i+1, i~i-1, i~i+2, i~i-2.
        # 1-2-4-3-1 is a 4-cycle. Girth 4.
        # So circulant is bad for high girth.
        
        # Return None if failed
        return None

    def get_edges_from_adj(adj):
        edges = []
        n = len(adj)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    def check_girth(adj, target_girth):
        """Returns (girth, cycle) where cycle is a list of vertices if girth < target_girth."""
        n = len(adj)
        min_girth = float('inf')
        min_cycle = None
        
        for start in range(n):
            # BFS
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
                    elif v != parent[u] and dist[u] + 1 > dist[v]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            # Reconstruct cycle
                            path1 = []
                            curr = u
                            while curr != -1:
                                path1.append(curr)
                                curr = parent[curr]
                            path2 = []
                            curr = v
                            while curr != -1:
                                path2.append(curr)
                                curr = parent[curr]
                            # Find LCA
                            # Simplified: just return length and a dummy cycle for breaking
                            # We need the actual edges to break
                            min_cycle = (u, v, dist[u], dist[v])
                            # Early exit if we found a very short cycle
                            if min_girth <= 4:
                                return min_girth, min_cycle
            if min_girth <= 4:
                break
        
        return min_girth, min_cycle

    def break_cycle(adj, u, v):
        """
        Breaks the cycle involving u and v by rewiring an edge incident to u or v.
        Returns True if successful rewire.
        """
        n = len(adj)
        # Pick a random neighbor of u to rewire
        neighbors = list(adj[u])
        if not neighbors:
            return False
        old_v = random.choice(neighbors)
        
        # Find a new vertex w to connect u to
        # w must not be u, not be in adj[u], and ideally not create a short cycle
        # Simple heuristic: pick random w not in adj[u] and w != u
        candidates = [i for i in range(n) if i != u and i not in adj[u]]
        if not candidates:
            return False
        new_w = random.choice(candidates)
        
        # Perform rewire
        adj[u].remove(old_v)
        adj[old_v].remove(u)
        adj[u].add(new_w)
        adj[new_w].add(u)
        return True

    # Main search loop
    best_n, best_edges = None, None
    
    # Try to find a valid graph with n=320
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        adj = generate_random_4_reg(target_n)
        if adj is None:
            continue
            
        # Iterative improvement
        for step in range(1000):
            if time.time() > deadline - 5:
                break
                
            girth, cycle_info = check_girth(adj, g)
            if girth >= g:
                best_n = target_n
                best_edges = get_edges_from_adj(adj)
                return best_n, best_edges
                
            if cycle_info is None:
                break
                
            # Break the cycle
            # cycle_info is (u, v, du, dv)
            # We just pick u and rewire one of its edges
            if not break_cycle(adj, cycle_info[0], cycle_info[1]):
                break
        
        # If we exited the step loop without success, try next attempt
        # Optionally, check if the final graph is valid (might have improved)
        girth, _ = check_girth(adj, g)
        if girth >= g:
            best_n = target_n
            best_edges = get_edges_from_adj(adj)
            return best_n, best_edges

    # If search failed, use LUW fallback
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
