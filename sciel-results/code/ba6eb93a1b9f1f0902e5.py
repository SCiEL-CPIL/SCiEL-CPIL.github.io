"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on a non-abelian group using involutory generators.
    If that fails within time, fall back to the Lazebnik-Ustimenko-Woldar construction.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as reliable fallback
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
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
    
    # Check if a graph has girth at least g using non-backtracking BFS from every vertex
    def check_girth(edges, n, g):
        """Check if graph has no cycle shorter than g using non-backtracking walks."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking BFS: state is (vertex, prev_vertex)
        # Distance is number of edges in the non-backtracking walk
        for start in range(n):
            # BFS from each neighbor of start
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((start, nb, 1))
            
            while queue:
                prev, curr, dist = queue.popleft()
                if dist >= g:
                    break
                if (prev, curr) in visited:
                    continue
                visited.add((prev, curr))
                
                for nb in adj[curr]:
                    if nb != prev:  # non-backtracking
                        if nb == start and dist + 1 < g:
                            return False  # Found a cycle shorter than g
                        queue.append((curr, nb, dist + 1))
        
        return True
    
    # Try random Cayley graph on (Z/2)^m x S_m or similar
    # For k=5, g=9, we need 5 involutory generators
    # Let's try on the wreath product or direct product of small groups
    
    # Approach: Use a group G and k involutory generators S = {s1, ..., sk}
    # The Cayley graph Cay(G, S) has girth >= g if there's no non-trivial relation
    # of length < g in the generators
    
    # For small groups, we can enumerate all elements and check
    
    def try_cayley_on_direct_product(m1, m2, num_trials=100):
        """Try Cayley graph on (Z/2)^m1 x (Z/2)^m2 with random involutory generators."""
        n1 = 1 << m1
        n2 = 1 << m2
        n = n1 * n2
        
        # Elements are (a, b) where a in [0, n1), b in [0, n2)
        # Involutions are elements of order 2
        # In (Z/2)^m, every element is an involution (except identity)
        
        # Generate random sets of k involutions (non-identity)
        all_involutions_1 = list(range(1, n1))
        all_involutions_2 = list(range(1, n2))
        
        best_result = None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Pick k random involutions from the direct product
            # Each generator is (a, b) where a in Z/2^m1, b in Z/2^m2, not both zero
            generators = []
            for _ in range(k):
                a = random.choice(all_involutions_1) if random.random() < 0.5 else 0
                b = random.choice(all_involutions_2) if random.random() < 0.5 else 0
                if a == 0 and b == 0:
                    continue
                if (a, b) in generators:
                    continue
                generators.append((a, b))
            
            if len(generators) < k:
                continue
            
            # Build the Cayley graph
            # Vertex is (i, j) where i in [0, n1), j in [0, n2)
            # Map to single index: i * n2 + j
            edges = []
            edge_set = set()
            
            for i in range(n1):
                for j in range(n2):
                    v = i * n2 + j
                    for (a, b) in generators:
                        ni = i ^ a
                        nj = j ^ b
                        nv = ni * n2 + nj
                        if nv > v:
                            edge_set.add((v, nv))
            
            edges = list(edge_set)
            
            # Check girth
            if check_girth(edges, n, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
            
            if best_result is not None and n < mb + 100:
                break
        
        return best_result
    
    # Try different group sizes
    best_overall = None
    
    # Try (Z/2)^m for various m
    # For k=5, we need at least 5 non-identity elements, so m >= 3 (7 elements)
    # But we need enough elements for good girth, so try larger m
    
    for m in [6, 7, 8]:
        if time.time() > deadline:
            break
        
        n = 1 << m
        # Moore bound is ~426, so n should be > 426
        # 2^8 = 256 < 426, 2^9 = 512 > 426
        if n < mb:
            continue
        
        result = try_cayley_on_direct_product(m, 0, num_trials=50)
        if result is not None:
            if best_overall is None or result[0] < best_overall[0]:
                best_overall = result
    
    # Try direct products (Z/2)^a x (Z/2)^b
    for (a, b) in [(4, 4), (3, 5), (5, 3), (4, 5), (5, 4)]:
        if time.time() > deadline:
            break
        
        n = (1 << a) * (1 << b)
        if n < mb:
            continue
        
        result = try_cayley_on_direct_product(a, b, num_trials=30)
        if result is not None:
            if best_overall is None or result[0] < best_overall[0]:
                best_overall = result
    
    # If we found a good Cayley graph, return it
    if best_overall is not None:
        return best_overall
    
    # Fallback to L-U-W construction
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
