import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    # If not the specific hard case, just use LUW
    if k != 3 or g != 16:
        return luw_construct(k, g)

    # For k=3, g=16, try to find a smaller graph using random lifts/search
    
    def check_girth_fast(n, edges, g):
        """
        Check if graph has girth at least g.
        Returns True if valid, False otherwise.
        Uses BFS from each vertex up to depth g//2.
        """
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Degree check
        for i in range(n):
            if len(adj[i]) != k:
                return False

        max_depth = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Found a cycle
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def random_regular_graph(n, k, seed):
        """
        Generate a random k-regular graph on n vertices.
        Returns edges list or None if failed.
        """
        rng = random.Random(seed)
        # Use configuration model with rejection or simple random pairing
        # For k=3, n must be even.
        if (n * k) % 2 != 0:
            return None
        
        # Simple approach: Random pairing of stubs
        # This often creates multi-edges or self-loops, so we use a retry loop or a more robust method.
        # A better method for small k is to use a random permutation of stubs.
        
        stubs = [i for i in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        
        edges = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                return None
            edges.add((u, v))
        
        if len(edges) != n * k // 2:
            return None
            
        return list(edges)

    # Moore Bound for k=3, g=16
    # n >= k * (1 + (k-1) + ... + (k-1)^(g/2 - 1))
    # For even g: n >= k * ( ( (k-1)^(g/2) - 1 ) / (k-2) )
    # k=3, g=16 -> g/2 = 8
    # n >= 3 * ( (2^8 - 1) / 1 ) = 3 * 255 = 765
    # Wait, Moore bound for girth g (even) is n >= k * ( (k-1)^(g/2 - 1) + (k-1)^(g/2 - 2) + ... + 1 )?
    # Actually, the lower bound is k * ( (k-1)^(g/2) - 1 ) / (k-2) is not quite right.
    # Let's use the standard lower bound: n >= k * sum_{i=0}^{g/2 - 1} (k-1)^i
    # For k=3, g=16: sum_{i=0}^7 2^i = 2^8 - 1 = 255.
    # n >= 3 * 255 = 765.
    
    # The prompt says best lower bound is 512. My calculation might be off or the bound is tighter for cages.
    # Actually, for k=3, g=16, the Moore bound is 3 * (2^8 - 1) = 765.
    # The prompt says "best lower bound is 512". This suggests the theoretical Moore bound might be higher,
    # but there are better bounds or I am misremembering the formula.
    # Let's check: n(3,16) >= 3 * (2^8 - 1) = 765?
    # If the lower bound is 512, then 765 is not a lower bound?
    # Ah, the Moore bound is a lower bound. If the lower bound is 512, then 765 cannot be the Moore bound.
    # Let's re-evaluate.
    # For k=3, g=16:
    # Tree of depth 7 (radius g/2 - 1 = 7) from a vertex.
    # Nodes: 1 + 3 + 3*2 + ... + 3*2^6 = 1 + 3 * (2^7 - 1) = 1 + 3*127 = 382.
    # This is the number of vertices at distance < 7.
    # The total number of vertices n must be at least 2 * (number of vertices in a ball of radius g/2 - 1) - 1?
    # No, for even girth g=2r, n >= 2 * (1 + k * sum_{i=0}^{r-2} (k-1)^i).
    # r=8. sum_{i=0}^6 2^i = 2^7 - 1 = 127.
    # n >= 2 * (1 + 3 * 127) = 2 * 382 = 764.
    # So 764 is the Moore bound.
    # Why does the prompt say 512? Maybe the prompt is referring to a different bound or I am misinterpreting "best lower bound".
    # Regardless, 765 is a safe lower bound to start searching.
    
    # The current program uses LUW which produces n=936 (or similar).
    # We want to find n < 936.
    # Let's search from n=764 upwards.
    
    # However, random regular graphs of size ~800 with girth 16 are extremely rare.
    # The probability of a random 3-regular graph having girth 16 is very low for n=800.
    # Expected number of cycles of length c is approx ( (k-1)^c / (2c) ).
    # For c=16: (2^16) / 32 = 65536 / 32 = 2048 cycles.
    # So random graphs will have many short cycles.
    # We need a smarter construction or a much larger search space?
    # No, if random graphs have 2048 cycles of length 16, they definitely have cycles of length < 16.
    # So random search from n=764 is unlikely to succeed quickly.
    
    # Alternative: Use the LUW result as the fallback, but try to improve it?
    # The inspiration mentions "deterministic algebraic lift".
    # The current program already does LUW.
    
    # Let's try a different structural change:
    # Use a "cage" database approach? No, we must construct.
    
    # Maybe the LUW construction in the current program is not optimal?
    # The current program tries d from 6 to 12.
    # The standard LUW for k=3, g=16 uses d=11?
    # Let's check the score. Current score is 3.0.
    # Inspiration 1 and 2 have scores ~14.
    # They use "Deterministic Algebraic Lift construction".
    
    # I will implement a generic algebraic lift search that scans for valid parameters,
    # similar to the inspirations, but ensuring it's distinct from the simple LUW parameter sweep.
    # Specifically, I will try to construct graphs using the "lift" of a smaller base graph (like Petersen)
    # with varying lift sizes m, and verify girth.
    
    # Base graph: Petersen graph (10 vertices, girth 5).
    # Lift size m. Total vertices 10*m.
    # We want girth 16.
    # If we lift Petersen with m=100, n=1000.
    # We need to find the smallest m such that the lift has girth >= 16.
    
    def petersen_edges():
        # Standard Petersen graph edges
        # Outer pentagon: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        edges = []
        for i in range(5):
            edges.append((i, (i+1)%5))
        edges.append((5, 7))
        edges.append((7, 9))
        edges.append((9, 6))
        edges.append((6, 8))
        edges.append((8, 5))
        for i in range(5):
            edges.append((i, i+5))
        return edges

    def build_lift(m, seed):
        """
        Build a random lift of the Petersen graph of size m.
        """
        base_edges = petersen_edges()
        rng = random.Random(seed)
        n = 10 * m
        edges = set()
        
        # For each edge (u, v) in base, add edges (u + i*m, v + (i + offset) * m)
        # Or more generally, use a random permutation for each edge?
        # A standard lift uses a fixed offset or a random permutation.
        # Let's use random offsets for each edge type.
        
        # There are 15 edges in Petersen.
        # We assign a random shift s_e in [0, m-1] for each edge e.
        # Then for each i in 0..m-1, add edge (u + i*m, v + (i + s_e) % m * m) ?
        # No, vertex indices are (u, i).
        # Edge (u, v) becomes (u, i) -> (v, (i + s_e) % m).
        
        shifts = [rng.randint(0, m-1) for _ in base_edges]
        
        for idx, (u, v) in enumerate(base_edges):
            s = shifts[idx]
            for i in range(m):
                u_idx = u * m + i
                v_idx = v * m + ((i + s) % m)
                if u_idx == v_idx:
                    return None
                if u_idx > v_idx:
                    u_idx, v_idx = v_idx, u_idx
                edges.add((u_idx, v_idx))
                
        if len(edges) != 15 * m:
            return None
            
        return list(edges)

    best_n = float('inf')
    best_edges = None

    # Try LUW first as a baseline
    try:
        n_luw, edges_luw = luw_construct(k, g)
        if check_girth_fast(n_luw, edges_luw, g):
            best_n = n_luw
            best_edges = edges_luw
    except Exception:
        pass

    # Search for lifts
    # Start m from a reasonable lower bound.
    # n = 10*m.
    # If best_n is 936, m=94.
    # Let's try m from 50 to 100.
    
    if time.time() < deadline:
        for m in range(50, 101):
            if 10 * m >= best_n:
                continue
            # Try a few seeds
            for seed in range(10):
                if time.time() > deadline:
                    break
                edges_try = build_lift(m, seed)
                if edges_try:
                    if check_girth_fast(10*m, edges_try, g):
                        best_n = 10*m
                        best_edges = edges_try
                        break
            if 10 * m >= best_n:
                break

    if best_edges is None:
        # Fallback to LUW if nothing better found
        n_luw, edges_luw = luw_construct(k, g)
        return n_luw, edges_luw

    return best_n, best_edges
