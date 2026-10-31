import time
import random
import sys
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def seed_construct(k, g):
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

    def check_girth_fast(adj, n, g):
        """Check girth >= g using non-backtracking walks with array-based storage."""
        # For each start vertex, do non-backtracking BFS up to depth g//2
        # Track visited (node, prev) pairs using a flat array
        # Key: (node * n + prev) -> depth, stored in a dict for speed
        for start in range(n):
            visited = {}
            queue = []
            for nb in adj[start]:
                key = nb * n + start
                visited[key] = 1
                queue.append((nb, start, 1))
            head = 0
            while head < len(queue):
                u, prev_u, dist = queue[head]
                head += 1
                if dist >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev_u:
                        continue
                    if v == start:
                        if dist + 1 < g:
                            return False
                    else:
                        key = v * n + u
                        if key in visited:
                            if dist + 1 < g:
                                return False
                        else:
                            visited[key] = dist + 1
                            queue.append((v, u, dist + 1))
        return True

    def build_random_3reg(n, rng):
        """Build a random 3-regular graph on n vertices (n must be even)."""
        if n < 4 or n % 2 != 0:
            return None
        # Configuration model: create 3n half-edges, pair them randomly
        half_edges = list(range(n)) * 3
        rng.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        for i in range(0, 3 * n, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return adj

    def build_petersen_2lift(m, rng):
        """Build a 2-lift of the Petersen graph using random permutations."""
        # Petersen graph: 10 vertices, 15 edges
        petersen_edges = []
        for i in range(5):
            petersen_edges.append((i, (i + 1) % 5))
        inner = [5, 6, 7, 8, 9]
        for i in range(5):
            petersen_edges.append((inner[i], inner[(i + 2) % 5]))
        for i in range(5):
            petersen_edges.append((i, i + 5))
        
        n = 20  # 10 * 2
        adj = [[] for _ in range(n)]
        
        for (p, q) in petersen_edges:
            # For 2-lift: each vertex splits into two
            # Use a random bit to decide the lift
            b = rng.randint(0, 1)
            # p -> p*2, p*2+1 ; q -> q*2, q*2+1
            # If b=0: p*2 -> q*2, p*2+1 -> q*2+1
            # If b=1: p*2 -> q*2+1, p*2+1 -> q*2
            for i in range(2):
                u = p * 2 + i
                v = q * 2 + (i ^ b)
                adj[u].append(v)
                adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return adj

    def build_random_lift(base_n, base_edges, m, rng):
        """Build an m-lift of a 3-regular base graph using random permutations."""
        n = base_n * m
        adj = [[] for _ in range(n)]
        
        for (p, q) in base_edges:
            perm = list(range(m))
            rng.shuffle(perm)
            for i in range(m):
                u = p * m + i
                v = q * m + perm[i]
                adj[u].append(v)
                adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return adj

    # Petersen graph edges
    petersen_edges = []
    for i in range(5):
        petersen_edges.append((i, (i + 1) % 5))
    inner = [5, 6, 7, 8, 9]
    for i in range(5):
        petersen_edges.append((inner[i], inner[(i + 2) % 5]))
    for i in range(5):
        petersen_edges.append((i, i + 5))

    # Strategy: Try various constructions
    # 1. Random 3-regular graphs (need n large enough for girth 16)
    #    Girth of random 3-reg ~ (2/3) log_2(n), so for girth 16: n ~ 2^24 (too large)
    #    But we can try and check quickly
    # 2. Lifts of Petersen graph (girth 5 base, lifts can have higher girth)
    # 3. Use the seed construction as fallback

    rng = random.Random(42)

    # First, try to find a good graph via random 3-regular with n around 1000-2000
    # The expected girth of random 3-regular on n vertices is about (2/3)*log_2(n)
    # For n=1000: ~ (2/3)*10 = 6.7 -> too small
    # For n=10000: ~ (2/3)*13.3 = 8.9 -> still too small
    # Random 3-regular won't work for girth 16 with reasonable n
    
    # Better: Use algebraic constructions or known high-girth graph families
    # The LUW construction gives 936 vertices. Let's try to find something smaller.
    # 
    # Key insight: Use a 2-lift or small lift of a graph with girth >= 14
    # to potentially get girth 16 with fewer vertices than 936.
    #
    # Actually, let's try a different approach: use the known fact that
    # the (3,16)-cage has 936 vertices, but try to find a graph with
    # fewer vertices that still has girth >= 16.
    #
    # Strategy: Try random 3-regular graphs of various sizes and check girth.
    # For efficiency, only check girth up to a certain depth first.

    # Try random 3-regular graphs
    # For girth 16, we need the graph to have no cycles of length 4, 6, 8, 10, 12, 14
    # Random 3-regular graphs will almost certainly have 4-cycles for n > 100
    
    # Better structural approach: Use a bipartite 3-regular graph construction
    # or use the fact that we can build graphs from algebraic structures.
    
    # Let's try: construct a 3-regular graph using a "wiring diagram" approach
    # where we control the girth more carefully.
    
    # Approach: Start with a path/cycle and add edges carefully to avoid short cycles.
    # This is essentially a greedy construction.
    
    def greedy_3reg_girth(n, target_girth, rng, max_time=30):
        """Greedy construction of 3-regular graph with girth >= target_girth."""
        adj = [[] for _ in range(n)]
        remaining = list(range(n))
        rng.shuffle(remaining)
        
        # Track which edges we can add without creating short cycles
        # For simplicity, use a BFS-based check for each candidate edge
        
        def would_create_short_cycle(u, v, adj, max_cycle_len):
            """Check if adding edge (u,v) would create a cycle shorter than max_cycle_len."""
            # BFS from u avoiding v, see if we can reach v in < max_cycle_len steps
            # Actually, we need to check if there's a path from u to v of length < max_cycle_len - 1
            # that doesn't use the edge (u,v)
            if u == v:
                return True
            if v in adj[u]:
                return True  # would be multi-edge
            
            # BFS from u
            visited = {u: 0}
            queue = [u]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                dist = visited[node]
                if dist >= max_cycle_len - 1:
                    continue
                for nb in adj[node]:
                    if nb == v and dist + 1 >= max_cycle_len - 1:
                        # This would create a cycle of length dist + 1 + 1 = dist + 2
                        # which is >= max_cycle_len, so OK
                        continue
                    if nb == v:
                        return True  # cycle of length dist + 2 < max_cycle_len
                    if nb not in visited:
                        visited[nb] = dist + 1
                        queue.append(nb)
            return False

        # This is too slow for large n. Let's use a different approach.
        # Instead, build the graph using the configuration model but reject
        # any pairing that creates a short cycle.
        
        return None

    # The greedy approach is too slow. Let's use a different strategy:
    # Use the known LUW construction but try to find a smaller graph.
    # 
    # Actually, the best approach given the constraints:
    # 1. Try a few random 3-regular graphs of size ~500-1000 (quick girth check)
    # 2. If none work, fall back to the seed construction (936 vertices)
    # 
    # But random 3-regular graphs of size 500-1000 will have girth ~ 5-7,
    # far below 16. So this won't work.
    #
    # The real structural change: Use a **voltage graph / covering graph** approach
    # where we take a small base graph with high girth and lift it.
    #
    # For 3-regular graphs, the Petersen graph (girth 5) is the smallest.
    # A 2-lift of Petersen has 20 vertices, still small girth.
    # We need much larger lifts.
    #
    # Key insight: The girth of a random m-lift of a graph with girth g_base
    # is approximately g_base + c*log(m) for some constant c.
    # For Petersen (girth 5), we need 5 + c*log(m) >= 16, so log(m) >= 11/c.
    # If c ~ 1, m >= 2^11 = 2048, giving n = 10*2048 = 20480 vertices. Too large.
    #
    # Better base: Use a graph with girth 8 or higher.
    # The (3,8)-cage has 30 vertices (the (3,8)-cage is known).
    # Actually, the smallest 3-regular graph with girth 8 has 30 vertices? No.
    # n(3,8) = 30 is the Moore bound, but the actual cage might be larger.
    # n(3,6) = 10 (Petersen), n(3,8) is not achieved by Moore bound.
    # The (3,8)-cage has 30 vertices? Let me think...
    # Actually for girth 8, degree 3: the Moore bound is 2*(3^(3-1) - 1) + 2 = 2*8+2 = 18? No.
    # Moore bound for odd girth g: 1 + d * sum_{i=0}^{(g-3)/2} (d-1)^i
    # For g=8 (even): 2 * sum_{i=0}^{g/2-1} (d-1)^i = 2*(1 + 2 + 4) = 14? 
    # Hmm, let me just use known cages.
    #
    # Known: n(3,5)=10, n(3,6)=10 (Petersen has girth 5, not 6)
    # Actually Petersen has girth 5. n(3,6) >= 10, and there exist 3-regular girth-6 graphs on 10 vertices? No, Petersen has 5-cycles.
    # n(3,6) = 10 is not achievable. The (3,6)-cage has 10 vertices? No.
    # Let me just focus on what works: the seed gives 936.
    #
    # NEW STRUCTURAL APPROACH: Use a **Heisenberg group Cayley graph** properly.
    # The Margulis construction for degree 3:
    # - Take the Heisenberg group H_p over F_p (order p^3)
    # - The Cayley graph with generators {(1,0,0), (0,1,0), (0,0,1)} and their inverses
    #   is 6-regular on p^3 vertices.
    # - To get 3-regular: use a 2-to-1 covering or a different generator set.
    #
    # Actually, for the Margulis construction of 3-regular Ramanujan graphs:
    # Use the group G = { (x,y,z) in F_p^3 } with multiplication
    # (x1,y1,z1)*(x2,y2,z2) = (x1+x2, y1+y2, z1+z2+x1*y2)
    # Generators: a = (1,0,0), b = (0,1,0), c = (0,0,1)
    # The Cayley graph with S = {a, b, c, a^{-1}, b^{-1}, c^{-1}} is 6-regular.
    #
    # For 3-regular: we can use a **bipartite double cover**.
    # Vertices: (h, s) where h in H_p, s in {0,1}
    # Edges: (h, 0) -- (h*a, 1), (h, 0) -- (h*b, 1), (h, 0) -- (h*c, 1)
    #         (h, 1) -- (h*a^{-1}, 0), (h, 1) -- (h*b^{-1}, 0), (h, 1) -- (h*c^{-1}, 0)
    # This gives a 3-regular bipartite graph on 2*p^3 vertices.
    #
    # The girth of this graph is related to the girth of the 6-regular Cayley graph.
    # For p=3: 2*27 = 54 vertices. Girth will be small.
    # For p=5: 2*125 = 250 vertices.
    # For p=7: 2*343 = 686 vertices.
    # For p=11: 2*1331 = 2662 vertices.
    #
    # The girth of the Margulis 6-regular Cayley graph of H_p is about 2*log_p(p^3) = 6.
    # So the bipartite double cover will have girth about 12. Not quite 16.
    #
    # For larger p: p=13: 2*2197 = 4394. Too large.
    # p=7 gives 686 vertices. If girth is ~12, not enough.
    #
    # Let's just try it and check the actual girth.

    def heisenberg_bipartite(p, rng):
        """Build 3-regular bipartite graph from Heisenberg group H_p."""
        n = 2 * p * p * p
        # Group elements: (x, y, z) in F_p^3
        # Multiplication: (x1,y1,z1)*(x2,y2,z2) = (x1+x2, y1+y2, z1+z2+x1*y2) mod p
        # Inverse: (x,y,z)^{-1} = (-x, -y, -z - (-x)*y) = (-x, -y, z + x*y) mod p
        # Wait: (x,y,z)*(-x,-y,-z+x*y) should = (0,0,0)
        # (x+(-x), y+(-y), z+(-z+x*y)+x*(-y)) = (0, 0, x*y - x*y) = (0,0,0). Yes!
        
        def group_mult(e1, e2):
            x1, y1, z1 = e1
            x2, y2, z2 = e2
            return ((x1+x2) % p, (y1+y2) % p, (z1+z2+x1*y2) % p)
        
        def group_inv(e):
            x, y, z = e
            return ((-x) % p, (-y) % p, (z + x*y) % p)
        
        # Generators
        a = (1, 0, 0)
        b = (0, 1, 0)
        c = (0, 0, 1)
        a_inv = group_inv(a)
        b_inv = group_inv(b)
        c_inv = group_inv(c)
        
        # Precompute all group elements and their indices
        elements = [(x, y, z) for x in range(p) for y in range(p) for z in range(p)]
        elem_index = {e: i for i, e in enumerate(elements)}
        
        # Bipartite: side 0 = elements, side 1 = elements
        # Vertex (e, 0) connects to (e*a, 1), (e*b, 1), (e*c, 1)
        # Vertex (e, 1) connects to (e*a_inv, 0), (e*b_inv, 0), (e*c_inv, 0)
        
        adj = [[] for _ in range(n)]
        half = p * p * p
        
        for i, e in enumerate(elements):
            # Side 0 vertex: i
            for gen in [a, b, c]:
                f = group_mult(e, gen)
                j = elem_index[f]
                adj[i].append(half + j)
                adj[half + j].append(i)
            # Side 1 vertex: half + i
            for gen in [a_inv, b_inv, c_inv]:
                f = group_mult(e, gen)
                j = elem_index[f]
                adj[half + i].append(j)
                adj[j].append(half + i)
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return adj

    # Try Heisenberg bipartite for various p
    for p in [3, 5, 7]:
        if time.time() > deadline - 20:
            break
        adj = heisenberg_bipartite(p, rng)
        if adj is None:
            continue
        n = len(adj)
        
        # Check girth
        if check_girth_fast(adj, n, g):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
    
    # Try Petersen m-lifts with larger m
    for m in [100, 200, 300, 500, 700, 900]:
        if time.time() > deadline - 10:
            break
        n = 10 * m
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            adj = build_random_lift(10, petersen_edges, m, rng)
            if adj is None:
                continue
            if check_girth_fast(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # Fallback to seed
    return seed_construct(k, g)
