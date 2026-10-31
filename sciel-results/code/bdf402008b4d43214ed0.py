import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # Moore bound check: for k=3, g=16, Moore bound is 2 + 3*(1+2+...+2^7) = 2 + 3*255 = 767
    # Actually the Moore bound formula: n >= 2 + k * sum_{i=0}^{floor((g-2)/2)-1} (k-1)^i
    # For g=16: floor(14/2) = 7, so sum i=0 to 6 of 2^i = 127, n >= 2 + 3*127 = 383
    # Wait, let me recalculate. For girth g (even), g=2r: n >= 2 + k * sum_{i=0}^{r-2} (k-1)^i
    # r = g/2 = 8, sum i=0 to 6 of 2^i = 127, n >= 2 + 3*127 = 383
    # But the task says Moore bound is 512. Let me check: 2 + 3*(1+2+4+8+16+32+64) = 2+3*127=383
    # Hmm, maybe I'm miscounting. Let's just use 512 as stated.

    def check_girth_full(adj, n, g):
        """Full girth verification using non-backtracking BFS from every vertex."""
        for start in range(n):
            dist = {}
            queue = deque()
            for nb in adj[start]:
                dist[(nb, start)] = 1
                queue.append((nb, start, 1))
            while queue:
                u, prev_u, d = queue.popleft()
                if d >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev_u:
                        continue
                    if v == start:
                        if d + 1 < g:
                            return False
                    elif (v, u) in dist:
                        if dist[(v, u)] + 1 < g:
                            return False
                    else:
                        dist[(v, u)] = d + 1
                        queue.append((v, u, d + 1))
        return True

    def check_girth_sampled(adj, n, g, sample_size, rng):
        """Sampled girth check for speed."""
        vertices = list(range(n))
        rng.shuffle(vertices)
        vertices = vertices[:min(n, sample_size)]
        for start in vertices:
            dist = {}
            queue = deque()
            for nb in adj[start]:
                dist[(nb, start)] = 1
                queue.append((nb, start, 1))
            while queue:
                u, prev_u, d = queue.popleft()
                if d >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev_u:
                        continue
                    if v == start:
                        if d + 1 < g:
                            return False
                    elif (v, u) in dist:
                        if dist[(v, u)] + 1 < g:
                            return False
                    else:
                        dist[(v, u)] = d + 1
                        queue.append((v, u, d + 1))
        return True

    def build_petersen_lift(m, rng):
        """Build a random m-lift of the Petersen graph (3-regular, 10m vertices)."""
        petersen_edges = []
        for i in range(5):
            petersen_edges.append((i, (i + 1) % 5))
        inner = [5, 6, 7, 8, 9]
        for i in range(5):
            petersen_edges.append((inner[i], inner[(i + 2) % 5]))
        for i in range(5):
            petersen_edges.append((i, i + 5))

        # For each edge in base graph, assign a random permutation of {0,...,m-1}
        edge_perms = {}
        for (p, q) in petersen_edges:
            perm = list(range(m))
            rng.shuffle(perm)
            edge_perms[(p, q)] = perm
            inv_perm = [0] * m
            for i in range(m):
                inv_perm[perm[i]] = i
            edge_perms[(q, p)] = inv_perm

        n = 10 * m
        adj = [[] for _ in range(n)]
        for (p, q) in petersen_edges:
            perm = edge_perms[(p, q)]
            for i in range(m):
                u = p * m + i
                v = q * m + perm[i]
                adj[u].append(v)
                adj[v].append(u)

        # Verify 3-regular and simple
        for i in range(n):
            if len(adj[i]) != 3 or len(set(adj[i])) != 3:
                return None
            for nb in adj[i]:
                if nb < 0 or nb >= n or i not in adj[nb]:
                    return None
        return adj

    def build_circulant_3reg(n, rng):
        """
        Build a 3-regular circulant-like graph.
        Vertices 0..n-1. Connect i to i+a, i-b, i+c mod n for carefully chosen a,b,c.
        This gives a 3-regular graph if we pick a, b, c such that the three "directions"
        are distinct and involutive (i.e., we use pairs).
        
        Actually for a 3-regular undirected circulant, we need 3 generators where each
        is its own inverse or we pair them. Simplest: use generators a, b, c where
        a+b+c has some structure. 
        
        Better: use the "trivial" 3-regular circulant: connect i to i+1, i-1, i+n/2
        (only works for even n, and n/2 must not create short cycles easily).
        
        Even better: use a random 3-regular graph via configuration model, then
        check girth.
        """
        return None

    def build_random_3reg(n, rng):
        """Build a random 3-regular simple graph using configuration model."""
        if n < 4 or n % 2 != 0:
            return None
        # Each vertex has 3 stubs
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        rng.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        for i in range(n):
            u = stubs[2*i]
            v = stubs[2*i+1]
            if u == v:
                return None  # loop
            if v in adj[u]:
                return None  # multi-edge
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify
        for i in range(n):
            if len(adj[i]) != 3 or len(set(adj[i])) != 3:
                return None
        return adj

    def build_doubly_covered_petersen(m, rng):
        """
        Alternative: build a 3-regular graph by taking a 2-lift of a 3-regular
        base graph with additional structure. For now, stick with Petersen lifts.
        """
        return None

    # Strategy: Try Petersen lifts with increasing m, checking girth.
    # Expected girth of random m-lift of Petersen grows like c*log(m).
    # For girth 16, we need m large enough. Let's try m from 200 upward.
    
    rng = random.Random(42)
    best_result = None

    # Phase 1: Try Petersen lifts with various m values
    m_values = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]
    
    for m in m_values:
        if time.time() > deadline - 10:
            break
        n = 10 * m
        if best_result is not None and n >= best_result[0]:
            continue
        
        num_tries = max(2, 10 - m // 500)
        for attempt in range(num_tries):
            if time.time() > deadline - 10:
                break
            
            adj = build_petersen_lift(m, rng)
            if adj is None:
                continue
            
            # Quick sampled check first
            if check_girth_sampled(adj, n, g, min(n, 200), rng):
                # Full check
                if check_girth_full(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_result = (n, edges)
                    break  # found one for this m, try smaller m next
    
    # Phase 2: If no Petersen lift worked, try random 3-regular graphs
    if best_result is None and time.time() < deadline - 10:
        for n_try in [400, 500, 600, 800, 1000, 1200, 1500]:
            if time.time() > deadline - 10:
                break
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                adj = build_random_3reg(n_try, rng)
                if adj is None:
                    continue
                if check_girth_sampled(adj, n_try, g, min(n_try, 100), rng):
                    if check_girth_full(adj, n_try, g):
                        edges = []
                        for u in range(n_try):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_result = (n_try, edges)
                        break
            if best_result is not None:
                break

    # Phase 3: Fallback - use a known construction
    if best_result is None:
        # Use the seed construction (LUW-based) as fallback
        # This produces a valid graph but may be large
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
        best_result = (len(order), edges)

    return best_result
