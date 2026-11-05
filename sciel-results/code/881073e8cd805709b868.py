import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def is_quadratic_residue(a, q):
        a = a % q
        if a == 0:
            return True
        if not is_prime_power(q):
            return True
        p = None
        for cand in range(2, int(math.isqrt(q)) + 1):
            if q % cand == 0:
                t = q
                while t % cand == 0:
                    t //= cand
                if t == 1:
                    p = cand
                    break
        if p is None:
            p = q
        a_p = a % p
        if a_p == 0:
            return True
        return pow(a_p, (p - 1) // 2, p) == 1

    def modinv(a, m):
        if m <= 1:
            return None
        a = a % m
        if a == 0:
            return None
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x, y = egcd(b, a % b)
            return (g, y, x - (a // b) * y)
        g, x, _ = egcd(a, m)
        if g != 1:
            return None
        return x % m

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- graph validation ----------
    def get_girth_fast(adj, n, limit=None):
        """Returns the girth of the graph using BFS from each vertex.
        If limit is given, stops early if girth >= limit (returns limit)."""
        min_girth = float('inf')
        if limit is not None:
            min_girth = limit
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                if dist[x] + 1 >= min_girth:
                    continue
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if limit is not None and min_girth >= limit:
                                return min_girth
        return min_girth if min_girth != float('inf') else 0

    def is_connected(adj, n):
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return None
        if not is_connected(adj, n):
            return None
        girth = get_girth_fast(adj, n, limit=g)
        if girth < g:
            return None
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
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

    # ---------- PGL(2, q) machinery ----------
    def action_on_point(x, a, b, c, d, q):
        if x == q:
            if c == 0:
                return q
            inv_c = modinv(c, q)
            if inv_c is None:
                return None
            return (a * inv_c) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        inv_den = modinv(den, q)
        if inv_den is None:
            return None
        return (num * inv_den) % q

    def inv_matrix(a, b, c, d, q):
        det = (a * d - b * c) % q
        det_inv = modinv(det, q)
        if det_inv is None:
            return None
        return (d * det_inv % q, -b * det_inv % q, -c * det_inv % q, a * det_inv % q)

    def is_derangement(a, b, c, d, q):
        a, b, c, d = a % q, b % q, c % q, d % q
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        if is_quadratic_residue(disc, q):
            return False
        return True

    def build_from_gens(gens, q, k, g):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                adj[i].add(j)
                adj[j].add(i)
        return validate(adj, n, k, g)

    def random_search_pgl(q, k, g, budget):
        n = q + 1
        if n < n_min:
            return None
        if q % 2 == 0 or not is_prime_power(q):
            return None

        def random_gen():
            while True:
                a = random.randrange(q)
                b = random.randrange(q)
                c = random.randrange(q)
                d = random.randrange(q)
                if (a * d - b * c) % q != 0:
                    return (a, b, c, d)

        def random_gens():
            out = []
            attempts = 0
            while len(out) < k and attempts < k * 20:
                attempts += 1
                m = random_gen()
                mi = inv_matrix(*m, q)
                if mi is None:
                    continue
                if not is_derangement(*m, q):
                    continue
                a2 = (m[0]*m[0] + m[1]*m[2]) % q
                b2 = (m[1]*(m[0]+m[3])) % q
                c2 = (m[2]*(m[0]+m[3])) % q
                d2 = (m[3]*m[3] + m[1]*m[2]) % q
                is_involution = (a2 == 1 and b2 == 0 and c2 == 0 and d2 == 1)
                
                if is_involution or random.random() < 0.4:
                    out.append(m)
                    if mi != m and len(out) < k:
                        out.append(mi)
            while len(out) < k:
                m = random_gen()
                if is_derangement(*m, q):
                    out.append(m)
            return out[:k]

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # ---------- New: Greedy edge swap with cycle detection ----------
    def greedy_swap_search(n, k, g, budget):
        """
        Start with a random k-regular graph and greedily swap edges to increase girth.
        Use local cycle detection to avoid creating short cycles.
        """
        if n < n_min or n > 300:
            return None

        def random_k_regular(n, k):
            if (n * k) % 2 != 0:
                return None
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if not valid:
                return None
            if not is_connected(adj, n):
                return None
            return adj

        def find_short_cycle(adj, n, max_len):
            """Find a cycle of length < max_len, return it or None."""
            for start in range(n):
                # BFS with parent tracking
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([(start, -1)])  # (node, parent)
                while queue:
                    u, p = queue.popleft()
                    if dist[u] + 1 >= max_len:
                        continue
                    for v in adj[u]:
                        if v == p:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append((v, u))
                        else:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < max_len:
                                # Reconstruct cycle
                                cycle = [u]
                                x = u
                                while x != v:
                                    x = parent[x]
                                    cycle.append(x)
                                cycle.append(v)
                                # Reverse to get proper cycle
                                return cycle
            return None

        def try_swap(adj, n, k, g):
            """Try a random edge swap that doesn't create short cycles."""
            # Pick two random edges (u,v) and (x,y)
            u = random.randrange(n)
            if not adj[u]:
                return False
            v = random.choice(list(adj[u]))
            
            x = random.randrange(n)
            if x in (u, v) or x not in adj[u]:  # x not adjacent to u
                if x == u or x == v:
                    return False
                if x in adj[u]:
                    return False
            else:
                return False
            
            if not adj[x]:
                return False
            y = random.choice(list(adj[x]))
            
            if y in (u, v):
                return False
            if y in adj[v]:
                return False
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            # Check if this created a short cycle
            if find_short_cycle(adj, n, g) is not None:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj[u].add(v)
                adj[v].add(u)
                adj[x].add(y)
                adj[y].add(x)
                return False
            
            return True

        # Generate initial random k-regular graph
        adj = None
        for _ in range(50):
            if time.time() > budget:
                return None
            adj = random_k_regular(n, k)
            if adj:
                break
        if not adj:
            return None

        # Check initial girth
        girth = get_girth_fast(adj, n, limit=g)
        if girth >= g:
            return validate(adj, n, k, g)

        best_girth = girth
        best_adj = [set(s) for s in adj]
        
        # Greedy swaps
        stall_count = 0
        while time.time() < budget:
            if try_swap(adj, n, k, g):
                stall_count = 0
                new_girth = get_girth_fast(adj, n, limit=g)
                if new_girth >= g:
                    return validate(adj, n, k, g)
                if new_girth > best_girth:
                    best_girth = new_girth
                    best_adj = [set(s) for s in adj]
            else:
                stall_count += 1
                if stall_count > 1000:
                    break
        
        return None

    # ---------- New: Integer Programming via Constraint Propagation ----------
    def cp_construct(n, k, g, budget):
        """
        Use constraint propagation to build a k-regular graph with girth >= g.
        This is a structural approach different from local search.
        """
        if n < n_min or n > 200:
            return None

        # Start with an empty graph
        adj = [set() for _ in range(n)]
        
        # Build a spanning tree first to ensure connectivity
        # Random DFS tree
        visited = [False] * n
        visited[0] = True
        stack = [0]
        while stack:
            u = stack.pop()
            unvisited_neighbors = [v for v in range(n) if not visited[v]]
            random.shuffle(unvisited_neighbors)
            for v in unvisited_neighbors:
                if len(adj[u]) >= k:
                    break
                adj[u].add(v)
                adj[v].add(u)
                visited[v] = True
                stack.append(v)
        
        # Check if we have a spanning tree
        if not is_connected(adj, n):
            return None
        
        # Now add edges to reach k-regularity while avoiding short cycles
        def would_create_short_cycle(u, v, max_len):
            """Check if adding edge (u,v) creates a cycle shorter than max_len."""
            # BFS from u to v, avoiding direct edge
            if v in adj[u]:
                return False  # Edge already exists
            
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            found = False
            while queue:
                x = queue.popleft()
                if x == v:
                    found = True
                    break
                if dist[x] + 1 >= max_len:
                    continue
                for y in adj[x]:
                    if y == parent[x]:
                        continue
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
            
            if found:
                # The shortest path from u to v (excluding direct edge) has length dist[v]
                # Adding edge (u,v) creates a cycle of length dist[v] + 1
                if dist[v] + 1 < max_len:
                    return True
            return False

        # Add edges greedily
        progress = True
        while progress and time.time() < budget:
            progress = False
            # Find vertices with degree < k
            underfull = [v for v in range(n) if len(adj[v]) < k]
            random.shuffle(underfull)
            
            for u in underfull:
                if len(adj[u]) >= k:
                    continue
                # Find candidates to connect to
                candidates = [v for v in range(n) if v != u and v not in adj[u] and len(adj[v]) < k]
                random.shuffle(candidates)
                
                for v in candidates:
                    if len(adj[u]) >= k or len(adj[v]) >= k:
                        continue
                    if not would_create_short_cycle(u, v, g):
                        adj[u].add(v)
                        adj[v].add(u)
                        progress = True
                        break
        
        # Check if k-regular
        if all(len(adj[v]) == k for v in range(n)):
            girth = get_girth_fast(adj, n, limit=g)
            if girth >= g:
                return validate(adj, n, k, g)
        
        return None

    # ---------- Cyclic interval (circulant) construction ----------
    def circulant_construct(k, g, budget):
        """
        Try circulant graphs C(n, S) where vertex i connects to i+s mod n for s in S.
        For k-regular: if k is odd, S has (k+1)//2 elements with one being n/2 (when n even)
        or we use asymmetric shifts. For k even, S has k/2 elements.
        
        Key idea: enumerate shift sets and check girth via BFS on the circulant.
        """
        n_start = n_min
        # For circulant to be k-regular with k=5:
        # If n is even: shifts can be {d1, d2, d3, d4, n/2} (5 shifts, n/2 is self-inverse)
        # If n is odd: shifts must come in pairs {±d1, ±d2, ±d3} + one more... 
        # Actually for k=5 on Z_n: we need 5 neighbors. 
        # For even n: {±a, ±b, ±c, ±d, n/2} gives degree 8... no.
        # Let me think: each shift s adds edge i->i+s. If s != n-s, it's undirected pair.
        # For degree k=5: we need exactly 5 distinct neighbors.
        # If n even: shifts {a, b, c, d, n/2} where n/2 is self-inverse.
        #   Neighbors of i: i±a, i±b, i±c, i±d, i+n/2. That's 9 neighbors if all distinct.
        #   Wait, no. Each shift s means edge between i and i+s. Since undirected,
        #   shift s and shift n-s give the same edge. So we pick shifts from {1,...,n//2}.
        #   For degree k: pick k shifts from {1,...,n//2} (when n even, n/2 is valid).
        #   For n odd: pick k shifts from {1,...,(n-1)//2}, but each gives 2 neighbors (±s),
        #   so degree would be 2k... That doesn't work for odd k.
        #   
        # Correct approach: For circulant C(n, S) where S is a set of shifts,
        # vertex i is connected to i+s for all s in S. Degree = |S| if we count directed,
        # but undirected degree = |S ∪ (-S)| / ... 
        # Actually in standard circulant: C(n, S) has edges {i, i+s} for s in S.
        # If s in S and n-s in S, they give the same undirected edge.
        # So for k-regular: we need |{s mod n : s in S, s != 0}| = k,
        # and S should be symmetric (if s in S then n-s in S) for the graph to be well-defined
        # as undirected. Then degree = |S|/2 * 2 = |S|... 
        # 
        # Simpler: just pick k distinct nonzero shifts s_1,...,s_k from Z_n,
        # and for each i, connect i to i+s_j for j=1..k.
        # This gives degree k. The graph is simple as long as all s_j are distinct mod n
        # and no s_j = 0.
        # For undirected: edge (i, i+s) = edge (i+s, i) = edge (i+s, i+2s)... 
        # No wait, in a circulant with shifts S, the edges are {(i, i+s) : i in Z_n, s in S}.
        # This is undirected if for every s in S, -s is also in S (or we just take the set
        # of undirected edges). Let me just use: neighbors of i are {i+s mod n : s in S}
        # where S has exactly k elements, none being 0. For the graph to be undirected and
        # simple, we need that if i connects to j then j connects to i, which means
        # j-i in S implies i-j in S, i.e., S = -S. So S must be symmetric.
        # For k odd: S = {a, -a, b, -b, c, -c, ...} doesn't work for odd k.
        # For k=5: S = {a, -a, b, -b, c} where c = -c mod n, i.e., 2c = 0 mod n,
        # so c = n/2 (requires n even). So n must be even, and S = {±a, ±b, n/2}.
        # That gives degree 5. 
        # For k even: S = {±a_1, ..., ±a_{k/2}}.
        
        if k % 2 == 1:
            # Need n even, S = {±a, ±b, ±c, n/2} for k=5
            # General odd k: need k-1 even shifts in pairs + one self-inverse (n/2)
            # For k=5: 4 shifts in 2 pairs + n/2
            if k != 5:
                # General odd k: (k-1)/2 pairs + n/2
                half = (k - 1) // 2
                n_max = min(n_start + 500, 600)
                for n in range(n_start, n_max + 1):
                    if time.time() > budget:
                        break
                    if n % 2 != 0:
                        continue
                    n_half = n // 2
                    # Choose half pairs from {1, ..., n_half-1}
                    # Try systematic: use small shifts
                    # Number of pairs to choose: half
                    # For half=2 (k=5): choose 2 values from {1,...,n/2-1}
                    if half == 2:
                        # C(n/2-1, 2) possibilities - scan systematically
                        for a in range(1, n_half):
                            for b in range(a + 1, n_half):
                                S = [a, n - a, b, n - b, n_half]
                                # Build and check
                                adj = [set() for _ in range(n)]
                                valid = True
                                for i in range(n):
                                    for s in S:
                                        j = (i + s) % n
                                        adj[i].add(j)
                                # Check degree and simplicity
                                for i in range(n):
                                    if len(adj[i]) != k:
                                        valid = False
                                        break
                                if not valid:
                                    continue
                                # Check girth
                                girth = get_girth_fast(adj, n, limit=g)
                                if girth >= g:
                                    if is_connected(adj, n):
                                        edges = []
                                        for i in range(n):
                                            for j in adj[i]:
                                                if i < j:
                                                    edges.append((i, j))
                                        return (n, edges)
                    elif half == 3:
                        for a in range(1, n_half):
                            for b in range(a + 1, n_half):
                                for c in range(b + 1, n_half):
                                    S = [a, n-a, b, n-b, c, n-c, n_half]
                                    adj = [set() for _ in range(n)]
                                    valid = True
                                    for i in range(n):
                                        for s in S:
                                            j = (i + s) % n
                                            adj[i].add(j)
                                    for i in range(n):
                                        if len(adj[i]) != k:
                                            valid = False
                                            break
                                    if not valid:
                                        continue
                                    girth = get_girth_fast(adj, n, limit=g)
                                    if girth >= g:
                                        if is_connected(adj, n):
                                            edges = []
                                            for i in range(n):
                                                for j in adj[i]:
                                                    if i < j:
                                                        edges.append((i, j))
                                            return (n, edges)
            return None
        else:
            # k even: S = {±a_1, ..., ±a_{k/2}}
            half = k // 2
            n_max = min(n_start + 500, 600)
            for n in range(n_start, n_max + 1):
                if time.time() > budget:
                    break
                if half == 2:
                    for a in range(1, n // 2):
                        for b in range(a + 1, n // 2):
                            S = [a, n-a, b, n-b]
                            adj = [set() for _ in range(n)]
                            for i in range(n):
                                for s in S:
                                    adj[i].add((i + s) % n)
                            if all(len(adj[i]) == k for i in range(n)):
                                girth = get_girth_fast(adj, n, limit=g)
                                if girth >= g and is_connected(adj, n):
                                    edges = []
                                    for i in range(n):
                                        for j in adj[i]:
                                            if i < j:
                                                edges.append((i, j))
                                    return (n, edges)
                elif half == 3:
                    for a in range(1, n // 2):
                        for b in range(a + 1, n // 2):
                            for c in range(b + 1, n // 2):
                                S = [a, n-a, b, n-b, c, n-c]
                                adj = [set() for _ in range(n)]
                                for i in range(n):
                                    for s in S:
                                        adj[i].add((i + s) % n)
                                if all(len(adj[i]) == k for i in range(n)):
                                    girth = get_girth_fast(adj, n, limit=g)
                                    if girth >= g and is_connected(adj, n):
                                        edges = []
                                        for i in range(n):
                                            for j in adj[i]:
                                                if i < j:
                                                    edges.append((i, j))
                                        return (n, edges)
            return None

    # ---------- main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # 2. Try circulant construction for smaller n
    circ_budget = min(30.0, deadline - 25.0 - time.time())
    if circ_budget > 5.0:
        res = circulant_construct(k, g, deadline - 25.0 - time.time())
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res

    # 3. Try CP construction for smaller n
    cp_budget = min(30.0, deadline - 20.0 - time.time())
    if cp_budget > 10.0:
        for n_try in range(n_min, best_n[0]):
            if time.time() > deadline - 20.0:
                break
            res = cp_construct(n_try, k, g, deadline - 20.0 - time.time())
            if res:
                best_n[0] = res[0]
                best_result = res
                break

    # 4. Try greedy swap search for smaller n
    swap_budget = min(30.0, deadline - 15.0 - time.time())
    if swap_budget > 10.0:
        for n_try in range(n_min, best_n[0]):
            if time.time() > deadline - 15.0:
                break
            res = greedy_swap_search(n_try, k, g, deadline - 15.0 - time.time())
            if res:
                best_n[0] = res[0]
                best_result = res
                break

    # 5. Search for smaller n using PGL
    n_cursor = n_min
    while n_cursor < best_n[0] and time.time() < deadline - 10.0:
        q = n_cursor - 1
        if q >= 3 and is_prime_power(q):
            pgl_budget = min(5.0, deadline - 10.0 - time.time())
            res = random_search_pgl(q, k, g, pgl_budget)
            if res:
                best_n[0] = res[0]
                best_result = res
                n_cursor += 1
                continue
        n_cursor += 1

    return best_result

# EVOLVE-BLOCK-END
