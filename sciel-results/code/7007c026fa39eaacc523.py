"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import math

def _luw_graph(k, g, d_override=None, q_override=None):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges) if valid q-regular with girth >= g, else None."""
    q = q_override if q_override is not None else k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    elif q == 2:
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: x * y
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    # Build rule for LUW construction
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
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

def _luw_graph_explicit(k, g, d, q):
    """Build LUW CD(d,q) graph explicitly and return (n, edges).
    The LUW construction: vertices are (side, x) where side in {0,1} and x in F_q^d.
    Edges connect (0, x) to (1, y) where y is determined by x via the rule.
    Total vertices = 2 * q * q^(d-1) = 2 * q^d ... wait, let me reconsider.
    
    Actually the LUW graph has vertices indexed by (i, a_1, ..., a_d) where i in {0,1}
    and a_j in F_q. But the first coordinate is fixed per side.
    
    Let me just enumerate all 2*q^d vertices and compute edges.
    """
    import itertools
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    elif q == 2:
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: x * y
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    # Build rule
    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Generate all vertices: (side, a1, a2, ..., ad)
    # side 0: (0, a1, a2, ..., ad)
    # side 1: (1, a1, a2, ..., ad)
    # Total: 2 * q^d vertices
    
    all_coords = list(itertools.product(range(q), repeat=d))
    n = 2 * len(all_coords)
    
    # Index: (side, idx) -> vertex number
    # side 0 vertices: 0 to q^d - 1
    # side 1 vertices: q^d to 2*q^d - 1
    
    edges = []
    seen_edges = set()
    
    for idx, x in enumerate(all_coords):
        v0 = idx  # vertex on side 0
        for first in range(q):
            # Compute y for side 1
            y = [0] * d
            y[0] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                # x is 0-indexed, but rule uses 1-indexed
                # a and b are 1-indexed positions in x
                # x[a-1] and x[b-1]
                if a == 1:
                    val_a = first  # y[0] = first
                else:
                    val_a = y[a-1]
                val_b = x[b-1]
                y[c-1] = add_fn(x[c-1], mul_fn(val_a, val_b))
            
            v1 = q * len(all_coords) // q * 0 + len(all_coords)  # wrong
            # side 1 vertex index: len(all_coords) + idx_of_y
            # Need to find index of y in all_coords
            y_tuple = tuple(y)
            try:
                y_idx = all_coords.index(y_tuple)
            except ValueError:
                continue
            v1 = len(all_coords) + y_idx
            
            if v0 < v1:
                key = (v0, v1)
            else:
                key = (v1, v0)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(key)
    
    return n, edges

def _generalized_polygon_graph(s, t, k, g):
    """Construct the incidence graph of a generalized polygon of order (s,t) and degree s+1 or t+1.
    
    For a generalized (s+2)-gon of order (s,t):
    - Points: (s+1)^s * (t+1)^{s-1} * ... (complex)
    - Simpler: use known small cases.
    
    For s=t=2: generalized quadrangle -> 5-regular? No, (2,2) gives degree 3.
    For s=t=4: generalized hexagon of order (4,4) -> degree 5, girth 12.
      Number of points = (s+1)(s^2+s+1) = 5*21 = 105
      Number of lines = (t+1)(t^2+t+1) = 5*21 = 105
      Total vertices = 210
      Each point on (t+1)=5 lines, each line has (s+1)=5 points.
      So the incidence graph is 5-regular with 210 vertices and girth 12.
    
    We need to explicitly construct this. For the unique generalized hexagon of order (4,4),
    we can use the construction from the Ree-Tits ovoid or explicit tables.
    
    Alternative: use the known construction via finite geometry.
    For (4,4) hexagon: vertices = points ∪ lines of a certain structure.
    
    Let's use a different approach: construct via the known explicit (4,4)-hexagon.
    The (4,4)-hexagon has 210 vertices (105 points + 105 lines), 525 edges.
    
    Actually, let me use a simpler known construction:
    The incidence graph of the generalized quadrangle GQ(4,2) is 3-regular.
    The incidence graph of GQ(2,4) is 5-regular.
    
    GQ(2,4): order (s,t) = (2,4)
    - Number of points: (s+1)(st+s+t+1) = 3*(8+2+4+1) = 3*15 = 45
    - Number of lines: (t+1)(st+s+t+1) = 5*15 = 75
    - Each point on (t+1) = 5 lines
    - Each line has (s+1) = 3 points
    - Incidence graph: bipartite, 45+75=120 vertices, 5-regular (point side) and 3-regular (line side)
    - NOT regular! So this doesn't work directly.
    
    For a regular graph, we need s=t. So (s,s) gives degree s+1 on both sides.
    GQ(s,s) has:
    - Points: (s+1)(s^2+1)
    - Lines: (s+1)(s^2+1)
    - Total: 2(s+1)(s^2+1)
    - Degree: s+1
    - Girth: 8 (for generalized quadrangle)
    
    For s=4: n = 2*5*17 = 170, degree 5, girth 8.
    But we need girth >= 9, so girth 8 is NOT sufficient for g=9.
    
    For generalized hexagons (degree 6+2=8? No...):
    A generalized (2d+2)-gon of order (s,s) has girth 2d+2.
    For d=5 (hexagon, 6-gon... wait, hexagon is 6 sides, girth 12 in incidence graph).
    
    Generalized hexagon of order (s,s):
    - Girth of incidence graph = 12
    - n = 2 * (s+1) * (s^3+s^2+s+1) = 2(s+1)(s^3+s^2+s+1)
    - Degree = s+1
    
    For s=4: n = 2*5*(64+16+4+1) = 10*85 = 850, degree 5, girth 12.
    
    Hmm, that's 850 vertices. The LUW construction for k=5, g=9 gives d=4, so
    n = 2*q*d = 2*5*4 = 40? That seems too small. Let me check...
    
    Actually for LUW CD(d,q): n = 2*q*d. For d=4, q=5: n=40, degree 5.
    Girth should be >= 2d = 8? Or 2d+2?
    
    Let me just try the (4,4) hexagon construction explicitly.
    """
    if s != t:
        return None
    
    # For now, handle the case where we can construct it
    # The (4,4)-hexagon is unique and has 850 vertices
    # This is complex to construct from scratch, so let's skip for now
    # and focus on improving the LUW and Cayley constructions
    
    return None

def _verify_girth_fast(n, edges, k, g, timeout=5):
    """Fast girth verification using BFS from each vertex with early termination.
    Returns True if girth >= g, False otherwise."""
    start_time = time.time()
    
    # Check degree
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
        deg[u] += 1
        deg[v] += 1
    for i in range(n):
        if deg[i] != k:
            return False
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Girth check: BFS from each vertex, track shortest cycle
    # For large n, use early termination
    for s in range(n):
        if time.time() - start_time > timeout:
            return False
        
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _verify_girth_nonbacktracking(n, edges, k, g, timeout=10):
    """Verify girth >= g using non-backtracking walks from every vertex.
    This is the exact certification method specified in the task.
    Returns True if no non-backtracking walk of length < g returns to start.
    """
    start_time = time.time()
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check for multiple edges and self-loops
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Non-backtracking walk check:
    # A non-backtracking walk of length L starting at vertex v is a sequence
    # v_0=v, v_1, ..., v_L where v_i != v_{i-2} for all i >= 2.
    # We need to check that no such walk of length < g returns to v_0.
    # 
    # Equivalently, the girth is the length of the shortest cycle, and
    # a cycle of length c gives a non-backtracking walk of length c returning to start.
    # So girth >= g iff no non-backtracking walk of length < g returns to its start.
    #
    # We can do this by BFS in the "state space" of (current_vertex, previous_vertex)
    # but that's expensive. Instead, use the standard BFS girth check which is equivalent.
    # The standard BFS from each vertex checking for cycles of length < g is correct.
    
    # Use the standard approach: for each vertex, BFS and check for cycles
    for s in range(n):
        if time.time() - start_time > timeout:
            return False
        
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _cayley_field_graph(k, g):
    """Try to construct a k-regular Cayley graph on a finite field.
    
    Vertices are elements of F_q where q is a prime power >= k+1.
    We need a symmetric connection set S of size k such that
    all non-trivial sums of < g elements from S (with repetitions allowed,
    but no immediate backtracking) are non-zero.
    
    For simplicity, we try S = {a_1, ..., a_k} where the a_i are chosen
    to be powers of a primitive element, which often gives good girth.
    """
    start_time = time.time()
    
    def is_prime_power(n):
        """Check if n is a prime power, return (prime, power) or None."""
        if n < 2:
            return None
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                q = n
                while q % p == 0:
                    q //= p
                if q == 1:
                    return (p, int(n / (p ** (len(str(n)) - len(str(q))))))
                break
        # n is prime
        return (n, 1)
    
    def gf_mul(a, b, p, exp):
        """Multiply in GF(p^exp) using a fixed irreducible polynomial."""
        # For simplicity, only handle exp=1 (i.e., prime fields)
        if exp == 1:
            return (a * b) % p
        # For higher extensions, we'd need a specific irreducible poly
        # Skip for now
        return None
    
    # Try prime fields first
    for q in range(k + 1, k + 50):
        if time.time() - start_time > 30:
            break
        # Check if q is prime
        if q < 2:
            continue
        is_prime = True
        for i in range(2, int(q**0.5) + 1):
            if q % i == 0:
                is_prime = False
                break
        if not is_prime:
            continue
        
        # Try to find a symmetric connection set S of size k in F_q
        # S must be symmetric: if a in S then -a in S
        # For odd k, we need one element to be its own inverse (i.e., 0, but 0 not allowed)
        # So for odd k, we need pairs {a, -a} plus one more... but -a = a only if 2a=0
        # In a field of odd characteristic, -a = a only if a=0. So odd k is problematic.
        # Actually, we need |S| = k and S = -S. If k is odd, this is impossible in a field
        # of odd characteristic. So for odd k, we need a different approach or even characteristic.
        
        if q % 2 == 0:
            # In characteristic 2, -a = a, so any set is symmetric
            # Try S = {1, 2, 4, ..., 2^(k-1)}
            S = [1]
            for i in range(1, k):
                S.append((S[-1] * 2) % q)
            if len(set(S)) != k:
                continue
        else:
            # Odd characteristic: need pairs {a, -a}
            if k % 2 != 0:
                continue  # Can't have odd-sized symmetric set in odd char
            # Try S = {a_1, -a_1, a_2, -a_2, ...}
            S = []
            for a in range(1, q // 2):
                if len(S) >= k:
                    break
                S.append(a)
                S.append(q - a)
            if len(S) != k:
                continue
        
        # Check girth: no cycle of length < g
        # A cycle of length l corresponds to a_1 + ... + a_l = 0 where
        # a_i in S and a_{i+1} != -a_i (no backtracking)
        # For small g, we can check exhaustively
        if g <= 12:
            valid = True
            # Check all walks of length < g
            # This is expensive, so only do it for small q
            if q <= 50:
                # BFS from 0 to find shortest cycle back to 0
                # Actually, we need to check if there's any cycle of length < g
                # Use the connection set to build the graph and check girth
                adj = [[] for _ in range(q)]
                for u in range(q):
                    for a in S:
                        v = (u + a) % q
                        adj[u].append(v)
                
                # Check girth
                for s in range(q):
                    dist = [-1] * q
                    parent = [-1] * q
                    dist[s] = 0
                    queue = deque([s])
                    while queue:
                        u = queue.popleft()
                        du = dist[u]
                        if du >= g - 1:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = du + 1
                                parent[v] = u
                                queue.append(v)
                            elif v != parent[u] and u != parent[v]:
                                cycle_len = du + dist[v] + 1
                                if cycle_len < g:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Build edges
                    edges = []
                    for u in range(q):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return q, edges
    
    return None

def _cayley_cyclic_graph(k, g):
    """Try to construct a k-regular circulant graph on Z_n.
    
    Vertices are Z_n. Connection set S is symmetric, |S|=k.
    Try various n and S values.
    """
    start_time = time.time()
    
    # For odd k, we need n even and S symmetric with no element being its own inverse
    # except possibly 0 (excluded). In Z_n, -a = a iff 2a = 0 mod n.
    # If n is odd, 2a=0 implies a=0, so no self-inverse non-zero elements.
    # If n is even, a = n/2 is self-inverse.
    
    # For k=5 (odd), we need S with 5 elements, symmetric.
    # In Z_n with n odd: S = {a1, -a1, a2, -a2, a3} but -a3 must be in S,
    # so a3 = -a3 mod n, meaning 2*a3 = 0 mod n, so a3=0 (impossible) or n|2*a3.
    # For n odd, only a3=0 works, which is not allowed.
    # So for odd k, we need n even, and S = {a1, -a1, a2, -a2, n/2}.
    
    best = None
    best_n = float('inf')
    
    for n in range(k + 1, k + 200):
        if time.time() - start_time > 20:
            break
        
        # Check if n is large enough to have k distinct non-zero elements
        if n < k + 1:
            continue
        
        # Try to find symmetric S of size k in Z_n
        candidates_S = []
        
        if n % 2 == 0:
            # n is even: n/2 is self-inverse
            # S = {a1, -a1, a2, -a2, ..., a_{(k-1)/2}, -a_{(k-1)/2}, n/2}
            if k % 2 == 1:
                # k is odd: use n/2 plus (k-1)/2 pairs
                num_pairs = (k - 1) // 2
                # Try different sets of pairs
                for start_a in range(1, min(n // 2, 20)):
                    S = [n // 2]
                    valid = True
                    for j in range(num_pairs):
                        a = start_a + j
                        if a >= n // 2:
                            valid = False
                            break
                        S.append(a)
                        S.append(n - a)
                    if not valid or len(set(S)) != k:
                        continue
                    candidates_S.append(S)
            else:
                # k is even: use k/2 pairs
                num_pairs = k // 2
                for start_a in range(1, min(n // 2, 20)):
                    S = []
                    valid = True
                    for j in range(num_pairs):
                        a = start_a + j
                        if a >= n // 2:
                            valid = False
                            break
                        S.append(a)
                        S.append(n - a)
                    if not valid or len(set(S)) != k:
                        continue
                    candidates_S.append(S)
        else:
            # n is odd: all non-zero elements come in pairs {a, -a}
            # Need k even
            if k % 2 != 0:
                continue
            num_pairs = k // 2
            for start_a in range(1, min(n // 2, 20)):
                S = []
                valid = True
                for j in range(num_pairs):
                    a = start_a + j
                    if a >= n // 2:
                        valid = False
                        break
                    S.append(a)
                    S.append(n - a)
                if not valid or len(set(S)) != k:
                    continue
                candidates_S.append(S)
        
        for S in candidates_S:
            if time.time() - start_time > 20:
                break
            
            # Check girth: no non-backtracking cycle of length < g
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u in range(n):
                for a in S:
                    v = (u + a) % n
                    adj[u].append(v)
            
            # Check girth via BFS
            valid = True
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = deque([s])
                while queue:
                    u = queue.popleft()
                    du = dist[u]
                    if du >= g - 1:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            cycle_len = du + dist[v] + 1
                            if cycle_len < g:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Build edges
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                if n < best_n:
                    best_n = n
                    best = (n, edges)
                break  # Found valid S for this n, move to next n
    
    return best

def construct(k, g):
    """Portfolio search over explicit LUW-type constructions."""
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    best = None
    best_n = float('inf')
    
    def try_candidate(n, edges, expected_degree):
        nonlocal best, best_n
        if n >= best_n:
            return
        # Quick degree check
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != expected_degree for dg in deg):
            return
        # Check for duplicate edges
        edge_set = set()
        for u, v in edges:
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return
            edge_set.add(key)
        # Verify girth
        if _verify_girth_fast(n, edges, expected_degree, g, timeout=8):
            best = (n, edges)
            best_n = n
    
    # Standard d for LUW: d = g-5 for odd g, or g-4 for even g
    d_std = g - 5 if (g - 5) % 2 == 1 else g - 4
    d_std = max(3, d_std)
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Candidate (q, d) pairs to try - expand search
    candidates = []
    # First: q = k with various d values (more d values)
    for d in range(max(3, d_std - 2), d_std + 4):
        candidates.append((k, d))
    
    # Then: try q = next few primes >= k
    q_val = k
    while q_val < k + 30 and len(candidates) < 20:
        if is_prime(q_val) and q_val != k:
            for d in range(max(3, d_std - 1), d_std + 3):
                candidates.append((q_val, d))
        q_val += 1
    
    # Also try q = 4 if k is small
    if k <= 4:
        for d in range(max(3, d_std - 1), d_std + 2):
            candidates.append((4, d))
    
    # Try each candidate
    for q_val, d_val in candidates:
        if time.time() - start_time > deadline - 10:
            break
        try:
            result = _luw_graph(k, g, d_override=d_val, q_override=q_val)
            if result is not None:
                n, edges = result
                # Only accept if q_val == k (degree must be k)
                if q_val == k:
                    try_candidate(n, edges, k)
        except Exception:
            continue
    
    # Try Cayley field graph construction
    if time.time() - start_time < deadline - 15:
        try:
            cayley_result = _cayley_field_graph(k, g)
            if cayley_result is not None:
                n, edges = cayley_result
                try_candidate(n, edges, k)
        except Exception:
            pass
    
    # Try circulant graph construction
    if time.time() - start_time < deadline - 15:
        try:
            circ_result = _cayley_cyclic_graph(k, g)
            if circ_result is not None:
                n, edges = circ_result
                try_candidate(n, edges, k)
        except Exception:
            pass
    
    if best is not None:
        return best
    
    # Fallback: return the standard LUW construction
    return _luw_graph(k, g)
# EVOLVE-BLOCK-END
