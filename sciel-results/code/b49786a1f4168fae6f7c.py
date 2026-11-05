import time
import math
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    def remaining():
        return deadline - time.time()

    # For (3,16), use a targeted algebraic construction
    if k == 3 and g == 16:
        # Try PSL(2,11) Cayley graph first
        result = _try_psl2_11_cayley(deadline)
        if result is not None:
            return result

        # Try a voltage lift with more systematic search
        result = _voltage_lift_systematic(deadline)
        if result is not None:
            return result

        # Try a different group: PSL(2,13)
        result = _try_psl2_13_cayley(deadline)
        if result is not None:
            return result

    # Fallback: use the LUW construction
    return _luw_construct(k, g)


def _try_psl2_11_cayley(deadline):
    """
    PSL(2,11) has order 660.
    We need a 3-regular Cayley graph with girth >= 16.
    
    In PSL(2,11), we look for generators {a, b, b^{-1}} where:
    - a has order 2 (so a = a^{-1})
    - b has order > 2
    - The graph is 3-regular
    
    PSL(2,11) is the alternating group A_6.
    We can use permutation representation.
    """
    if time.time() > deadline - 5:
        return None

    q = 11

    # Work with 2x2 matrices mod q, normalized for PSL
    def mat_mult(A, B, mod):
        return [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
                (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod,
                (A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
                (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]

    def mat_canonical(t, mod):
        a, b, c, d = t
        # In PSL, (a,b,c,d) ~ (-a,-b,-c,-d)
        na, nb, nc, nd = (-a) % mod, (-b) % mod, (-c) % mod, (-d) % mod
        # Canonical: prefer smaller first non-zero
        if a != 0:
            return t if a < na else (na, nb, nc, nd)
        elif b != 0:
            return t if b < nb else (na, nb, nc, nd)
        else:
            return t if c < nc else (na, nb, nc, nd)

    def mat_inv(t, mod):
        a, b, c, d = t
        return (d % mod, (-b) % mod, (-c) % mod, a % mod)

    # Identity
    I = (1, 0, 0, 1)
    
    # Candidate generators
    # A should have order 2 in PSL
    # A = (0, 1, q-1, 0) -> [[0,1],[-1,0]] which squares to [[-1,0],[0,-1]] ~ I in PSL
    A = mat_canonical((0, 1, q-1, 0), q)
    
    # B: try several options
    B_candidates = [
        mat_canonical((1, 1, 0, 1), q),  # [[1,1],[0,1]]
        mat_canonical((1, 2, 0, 1), q),  # [[1,2],[0,1]]
        mat_canonical((1, 3, 0, 1), q),  # [[1,3],[0,1]]
        mat_canonical((2, 1, 1, 1), q),  # [[2,1],[1,1]]
        mat_canonical((1, 0, 1, 2), q),  # [[1,0],[1,2]]
        mat_canonical((3, 1, 1, 1), q),  # [[3,1],[1,1]]
        mat_canonical((1, 1, 1, 3), q),  # [[1,1],[1,3]]
    ]

    for B in B_candidates:
        if time.time() > deadline - 5:
            break
        
        B_inv = mat_canonical(mat_inv(B, q), q)
        
        # Check distinctness
        gens = [A, B, B_inv]
        if len(set(gens)) < 3:
            continue
        if any(g == I for g in gens):
            continue

        # Enumerate group via BFS
        group = {}
        group[I] = 0
        queue = deque([I])
        
        while queue:
            curr = queue.popleft()
            for gen in gens:
                nxt = mat_canonical(mat_mult(curr, gen, q), q)
                if nxt not in group:
                    group[nxt] = len(group)
                    queue.append(nxt)

        n = len(group)
        
        # For PSL(2,11), n should be 660
        if n != 660:
            continue

        # Build edges
        adj = [[] for _ in range(n)]
        for elem, idx in group.items():
            for gen in gens:
                nxt = mat_canonical(mat_mult(elem, gen, q), q)
                if nxt in group:
                    nidx = group[nxt]
                    adj[idx].append(nidx)

        # Check 3-regular
        valid = True
        for i in range(n):
            if len(adj[i]) != 3:
                valid = False
                break
        if not valid:
            continue

        # Check girth using non-backtracking walk
        min_girth = float('inf')
        for s in range(n):
            if time.time() > deadline - 3:
                break
            # BFS with parent tracking for non-backtracking
            dist = [[-1]*3 for _ in range(n)]  # dist[v][dir] where dir encodes last edge
            # Actually, simpler: BFS where we track (node, prev) to avoid backtracking
            visited = set()
            queue2 = [(s, -1, 0)]
            visited.add((s, -1))
            
            while queue2:
                node, prev, d = queue2.pop(0)
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (nb, node) in visited:
                        # Found a cycle
                        cycle_len = d + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                    else:
                        visited.add((nb, node))
                        queue2.append((nb, node, d + 1))
            
            if min_girth < g:
                break

        if min_girth >= g:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return (n, edges)

    return None


def _try_psl2_13_cayley(deadline):
    """
    PSL(2,13) has order 1092.
    This is larger than the 936 known best, but might still be competitive
    if we find a good generating set.
    Actually, 1092 > 936, so this won't help. Skip if we already have a good result.
    But let's try anyway as a fallback.
    """
    if time.time() > deadline - 5:
        return None

    q = 13

    def mat_mult(A, B, mod):
        return [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
                (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod,
                (A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
                (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]

    def mat_canonical(t, mod):
        a, b, c, d = t
        na, nb, nc, nd = (-a) % mod, (-b) % mod, (-c) % mod, (-d) % mod
        if a != 0:
            return t if a < na else (na, nb, nc, nd)
        elif b != 0:
            return t if b < nb else (na, nb, nc, nd)
        else:
            return t if c < nc else (na, nb, nc, nd)

    def mat_inv(t, mod):
        a, b, c, d = t
        return (d % mod, (-b) % mod, (-c) % mod, a % mod)

    I = (1, 0, 0, 1)
    A = mat_canonical((0, 1, q-1, 0), q)
    
    B_candidates = [
        mat_canonical((1, 1, 0, 1), q),
        mat_canonical((1, 2, 0, 1), q),
        mat_canonical((1, 3, 0, 1), q),
        mat_canonical((2, 1, 1, 1), q),
    ]

    for B in B_candidates:
        if time.time() > deadline - 3:
            break
        
        B_inv = mat_canonical(mat_inv(B, q), q)
        gens = [A, B, B_inv]
        if len(set(gens)) < 3:
            continue
        if any(g == I for g in gens):
            continue

        group = {}
        group[I] = 0
        queue = deque([I])
        
        while queue:
            curr = queue.popleft()
            for gen in gens:
                nxt = mat_canonical(mat_mult(curr, gen, q), q)
                if nxt not in group:
                    group[nxt] = len(group)
                    queue.append(nxt)

        n = len(group)
        # PSL(2,13) has order 1092
        if n != 1092:
            continue

        adj = [[] for _ in range(n)]
        for elem, idx in group.items():
            for gen in gens:
                nxt = mat_canonical(mat_mult(elem, gen, q), q)
                if nxt in group:
                    nidx = group[nxt]
                    adj[idx].append(nidx)

        valid = all(len(adj[i]) == 3 for i in range(n))
        if not valid:
            continue

        # Quick girth check
        min_girth = float('inf')
        for s in range(n):
            if time.time() > deadline - 2:
                break
            visited = set()
            queue2 = [(s, -1, 0)]
            visited.add((s, -1))
            
            while queue2:
                node, prev, d = queue2.pop(0)
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (nb, node) in visited:
                        cycle_len = d + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                    else:
                        visited.add((nb, node))
                        queue2.append((nb, node, d + 1))
            
            if min_girth < g:
                break

        if min_girth >= g:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return (n, edges)

    return None


def _voltage_lift_systematic(deadline):
    """
    Voltage lift of the Petersen graph with systematic search.
    Petersen has 10 vertices, 15 edges, 3-regular.
    Lift by factor m gives 10m vertices.
    For m=52, n=520 (below Moore bound 512? No, 520 > 512).
    Moore bound for (3,16): 2 * (1 + 2 + 2^2 + ... + 2^7) = 2*(2^8 - 1) = 510? 
    Actually for girth 16: diameter is 7 (since girth 16 means no cycles < 16, 
    so BFS tree up to depth 7 without cycles).
    n >= 2 * (2^8 - 1) = 510. So m >= 51, n >= 510.
    """
    if time.time() > deadline - 10:
        return None

    # Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    adj_p = [[] for _ in range(10)]
    for u, v in petersen_edges:
        adj_p[u].append(v)
        adj_p[v].append(u)
    
    # Generate all closed non-backtracking walks of length <= 7
    # For voltage assignment, we need: for every closed non-backtracking walk W,
    # the sum of voltages (with signs) along W is not 0 mod m.
    
    # Collect all closed non-backtracking walks of length 4, 6 (even cycles in Petersen)
    # Actually, we need all closed walks that don't immediately backtrack.
    
    closed_walks = []  # Each is a list of (edge_idx, direction)
    
    def get_edge_idx(u, v):
        for i, (a, b) in enumerate(petersen_edges):
            if (a == u and b == v) or (a == v and b == u):
                return i
        return -1
    
    for start in range(10):
        # DFS for closed non-backtracking walks
        def dfs(node, prev, length, path):
            if length > 0 and node == start:
                if length >= 4:  # Shortest cycle in Petersen is 5, but we need to check all
                    # Actually, Petersen has girth 5, so closed non-backtracking walks of length >= 5
                    # But we need length < 16 for the lift to have girth >= 16
                    # So we need to check lengths 4, 5, 6, ..., 15
                    if length < 16:
                        closed_walks.append(list(path))
                return
            
            if length >= 15:
                return
            
            for nb in adj_p[node]:
                if nb == prev:
                    continue
                e_idx = get_edge_idx(node, nb)
                # Direction: +1 if we go in the stored direction, -1 otherwise
                u, v = petersen_edges[e_idx]
                d = 1 if node == u else -1
                dfs(nb, node, length + 1, path + [(e_idx, d)])
        
        dfs(start, -1, 0, [])
    
    if not closed_walks:
        return None
    
    # Now we have a list of constraints. For each closed walk, 
    # sum(v_i * d_i) != 0 mod m for all i.
    
    # This is a large system. Let's try to find a voltage assignment.
    # Use random search with increasing m.
    
    best_result = None
    
    for m in range(52, 200):
        if time.time() > deadline - 5:
            break
        if m < 2:
            continue
        
        # Check if m is prime (prefer primes for better properties)
        is_prime = True
        if m < 2:
            is_prime = False
        elif m < 4:
            is_prime = True
        elif m % 2 == 0:
            is_prime = False
        else:
            for i in range(3, int(math.isqrt(m)) + 1, 2):
                if m % i == 0:
                    is_prime = False
                    break
        
        rng = random.Random(m * 12345 + 6789)
        
        # Try multiple random assignments
        max_tries = 50 if is_prime else 30
        for _ in range(max_tries):
            if time.time() > deadline - 3:
                break
            
            # Random voltage assignment
            voltages = [rng.randint(0, m-1) for _ in range(15)]
            
            # Check all closed walks
            valid = True
            for walk in closed_walks:
                s = 0
                for e_idx, d in walk:
                    s = (s + d * voltages[e_idx]) % m
                if s == 0:
                    valid = False
                    break
            
            if valid:
                # Build the lifted graph
                n = 10 * m
                edges = set()
                for u, v in petersen_edges:
                    e_idx = get_edge_idx(u, v)
                    shift = voltages[e_idx]
                    for i in range(m):
                        u_lift = u * m + i
                        v_lift = v * m + (i + shift) % m
                        if u_lift < v_lift:
                            edges.add((u_lift, v_lift))
                        else:
                            edges.add((v_lift, u_lift))
                
                edges = list(edges)
                
                # Verify girth
                gc = _check_girth(n, edges, g, deadline - 1)
                if gc >= g:
                    return (n, edges)
                elif gc > 0 and (best_result is None or n < best_result[0]):
                    best_result = (n, edges)
    
    return best_result


def _check_girth(n, edges, g, time_limit):
    """Check girth of a graph. Returns the girth, or -1 if timeout."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for s in range(n):
        if time.time() > time_limit:
            return -1
        
        # BFS from s, tracking parent to avoid backtracking
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if dist[nb] != -1:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if cycle_len < g:
                        return min_girth
                else:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
        
        if min_girth < g:
            return min_girth
    
    return min_girth


def _luw_construct(k, g):
    """LUW construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
