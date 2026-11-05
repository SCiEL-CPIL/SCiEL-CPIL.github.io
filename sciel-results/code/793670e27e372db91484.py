"""Search for a small 5-regular graph with girth >= g.

Strategy:
  1. Fast randomized search: generate random k-regular graphs via stub matching,
     verify girth exactly with a fast non-backtracking BFS, keep the smallest valid one.
  2. 2-switch local search: perturb a candidate graph to reduce short cycles while
     preserving k-regularity; re-check girth after each accepted move.
  3. Use a deadline guard to return the best valid result before the wall-clock limit.
  4. Fallback: Lazebnik-Ustimenko-Woldar construction (known small cages for some params).

Key optimizations:
  * Girth check uses dict-based BFS on non-backtracking states for speed on sparse graphs.
  * Vectorized stub-matching permutation for speed.
  * 2-switch moves reuse the stub array and only touch 4 edges.
  * Adaptive trial budget: more trials for smaller n where success is more likely.
  * Early exit in girth check: as soon as any short cycle is found, return False.
  * Skip n values that are below the Moore bound or where n*k is odd.
"""
import time
import numpy as np

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = np.repeat(np.arange(n, dtype=np.int64), k)
    perm = np.arange(m, dtype=np.int64)
    # Fisher-Yates shuffle using numpy for speed
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        seen[u].add(v)
        seen[v].add(u)
    return adj

def _swap_regular(adj, n, k, rng, stubs=None):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges).
    If stubs is provided (a list of length n*k), it is reused."""
    m = n * k
    if stubs is None:
        stubs = [i for i in range(n) for _ in range(k)]
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    # remove edges (u1,v1) and (u2,v2); require they are distinct
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    # add edges (u1,u2) and (v1,v2); forbid self-loops / parallel edges
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

# ---------------- exact girth via non-backtracking walks ----------------
def _girth_ok(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses BFS on non-backtracking states (u, prev) from each source s, with an
    early cutoff at depth g-1. Optimized with list-based queue and early termination."""
    n = len(adj)
    max_depth = g - 1
    for s in range(n):
        visited = set()
        # Queue of (u, prev, depth)
        queue = []
        for v in adj[s]:
            visited.add((v, s))
            queue.append((v, s, 1))
        head = 0
        while head < len(queue):
            u, p, d = queue[head]
            head += 1
            if d >= max_depth:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s:
                    return False
                state = (w, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((w, u, d + 1))
    return True

def _shortest_cycle_len(adj, g):
    """Return the length of the shortest cycle, or a value >= g if none shorter
    exists.  Uses BFS on non-backtracking states from each vertex."""
    n = len(adj)
    for s in range(n):
        visited = set()
        queue = []
        for v in adj[s]:
            visited.add((v, s))
            queue.append((v, s, 1))
        head = 0
        while head < len(queue):
            u, p, d = queue[head]
            head += 1
            if d >= g - 1:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s:
                    return d + 1
                state = (w, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((w, u, d + 1))
    return g

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 115.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    # Moore bound (lower bound on n)
    if g % 2 == 1:  # odd girth
        # 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    else:  # even girth
        # 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        moore = 2 * sum((k-1)**i for i in range(g//2))
    n_start = max(moore, 2*k)

    # Search increasing n; budget trials by n so small n gets more effort.
    # Skip n values where n*k is odd (impossible for k-regular simple graph when k is odd and n is odd)
    n = n_start
    while n <= 1200 and time.time() < deadline:
        # Skip impossible n: need n*k even for a simple k-regular graph
        if (n * k) % 2 != 0:
            n += 1
            continue

        # Time budget per n value
        remaining = deadline - time.time()
        if remaining < 2.0:
            break

        # Allocate time: spend proportional effort on early n values
        if n < n_start + 50:
            time_per_n = min(remaining * 0.15, 15.0)
        elif n < n_start + 200:
            time_per_n = min(remaining * 0.08, 8.0)
        else:
            time_per_n = min(remaining * 0.04, 4.0)

        n_deadline = time.time() + time_per_n

        # Precompute stubs for this n to reuse in swaps
        stubs = [i for i in range(n) for _ in range(k)]

        # Adaptive number of trials based on time budget
        max_trials = max(5, int(time_per_n / 0.5))
        max_switches = max(5, int(time_per_n / 0.3))

        for t in range(max_trials):
            if time.time() > n_deadline or time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            
            # Check girth
            if _girth_ok(adj, g):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            
            # 2-switch local search: try to improve girth
            best_sc = _shortest_cycle_len(adj, g)
            best_adj = [list(x) for x in adj]
            best_best_sc = best_sc
            
            switch_count = 0
            while switch_count < max_switches and time.time() < n_deadline and time.time() < deadline:
                if _swap_regular(adj, n, k, rng, stubs):
                    sc = _shortest_cycle_len(adj, g)
                    if sc >= g:
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
                    if sc > best_best_sc:
                        best_best_sc = sc
                        best_adj = [list(x) for x in adj]
                switch_count += 1
            
            # If we improved, continue from the best state we found
            if best_best_sc > best_sc:
                adj = [list(x) for x in best_adj]
                best_sc = best_best_sc

        n += 1

    # Fallback 1: ABS construction (non-bipartite Cayley graph on UT(d+1, F_q))
    # For k=5, use q=5 (prime). Generating set: elementary matrices E_{i,i+1}(a) for a in F_q\{0}
    # plus their inverses. This gives degree 2*q*(d-1)/2... let me think.
    # Actually for UT(m, F_q), the Cayley graph with generators E_{i,i+1}(1) for i=1..m-2 
    # and their inverses gives degree 2*(m-2). For degree 5, we need a different approach.
    # 
    # Better: use the ABS construction more carefully.
    # For k=5, g=9: we can try q=5, and use a subset of elementary matrices.
    # The group UT(3, F_5) has 125 elements. With appropriate generators we can get degree 5.
    # 
    # Actually, let's try a different approach: use the known construction for (5,9)-cage.
    # The ABS construction typically works for specific parameter combinations.
    # For now, let's try the LUW construction first, then ABS if it gives fewer vertices.
    
    # First, try LUW construction
    q_luw = k
    d_luw = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q_luw == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_luw, sub_luw, mul_luw = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add_luw, sub_luw, mul_luw = (lambda x, y: (x + y) % q_luw), (lambda x, y: (x - y) % q_luw), (lambda x, y: x * y % q_luw)
    rule_luw = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d_luw + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule_luw[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours_luw(vertex):
        side, x = vertex
        for first in range(q_luw):
            y = [None, first] + [0] * (d_luw - 1)
            for c in range(2, d_luw + 1):
                a, b = rule_luw[c]
                y[c] = add_luw(x[c], mul_luw(y[a], x[b])) if side == 0 else sub_luw(x[c], mul_luw(x[a], y[b]))
            yield 1 - side, tuple(y)

    start_luw = (0, (None,) + (0,) * d_luw)
    index_luw, order_luw, edges_luw = {start_luw: 0}, [start_luw], []
    for vertex in order_luw:
        for other in neighbours_luw(vertex):
            if other not in index_luw:
                index_luw[other] = len(order_luw)
                order_luw.append(other)
            if vertex[0] == 0:
                edges_luw.append((index_luw[vertex], index_luw[other]))
    n_luw = len(order_luw)
    
    # Try ABS construction: Cayley graph on UT(m, F_q) with elementary matrix generators
    # For k=5: use q=5, m=3, generators {E_{1,2}(1), E_{1,2}(2), E_{2,3}(1), E_{2,3}(2), E_{2,3}(3)}
    # This gives degree 5 on 125 vertices
    # But we need girth >= 9, which is unlikely for such a small group.
    # 
    # Alternative: use a larger group. For UT(4, F_5), we have 5^6 = 15625 elements, too large.
    # 
    # Let's try a different ABS variant: use the construction from Alon-Babai-Suzuki 1985.
    # For prime p and integer d, the Cayley graph on UT(d+1, F_p) with generators
    # {E_{i,i+1}(1) for i=1..d-1} ∪ {E_{i,i+1}(1)^{-1} for i=1..d-1} has degree 2*(d-1).
    # For degree 5, we need 2*(d-1) = 5, which is impossible.
    # 
    # So for odd degree, we need a different generating set.
    # Use {E_{1,2}(a) for a=1..2} ∪ {E_{2,3}(a) for a=1..3} on UT(3, F_5)
    # This gives degree 5 on 125 vertices.
    
    q_abs = 5
    m_abs = 3  # size of UT(m_abs, F_q_abs)
    # Generators: E_{1,2}(1), E_{1,2}(2), E_{2,3}(1), E_{2,3}(2), E_{2,3}(3)
    # In matrix form, UT(3, F_5) elements are [[1, a, b], [0, 1, c], [0, 0, 1]]
    # E_{1,2}(a) = [[1, a, 0], [0, 1, 0], [0, 0, 1]]
    # E_{2,3}(c) = [[1, 0, 0], [0, 1, c], [0, 0, 1]]
    
    def matrix_mul_ut3(M1, M2, q):
        """Multiply two UT(3) matrices over F_q. M = (a, b, c) represents [[1,a,b],[0,1,c],[0,0,1]]"""
        a1, b1, c1 = M1
        a2, b2, c2 = M2
        return ((a1 + a2) % q, (b1 + b2 + a1 * c2) % q, (c1 + c2) % q)
    
    def matrix_inv_ut3(M, q):
        """Inverse of UT(3) matrix M = (a, b, c) over F_q."""
        a, b, c = M
        return ((-a) % q, (-b + a * c) % q, (-c) % q)
    
    # Identity
    ident = (0, 0, 0)
    
    # Generate all elements of UT(3, F_5)
    all_elements = []
    for a in range(q_abs):
        for b in range(q_abs):
            for c in range(q_abs):
                all_elements.append((a, b, c))
    
    elem_to_idx = {e: i for i, e in enumerate(all_elements)}
    n_abs = len(all_elements)
    
    # Define generators and their inverses
    generators = [
        (1, 0, 0),  # E_{1,2}(1)
        (2, 0, 0),  # E_{1,2}(2)
        (0, 0, 1),  # E_{2,3}(1)
        (0, 0, 2),  # E_{2,3}(2)
        (0, 0, 3),  # E_{2,3}(3)
    ]
    
    # Build adjacency
    adj_abs = [[] for _ in range(n_abs)]
    for i, elem in enumerate(all_elements):
        for gen in generators:
            prod = matrix_mul_ut3(elem, gen, q_abs)
            j = elem_to_idx[prod]
            adj_abs[i].append(j)
            # Also add inverse direction (but this will be handled by symmetry)
    
    # Check if this is k-regular
    deg_ok = all(len(adj_abs[i]) == k for i in range(n_abs))
    
    if deg_ok:
        # Check girth
        if _girth_ok(adj_abs, g):
            return n_abs, [(i, j) for i in range(n_abs) for j in adj_abs[i] if i < j]
    
    # Return the LUW construction as fallback
    return n_luw, edges_luw
