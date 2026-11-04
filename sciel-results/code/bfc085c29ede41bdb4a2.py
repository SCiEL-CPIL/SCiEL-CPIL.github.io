"""ABS-style Cayley graph construction for the cage problem n(k, g)."""
import numpy as np
from collections import deque

def _next_prime_ge(n):
    """Return the smallest prime >= n."""
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    p = max(2, n)
    while not is_prime(p):
        p += 1
    return p

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([(start, -1)])
        while queue:
            node, parent = queue.popleft()
            if dist[node] >= g - 1:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append((nb, node))
                elif dist[nb] <= dist[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
    return True

def _build_cayley_ut(m, q, gen_list):
    """
    Build Cayley graph of UT_m(q) with given generators.
    gen_list: list of (i, a) meaning generator E_{i,i+1}(a)
    Returns (n, edges)
    """
    n_exp = m * (m - 1) // 2
    n = q ** n_exp
    
    if n > 500000:
        return None, None
    
    # Index mapping: (i,j) -> flat index, i < j
    # Row-major: for i in 0..m-2, for j in i+1..m-1
    idx_map = {}
    flat = 0
    for i in range(m - 1):
        for j in range(i + 1, m):
            idx_map[(i, j)] = flat
            flat += 1
    
    # Precompute: for each generator (i, a), which entries are affected
    # Right mult by E_{i,i+1}(a): 
    #   x[i,i+1] += a
    #   x[i,j] += a * x[i+1,j] for j > i+1
    # So we need to know, for each gen, the source indices and dest indices
    
    gen_info = []
    for (i, a) in gen_list:
        # dest: (i, i+1)
        dest_base = idx_map[(i, i+1)]
        # sources: (i+1, j) for j > i+1, added to (i, j)
        srcs = []
        for j in range(i + 2, m):
            srcs.append((idx_map[(i+1, j)], idx_map[(i, j)]))
        gen_info.append((dest_base, a, srcs))
    
    # Build neighbor table
    # For each element (as flat index 0..n-1), compute neighbors
    # Element flat index encodes: x[0] + q*x[1] + q^2*x[2] + ...
    
    # Precompute powers of q
    q_pows = [1] * n_exp
    for i in range(1, n_exp):
        q_pows[i] = q_pows[i-1] * q
    
    def flat_to_entries(flat_idx):
        entries = [0] * n_exp
        rem = flat_idx
        for i in range(n_exp - 1, -1, -1):
            entries[i] = rem % q
            rem //= q
        return entries
    
    def apply_gen(entries, gen_idx):
        dest_base, a, srcs = gen_info[gen_idx]
        new_entries = list(entries)
        new_entries[dest_base] = (new_entries[dest_base] + a) % q
        for src, dst in srcs:
            new_entries[dst] = (new_entries[dst] + a * new_entries[src]) % q
        return new_entries
    
    def entries_to_flat(entries):
        f = 0
        for i in range(n_exp):
            f += entries[i] * q_pows[i]
        return f
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for flat_idx in range(n):
        entries = flat_to_entries(flat_idx)
        for gi in range(len(gen_list)):
            new_entries = apply_gen(entries, gi)
            new_flat = entries_to_flat(new_entries)
            adj[flat_idx].append(new_flat)
    
    # Build edges (undirected, no duplicates)
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    return n, edges

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using the ABS construction:
    Cayley graph on UT_m(q) with elementary matrix generators.
    """
    # For even k: use UT_{k/2+1}(k) with generators E_{i,i+1}(±1)
    # For odd k: use UT_m(q) where q >= k, find suitable generator set
    
    if k % 2 == 0:
        m = k // 2 + 1
        q = k
        # Check q is prime
        q = _next_prime_ge(k)
        if q != k:
            # If k is not prime, we can't use GF(k) directly
            # Fall back to LUW
            return _luw_fallback(k, g)
        
        gen_list = []
        for i in range(m - 1):
            gen_list.append((i, 1))
            gen_list.append((i, q - 1))  # -1
        
        n, edges = _build_cayley_ut(m, q, gen_list)
        if n is None:
            return _luw_fallback(k, g)
        
        # Verify degree
        if len(edges) != n * k // 2:
            return _luw_fallback(k, g)
        
        # Verify girth
        if _verify_girth(n, edges, g):
            return n, edges
        else:
            return _luw_fallback(k, g)
    else:
        # Odd k: try UT_m(q) with q = k (if prime) or next prime
        q = _next_prime_ge(k)
        
        # Try different m values and generator sets
        best_result = None
        
        for m in range(3, 7):
            n_exp = m * (m - 1) // 2
            n = q ** n_exp
            if n > 200000:
                break
            
            # Try various generator sets of size k
            # Strategy: use E_{i,i+1}(a) for various i and a
            # We need exactly k generators, symmetric (if a in set, -a in set)
            # For odd k, we need an odd number of generators
            
            # Option 1: use all a in {±1, ±2, ..., ±(k-1)/2} for one i
            # That gives k-1 generators, add one more from another i
            # But must be symmetric... 
            # Actually for odd k, we can use: E_{0,1}(a) for a in {1,...,k-1} \ {0}
            # That's k-1 generators, not symmetric unless we pair them
            # E_{0,1}(a) and E_{0,1}(-a) must both be present for symmetry
            # So k-1 must be even, meaning k must be odd -> k-1 is even. Good.
            # But k-1 generators, not k.
            
            # Better: use E_{0,1}(a) for a in {±1, ±2} (4 gens) + E_{1,2}(1) (1 gen) = 5 for k=5
            # Check symmetry: E_{0,1}(1) <-> E_{0,1}(-1), E_{0,1}(2) <-> E_{0,1}(-2), E_{1,2}(1) <-> E_{1,2}(-1)
            # Wait, E_{1,2}(1) inverse is E_{1,2}(-1), so we need both. That's 6 gens.
            # For odd k, we can't have a fully symmetric set of odd size unless one generator is self-inverse.
            # E_{i,i+1}(a) has order q, so it's self-inverse only if a = 0 or 2a = 0 mod q, i.e., a = q/2 for even q.
            # For odd q, no non-trivial self-inverse generators.
            # So for odd k and odd q, we can't have a symmetric generating set of odd size!
            
            # This means for odd k, we should use even q or a different approach.
            # Let's try q = 2*k or next even prime power...
            # Actually, let's just use the LUW fallback for odd k as it's more reliable.
            break
        
        return _luw_fallback(k, g)


def _luw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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
