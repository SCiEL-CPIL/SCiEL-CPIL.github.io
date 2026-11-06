import time
import random
import itertools

_DEADLINE = None

def _k4_found(adj, n):
    """Return a 4-tuple (a,b,c,d) forming a K4, or None. adj = list of int bitmasks."""
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c + 1, n):
                    if (abc >> d) & 1:
                        return (a, b, c, d)
    return None

def _ind6_found(adj, n):
    """Return a 6-tuple forming an independent set, or None. adj = list of int bitmasks."""
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d + 1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                        higher = rest & (~((1 << (e + 1)) - 1))
                        if higher:
                            f = (higher & -higher).bit_length() - 1
                            return (a, b, c, d, e, f)
    return None

def _is_valid(adj, n):
    return _k4_found(adj, n) is None and _ind6_found(adj, n) is None

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _build_35_vertex_graph():
    """
    Construct a 35-vertex (4,6)-Ramsey graph.
    We use a known algebraic construction based on Z_7 x Z_5.
    Vertices are (i, j). Edge if (i2-i1)%7 in QR7 and (j2-j1)%5 != 0.
    QR7 = {1, 2, 4}.
    """
    n = 35
    adj = [0] * n
    S7 = {1, 2, 4}
    
    for i1 in range(7):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(7):
                for j2 in range(5):
                    v2 = i2 * 5 + j2
                    if v1 >= v2:
                        continue
                    di = (i2 - i1) % 7
                    dj = (j2 - j1) % 5
                    if di in S7 and dj != 0:
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
    return adj

def _get_triangles(adj, n):
    """Return a list of tuples (a,b,c) that form triangles in the graph."""
    triangles = []
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            # c must be > b to avoid duplicates
            mask = ab & (~((1 << (b + 1)) - 1))
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                triangles.append((a, b, c))
                mask ^= lsb
    return triangles

def _get_independent_5sets(adj, n):
    """
    Return a list of tuples (a,b,c,d,e) that form independent sets of size 5.
    This is expensive, so we only do it for the base 35 graph once.
    """
    ind5s = []
    allmask = (1 << n) - 1
    # We can use a recursive search or iterative
    # For efficiency, we only need to find *some* if we are doing local search,
    # but for constraint generation, we need ALL.
    # 35 vertices is small enough for a optimized search if done carefully.
    # However, generating ALL ind5s might be slow. 
    # Alternative: Instead of pre-computing all, we can check violations on the fly.
    # But the "Constraint Programming" idea implies pre-computing constraints.
    # Let's try to generate them. If it takes too long, we fall back.
    
    # Optimization: Use bitmasks.
    # An independent set is a set of vertices with no edges between them.
    # This is equivalent to a clique in the complement graph.
    # Complement graph adjacency:
    comp_adj = [0] * n
    for i in range(n):
        # comp_adj[i] has bits set for all j != i where NOT (i,j) in adj
        # allmask without i
        mask = allmask ^ (1 << i)
        comp_adj[i] = mask ^ adj[i]
        
    # Find all 5-cliques in comp_adj
    # This is the same as _k4_found but for size 5 in comp_adj.
    def find_k5_in_comp(comp_adj, n):
        results = []
        for a in range(n):
            aa = comp_adj[a]
            for b in range(a + 1, n):
                if not (aa >> b) & 1:
                    continue
                ab = aa & comp_adj[b]
                for c in range(b + 1, n):
                    if not (ab >> c) & 1:
                        continue
                    abc = ab & comp_adj[c]
                    for d in range(c + 1, n):
                        if not (abc >> d) & 1:
                            continue
                        abcd = abc & comp_adj[d]
                        for e in range(d + 1, n):
                            if (abcd >> e) & 1:
                                results.append((a, b, c, d, e))
        return results

    return find_k5_in_comp(comp_adj, n)

def _solve_vertex_36(adj35, deadline):
    """
    Try to find a valid connection pattern for vertex 36 to the 35 existing vertices.
    Uses Simulated Annealing / Random Walk on the 35-bit mask.
    """
    n_old = 35
    # Pre-compute constraints
    # 1. Triangles in G35: If new vertex connects to all 3, K4 formed.
    triangles = _get_triangles(adj35, n_old)
    # 2. Independent 5-sets in G35: If new vertex connects to NONE of the 5, Ind6 formed.
    ind5s = _get_independent_5sets(adj35, n_old)
    
    # Safety: If constraints are too numerous, skip extension to avoid timeout
    if len(ind5s) > 50000 or len(triangles) > 50000:
        return None
        
    # If there are no constraints, any mask works.
    if not triangles and not ind5s:
        return (1 << n_old) - 1 # Connect to all (arbitrary choice)

    def count_violations(mask):
        v_count = 0
        # Check K4 violations: mask has 1s at all vertices of a triangle
        for tri in triangles:
            a, b, c = tri
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                v_count += 1
        # Check Ind6 violations: mask has 0s at all vertices of an ind5
        for ind in ind5s:
            a, b, c, d, e = ind
            # If mask has 0 at a, b, c, d, e
            # Equivalent to: mask & (1<<a | 1<<b | 1<<c | 1<<d | 1<<e) == 0
            mask_ind = (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e)
            if (mask & mask_ind) == 0:
                v_count += 1
        return v_count

    # Simulated Annealing
    # Start with a random mask
    random.seed(42)
    current_mask = 0
    for i in range(n_old):
        if random.random() < 0.5:
            current_mask |= (1 << i)
    
    current_violations = count_violations(current_mask)
    if current_violations == 0:
        return current_mask

    best_mask = current_mask
    best_violations = current_violations
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    steps = 0
    max_steps = 500000 # High limit, but we check time
    
    while T > min_T:
        if time.time() > deadline:
            break
        # Random move: flip a random bit
        i = random.randint(0, n_old - 1)
        new_mask = current_mask ^ (1 << i)
        new_violations = count_violations(new_mask)
        
        delta = new_violations - current_violations
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_mask = new_mask
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_mask = current_mask
                if best_violations == 0:
                    return best_mask
        T *= cooling
        steps += 1
        if steps > max_steps:
            break
            
    return best_mask if best_violations == 0 else None

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 110  # Safety margin
    
    # 1. Build 35-vertex graph
    adj35 = _build_35_vertex_graph()
    
    # Verify 35-vertex graph
    if not _is_valid(adj35, 35):
        # Fallback: Try to fix or use a smaller known valid graph
        # For now, if the algebraic one is invalid, we try a quick local search on it
        # to see if we can make it valid.
        random.seed(123)
        for _ in range(1000):
            if time.time() > _DEADLINE: break
            k4 = _k4_found(adj35, 35)
            if k4:
                a,b,c,d = k4
                # Remove a random edge
                u,v = random.choice([(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)])
                adj35[u] &= ~(1 << v)
                adj35[v] &= ~(1 << u)
            else:
                ind = _ind6_found(adj35, 35)
                if ind:
                    a,b,c,d,e,f = ind
                    # Add a random edge
                    pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
                    u,v = random.choice(pairs)
                    adj35[u] |= (1 << v)
                    adj35[v] |= (1 << u)
                else:
                    break
                    
        if not _is_valid(adj35, 35):
            # If still invalid, return a smaller known valid graph
            # E.g., Paley 17
            squares = {x * x % 17 for x in range(1, 17)}
            return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

    # 2. Try to extend to 36
    # The constraint solving approach
    mask36 = _solve_vertex_36(adj35, _DEADLINE)
    
    if mask36 is not None:
        # Construct the 36-vertex graph
        adj36 = adj35[:] + [0]
        for i in range(35):
            if (mask36 >> i) & 1:
                adj36[35] |= (1 << i)
                adj36[i] |= (1 << 35)
        
        # Final verification (paranoia)
        if _is_valid(adj36, 36):
            return {'n': 36, 'edges': _edges_from_adj(adj36, 36)}
        else:
            # If verification fails due to some bug in constraint logic, 
            # fall back to 35
            pass

    # Return 35-vertex graph
    return {'n': 35, 'edges': _edges_from_adj(adj35, 35)}
