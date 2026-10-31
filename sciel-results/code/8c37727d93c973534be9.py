"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using Cayley graphs of
    finite Heisenberg groups (Ramanujan-Margulis type construction).
    
    For k=3, g=16: use the Heisenberg group H_3 over GF(p) with generators
    {(1,0,0), (0,1,0), (0,0,1)} appropriately modified to form a 3-element
    symmetric generating set.
    
    The Heisenberg group H_3(F_p) has elements (a,b,c) with multiplication:
      (a,b,c) * (a',b',c') = (a+a', b+b', c+c' + a*b')
    It has p^3 elements. We need a 3-element symmetric generating set S
    such that the Cayley graph has girth >= g.
    """
    deadline = time.time() + 120.0
    
    if k != 3:
        # Fallback: use the original LUW construction for other k values
        return _luw_construct(k, g)
    
    # For k=3, g=16: try Heisenberg group constructions
    # The group H_3(F_p) = {(a,b,c) : a,b,c in F_p} with
    # (a,b,c)(a',b',c') = (a+a', b+b', c+c' + a*b')
    # 
    # We need a symmetric generating set S of size 3 (since k=3, the graph
    # is 3-regular, so |S| = 3 and S = S^{-1}, e not in S).
    #
    # For the Cayley graph to be 3-regular, we need exactly 3 generators
    # (counting each generator and its inverse as one undirected edge).
    # So S should have 3 elements where each element's inverse is also in S.
    # This means either:
    #   - S = {s, s^{-1}, t} where t = t^{-1} (order 2 element), or
    #   - S has 3 elements each paired with their inverse (but that's 6 elements)
    #
    # For a 3-regular undirected Cayley graph, we need |S| = 3 where S = S^{-1}.
    # This requires: one element of order 2 (self-inverse) and one pair {s, s^{-1}}.
    # So S = {a, b, c} where a = a^{-1}, and {b, c} = {b, b^{-1}}.
    
    # Let's try: generators in H_3(F_p)
    # e1 = (1, 0, 0), e2 = (0, 1, 0), e3 = (0, 0, 1)
    # 
    # In H_3, [e1, e2] = e3 (the commutator)
    # e1 has order p, e2 has order p, e3 has order p
    # 
    # For p=3: elements have orders dividing 3.
    # e1^3 = e, so e1^{-1} = e1^2 = (2,0,0)
    # e2^{-1} = (0,2,0)
    # e3^{-1} = (0,0,2)
    #
    # We need S = {s1, s2, s3} with S = S^{-1}, |S|=3.
    # Option: s1 of order 2, s2 and s3 inverses of each other.
    # But in H_3(F_3), no element has order 2 (since 2 doesn't divide 3).
    # 
    # For p=5: elements can have order 5. Still no order 2.
    #
    # Actually, in H_3(F_p) for odd p, the center is {e3} and has order p.
    # No element has order 2 if p is odd.
    #
    # Alternative: Use a different group or a different approach.
    #
    # Let me reconsider. For a 3-regular Cayley graph, we need a generating set S
    # of size 3 with S = S^{-1}. If no element has order 2, we can't do this
    # with 3 elements. We'd need |S| = 4 (two pairs of inverses) for a 4-regular graph.
    #
    # So for k=3, the Heisenberg group approach needs an element of order 2.
    # Let's use H_3(F_p) × Z_2 or a different nilpotent group.
    #
    # Alternative: Use the group (Z_p)^3 with a different multiplication, or
    # use a semidirect product.
    #
    # Actually, let me try a different approach: use the Cayley graph of
    # H_3(F_p) with generating set S = {e1, e2, e1*e2*e3^{-1}} or similar,
    # but this gives a 3-regular directed graph. For undirected, we need S = S^{-1}.
    #
    # Let me try: S = {a, b, c} where a*b*c = e (so all three are "inverses" in some sense).
    # This is tricky.
    #
    # Better approach: Use a different family. Let me try the Suzuki or other
    # explicit constructions, or fall back to a random 3-regular graph with
    # girth verification.
    #
    # Actually, for the cage problem n(3,16), known constructions exist.
    # Let me try a systematic search over small Heisenberg-type groups.
    
    # Try: Cayley graph of H_3(F_p) with S = {u, v, w} where u^{-1}=u (order 2)
    # This requires p even or a modified group.
    
    # Let me try a completely different structural approach:
    # Use a random 3-regular graph and verify girth, or use known explicit
    # constructions for n(3,16).
    
    # For now, let's try the LUW construction as fallback but also attempt
    # a random construction with girth check.
    
    # Try random 3-regular graphs and check girth
    import random
    random.seed(42)
    
    # Moore bound for n(3,16): lower bound is 2 * (1 + sum_{i=0}^{7} 2^i) = 2*(2^8-1) = 510
    # Actually for girth 16 (even), the Moore bound is 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*255 = 510
    # So n >= 510.
    
    best_n = None
    best_edges = None
    
    # Try a few random 3-regular graphs with increasing n
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        # Start with n around 512 and increase
        n = 512 + trial * 16
        
        if n > 2000:
            break
        
        # Generate random 3-regular graph
        edges = _random_3reg(n)
        if edges is None:
            continue
        
        # Verify girth >= g
        girth = _compute_girth(n, edges, g)
        if girth >= g:
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
                if n <= 512:  # Near Moore bound, very good
                    break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
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


def _random_3reg(n):
    """Generate a random 3-regular graph on n vertices. Returns list of edges or None."""
    if n < 4 or n % 2 != 0:
        return None
    
    import random
    
    # Use configuration model
    # Each vertex has 3 stubs
    stubs = []
    for v in range(n):
        stubs.extend([v, v, v])
    
    # Shuffle and pair
    attempts = 0
    while attempts < 100:
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
        
        if valid and len(edges) == 3 * n // 2:
            return list(edges)
        attempts += 1
    
    return None


def _compute_girth(n, edges, max_girth):
    """Compute the girth of a graph using BFS. Returns the girth or max_girth if girth >= max_girth."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS to find shortest cycle
    # A cycle through vertex v of length L means there's a path of length L-1 from v to a neighbor
    # that doesn't use the first edge.
    
    min_girth = max_girth  # We only care if girth < max_girth
    
    for start in range(n):
        # BFS from start, tracking parent and distance
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < max_girth:
                            return min_girth
    
    return min_girth
# EVOLVE-BLOCK-END
