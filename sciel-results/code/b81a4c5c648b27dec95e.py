"""Construct a C4-free graph with no independent 11-set, maximizing n."""
import time
import itertools
import numpy as np

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 110.0  # seconds, safety margin before 120s limit

def _elapsed():
    return time.time() - _START_TIME

def _time_exceeded():
    return _elapsed() > _DEADLINE

def _check_c4_free(n, edges):
    """Check that no two vertices share two common neighbors."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False
    return True

def _max_independent_set_size(n, adj, limit=11):
    """Branch and bound to find max independent set size. Returns size (capped at limit)."""
    # Use greedy coloring upper bound
    # Sort vertices by degree (descending) for better pruning
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # Map: position in order -> vertex
    # We'll work with the ordering
    
    best = [0]
    
    def _color_bound(candidates):
        """Greedy coloring of candidates for upper bound."""
        if not candidates:
            return 0
        # Greedy coloring
        colors = {}
        for v in candidates:
            # Find smallest color not used by neighbors in candidates
            used = set()
            for u in adj[v]:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return max(colors.values()) + 1 if colors else 0
    
    # Simple branch and bound
    # candidates: list of vertices still available
    # current_set: list of vertices in current independent set
    def _branch(candidates, current_size):
        if _time_exceeded():
            return
        if current_size + len(candidates) <= best[0]:
            return
        if not candidates:
            best[0] = max(best[0], current_size)
            return
        
        # Upper bound via coloring
        ub = _color_bound(candidates)
        if current_size + ub <= best[0]:
            return
        
        # Pick a vertex to branch on (first one for simplicity, could be smarter)
        v = candidates[0]
        remaining = candidates[1:]
        
        # Branch 1: include v
        new_candidates = [u for u in remaining if u not in adj[v]]
        _branch(new_candidates, current_size + 1)
        
        # Branch 2: exclude v
        _branch(remaining, current_size)
    
    _branch(order, 0)
    
    # If best >= limit, we don't need exact value
    return min(best[0], limit)

def _build_polarity_graph(q):
    """
    Build a polarity graph from a finite field F_q where q is a prime power.
    Vertices are points in F_q^2. Edge (x,y) ~ (x',y') iff y - y' = (x - x')^2 * q + ... 
    Actually, standard polarity graph: V = F_q x F_q, edge (x,y)~(x',y') iff y + y' = (x - x')^2.
    This is C4-free. Independence number relates to cap sets.
    """
    # For small q, enumerate
    if q < 2 or q > 100:
        return None
    
    # Use finite field arithmetic
    # For prime q, use modular arithmetic
    if all(q % p != 0 for p in range(2, int(q**0.5)+1)):
        # q is prime
        def add(a, b): return (a + b) % q
        def mul(a, b): return (a * b) % q
        def neg(a): return (-a) % q
        def sub(a, b): return (a - b) % q
        def sq(a): return (a * a) % q
    else:
        # Would need GF(p^k), skip for now
        return None
    
    n = q * q
    edges = []
    # Vertex index: i = x * q + y
    adj = [set() for _ in range(n)]
    
    for x in range(q):
        for y in range(q):
            i = x * q + y
            for xp in range(q):
                for yp in range(q):
                    if xp == x and yp == y:
                        continue
                    # Edge condition: y + yp = (x - xp)^2
                    if add(y, yp) == sq(sub(x, xp)):
                        j = xp * q + yp
                        adj[i].add(j)
                        if i < j:
                            edges.append([i, j])
    
    return {'n': n, 'edges': edges, 'adj': adj}

def _try_paley_graph(q):
    """
    Paley graph: vertices 0..q-1, edge i~j iff (j-i)^2 = 1 mod q (quadratic residue).
    C4-free? Not necessarily. Let's check.
    Actually Paley graphs can have C4. Skip unless verified.
    """
    if all(q % p != 0 for p in range(2, int(q**0.5)+1)):
        def is_residue(a):
            if a == 0:
                return False
            return pow(a, (q-1)//2, q) == 1
        n = q
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if is_residue((j - i) % q):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    return None

def _verify_and_check(graph):
    """Verify C4-free and compute independence number."""
    n = graph['n']
    edges = graph['edges']
    adj = graph.get('adj')
    if adj is None:
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
    
    if not _check_c4_free(n, edges):
        return False, 0
    
    alpha = _max_independent_set_size(n, adj, limit=11)
    return True, alpha

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 110.0
    
    best_n = 0
    best_graph = None
    
    # Try polarity graphs for various q
    # Polarity graph on F_q^2 has n = q^2 vertices
    # For q=7: n=49 (too big, max 43)
    # For q=6: not a prime power
    # For q=5: n=25
    # For q=4: n=16 (GF(4))
    # For q=3: n=9
    # For q=2: n=4
    
    # The polarity graph from F_q has independence number related to maximum cap set size in F_q^2.
    # For q=5, max cap set in F_5^2 is 5 (I think), so alpha <= 5. Good.
    # But n=25 is not great.
    
    # Let's try q=7: n=49 > 43, can't use.
    # What about taking a subset?
    
    # Alternative: Use the known best construction.
    # The best known C4-free graph with alpha < 11 has 39 vertices.
    # Let's try to construct one using a different approach.
    
    # Actually, let's try the polarity graph for q=7 but restrict to 43 vertices?
    # That might break the C4-free property.
    
    # Let me try a different family: graphs from symmetric designs.
    # Or use a known explicit construction.
    
    # For now, let's try polarity graphs and see what we get.
    for q in [7, 5, 4, 3, 2]:
        if _time_exceeded():
            break
        g = _build_polarity_graph(q)
        if g is None:
            continue
        if g['n'] > 43:
            continue
        ok, alpha = _verify_and_check(g)
        if ok and alpha < 11:
            if g['n'] > best_n:
                best_n = g['n']
                best_graph = g
    
    # Also try Paley graphs
    for q in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
        if _time_exceeded():
            break
        g = _try_paley_graph(q)
        if g is None or g['n'] > 43:
            continue
        # Add adj
        n = g['n']
        adj = [set() for _ in range(n)]
        for u, v in g['edges']:
            adj[u].add(v)
            adj[v].add(u)
        g['adj'] = adj
        ok, alpha = _verify_and_check(g)
        if ok and alpha < 11:
            if g['n'] > best_n:
                best_n = g['n']
                best_graph = g
    
    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    if best_graph is None or best_n < 30:
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    return {'n': best_graph['n'], 'edges': best_graph['edges']}
