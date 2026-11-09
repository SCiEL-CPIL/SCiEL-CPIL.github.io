"""Construct a K4-free graph with no independent 6-set, maximizing n."""
import time
import random

# EVOLVE-BLOCK-START
_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _START_TIME)

# ---------- Fast exact validators using bitmasks ----------

def _find_k4_fast(adj, n):
    """Return True if adj (list of int bitmasks) contains a K4."""
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common = adj[u] & adj[v]
            # For each w in common (w > v to reduce work), check if any x in common & adj[w]
            # x must differ from u, v, w
            c = common & ~((1 << (v + 1)) - 1)  # only w > v
            while c:
                lb = c & (-c)
                w = lb.bit_length() - 1
                c ^= lb
                # x in adj[u] & adj[v] & adj[w], x != u,v,w, x > w to avoid dup
                xmask = adj[u] & adj[v] & adj[w]
                xmask &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                xmask &= ~((1 << (w + 1)) - 1)  # x > w
                if xmask:
                    return True
    return False

def _find_indep6_fast(adj, n):
    """Return True if adj contains an independent set of size 6."""
    # Build complement adjacency
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ (1 << i) ^ adj[i]
    
    # Backtracking: find K6 in comp
    # Order vertices by comp-degree descending for better pruning
    order = sorted(range(n), key=lambda x: bin(comp[x]).count('1'), reverse=True)
    pos = {v: i for i, v in enumerate(order)}
    # Remap to ordered indices
    comp_ord = [0] * n
    for i, v in enumerate(order):
        mask = 0
        for j, u in enumerate(order):
            if j != i and (comp[v] >> u) & 1:
                mask |= 1 << j
        comp_ord[i] = mask
    
    found = [False]
    
    def bk(current_size, candidates):
        if found[0]:
            return
        if current_size == 6:
            found[0] = True
            return
        if not candidates:
            return
        if current_size + bin(candidates).count('1') < 6:
            return
        # Pick pivot: vertex in candidates with max common neighbors
        # For speed, just iterate
        c = candidates
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            c ^= lb
            new_cand = candidates & comp_ord[v]
            bk(current_size + 1, new_cand)
            if found[0]:
                return
            candidates &= ~lb
        return
    
    bk(0, (1 << n) - 1)
    return found[0]

def _is_valid(adj, n):
    if _find_k4_fast(adj, n):
        return False
    if _find_indep6_fast(adj, n):
        return False
    return True

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# ---------- Finite field / algebraic constructions ----------

def _paley_graph(q):
    """Paley graph on q vertices (q ≡ 1 mod 4 prime)."""
    n = q
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in squares:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj

def _cayley_gf2d(m):
    """
    Cayley graph on GF(2)^m vertices.
    Vertices are 0..2^m - 1.
    Edge between x and y if (x^y) is in a chosen connection set S.
    We choose S to be all vectors with Hamming weight >= 2 (to avoid K4).
    Actually, let's try: S = {v : wt(v) >= 2}.
    This makes the graph the complement of the m-dimensional hypercube.
    """
    n = 1 << m
    # Connection set: all nonzero vectors with weight >= 2
    S = set()
    for v in range(1, n):
        if bin(v).count('1') >= 2:
            S.add(v)
    adj = [0] * n
    for x in range(n):
        for s in S:
            y = x ^ s
            if y > x:
                adj[x] |= 1 << y
                adj[y] |= 1 << x
    return adj

def _cayley_gf2d_weight3(m):
    """Cayley graph on GF(2)^m, connection set = vectors with weight exactly 3."""
    n = 1 << m
    S = set()
    for v in range(1, n):
        if bin(v).count('1') == 3:
            S.add(v)
    adj = [0] * n
    for x in range(n):
        for s in S:
            y = x ^ s
            if y > x:
                adj[x] |= 1 << y
                adj[y] |= 1 << x
    return adj

def _cayley_gf2d_weight_ge2(m):
    """Cayley graph on GF(2)^m, connection set = vectors with weight >= 2."""
    n = 1 << m
    S = set()
    for v in range(1, n):
        if bin(v).count('1') >= 2:
            S.add(v)
    adj = [0] * n
    for x in range(n):
        for s in S:
            y = x ^ s
            if y > x:
                adj[x] |= 1 << y
                adj[y] |= 1 << x
    return adj

def _blowup(base_adj, base_n, copies_per_vertex):
    """
    Blow up each vertex of base into copies_per_vertex vertices.
    Within a copy: no edges (independent set).
    Between copies of u and v: complete bipartite if edge in base.
    This preserves: if base is K4-free, blowup is K4-free (since within a copy there are no edges,
    a K4 would need at least 2 vertices from different copies, and those copies' base vertices
    must form a K4 in base... actually need to be careful).
    
    Actually, for K4-freedom: a K4 in the blowup uses vertices from at most 4 distinct base vertices.
    If it uses 4 distinct base vertices, those must form a K4 in base.
    If it uses 3 distinct base vertices, say u,v,w, then we need edges between all pairs of the
    chosen copies. Since copies of u and v are fully connected, etc., we'd need u-v, v-w, u-w edges
    in base, plus we pick one vertex from each of 3 copies... but that's only 3 vertices. To get 4,
    we'd need 2 from one copy, but copies are independent. So K4 must use 4 distinct base vertices.
    
    For independence: an independent set in blowup uses at most one vertex per copy (since copies
    are independent... wait, copies ARE independent sets, so we can take multiple from one copy).
    Actually copies are independent, so an independent set can take all copies from a set of base
    vertices that form an independent set in base. So if base has independent set of size k,
    blowup has independent set of size k * copies_per_vertex.
    
    So this is useful if base has small independence number.
    """
    total_n = base_n * copies_per_vertex
    adj = [0] * total_n
    for u in range(base_n):
        for v in range(u + 1, base_n):
            if (base_adj[u] >> v) & 1:
                # Complete bipartite between copy u and copy v
                for i in range(copies_per_vertex):
                    for j in range(copies_per_vertex):
                        a = u * copies_per_vertex + i
                        b = v * copies_per_vertex + j
                        adj[a] |= 1 << b
                        adj[b] |= 1 << a
    return adj

def _construct_candidates():
    """Generate a list of candidate (n, adj) pairs from algebraic constructions."""
    candidates = []
    
    # Paley graphs
    for q in [5, 9, 13, 17, 29, 37]:
        if q == 9:
            continue  # 9 is not prime
        # Check q ≡ 1 mod 4
        if q % 4 != 1:
            continue
        try:
            adj = _paley_graph(q)
            candidates.append((q, adj))
        except Exception:
            pass
    
    # Cayley graphs on GF(2)^m
    for m in range(3, 7):
        n = 1 << m
        if n > 40:
            break
        try:
            adj = _cayley_gf2d_weight_ge2(m)
            candidates.append((n, adj))
        except Exception:
            pass
        try:
            adj = _cayley_gf2d_weight3(m)
            candidates.append((n, adj))
        except Exception:
            pass
    
    # Blowups
    # Paley 17 blown up by 2 -> 34 vertices
    try:
        p17 = _paley_graph(17)
        adj34 = _blowup(p17, 17, 2)
        candidates.append((34, adj34))
    except Exception:
        pass
    
    # Paley 5 blown up by 7 -> 35 vertices
    try:
        p5 = _paley_graph(5)
        adj35 = _blowup(p5, 5, 7)
        candidates.append((35, adj35))
    except Exception:
        pass
    
    # Paley 13 blown up by 3 -> 39 vertices
    try:
        p13 = _paley_graph(13)
        adj39 = _blowup(p13, 13, 3)
        candidates.append((39, adj39))
    except Exception:
        pass
    
    # Paley 9? No, 9 not prime.
    # Paley 29 blown up by 1 -> 29
    # Already covered.
    
    # GF(2)^5 = 32 vertices, weight>=2
    # Already covered.
    
    return candidates

def _local_search_extend(adj, n, max_time=20.0):
    """
    Try to extend the graph by adding vertices one by one.
    For each new vertex, find a valid neighborhood S (triangle-free in adj[S])
    such that the non-neighborhood has no independent 5-set.
    """
    start = time.time()
    rng = random.Random(12345)
    current_n = n
    current_adj = list(adj)
    
    while _time_left() > 5.0 and current_n < 40:
        target_n = current_n + 1
        added = False
        attempts = 0
        max_attempts = 500
        
        while attempts < max_attempts and _time_left() > 3.0:
            attempts += 1
            # Build a random triangle-free subset S
            S = set()
            order = list(range(current_n))
            rng.shuffle(order)
            for v in order:
                # Check if v can be added to S without creating a triangle
                ok = True
                for i in S:
                    if (adj[i] >> v) & 1:
                        # i and v are adjacent; check if any other in S adjacent to both
                        for j in S:
                            if j != i and (adj[i] >> j) & 1 and (adj[j] >> v) & 1:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    S.add(v)
            
            # Check independence condition: non-neighbors N must have no indep 5-set
            N = set(range(current_n)) - S
            if len(N) >= 5:
                # Quick check: sample some 5-subsets
                n_list = list(N)
                found_indep5 = False
                # If |N| is small, check all
                if len(N) <= 15:
                    from itertools import combinations
                    for combo in combinations(n_list, 5):
                        indep = True
                        for i in range(5):
                            for j in range(i + 1, 5):
                                if (current_adj[combo[i]] >> combo[j]) & 1:
                                    indep = False
                                    break
                            if not indep:
                                break
                        if indep:
                            found_indep5 = True
                            break
                else:
                    for _ in range(200):
                        combo = rng.sample(n_list, 5)
                        indep = True
                        for i in range(5):
                            for j in range(i + 1, 5):
                                if (current_adj[combo[i]] >> combo[j]) & 1:
                                    indep = False
                                    break
                            if not indep:
                                break
                        if indep:
                            found_indep5 = True
                            break
                if found_indep5:
                    continue
            
            # Valid: add vertex
            new_adj = list(current_adj)
            new_adj.append(0)
            for v in S:
                new_adj[v] |= 1 << (target_n - 1)
                new_adj[target_n - 1] |= 1 << v
            current_adj = new_adj
            current_n = target_n
            added = True
            break
        
        if not added:
            break
    
    return current_n, current_adj

def _edge_flip_search(adj, n, max_time=15.0):
    """
    Local search: try adding edges (to reduce independence) or removing edges (to reduce cliques).
    Accept if the result is still valid.
    """
    start = time.time()
    rng = random.Random(999)
    current_adj = list(adj)
    
    while _time_left() > 5.0 and (time.time() - start) < max_time:
        improved = False
        for _ in range(200):
            if time.time() - start > max_time:
                break
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            has_edge = (current_adj[u] >> v) & 1
            new_adj = list(current_adj)
            if has_edge:
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                new_adj[u] |= 1 << v
                new_adj[v] |= 1 << u
            
            if _is_valid(new_adj, n):
                current_adj = new_adj
                improved = True
                break
        
        if not improved:
            break
    
    return current_adj

def construct():
    # Generate candidates from algebraic constructions
    candidates = _construct_candidates()
    
    # Sort by n descending, try largest first
    candidates.sort(key=lambda x: -x[0])
    
    best_n = 0
    best_adj = None
    
    for n, adj in candidates:
        if _time_left() < 10.0:
            break
        if n > 40 or n < 1:
            continue
        try:
            if _is_valid(adj, n):
                if n > best_n:
                    best_n = n
                    best_adj = list(adj)
                # Try to extend
                if _time_left() > 10.0:
                    ext_n, ext_adj = _local_search_extend(adj, n, max_time=min(15.0, _time_left() - 10.0))
                    if ext_n > best_n and _is_valid(ext_adj, ext_n):
                        best_n = ext_n
                        best_adj = list(ext_adj)
        except Exception:
            continue
    
    # If we have a valid graph, try edge flip search to improve (not increase n, but might help)
    # Actually, edge flips don't change n, so they don't help maximize n.
    # Skip unless we need to fix something.
    
    if best_n > 0 and best_adj is not None:
        edges = _edges_from_adj(best_adj, best_n)
        return {'n': best_n, 'edges': edges}
    
    # Fallback: Paley 17
    try:
        adj17 = _paley_graph(17)
        if _is_valid(adj17, 17):
            edges = _edges_from_adj(adj17, 17)
            return {'n': 17, 'edges': edges}
    except Exception:
        pass
    
    # Last resort: return a trivial valid graph
    return {'n': 1, 'edges': []}
# EVOLVE-BLOCK-END
