"""Rewritten program: deterministic algebraic constructions + bitmask B&B verification."""
import time
import itertools

# Deadline guard: evaluator kills after 150s; keep a safety margin.
_START = time.time()
_DEADLINE = 140.0  # seconds of wall-clock allowed before we must return


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _has_triangle(adj_mask, n):
    """Exact check: does graph with adjacency bitmasks contain a triangle?"""
    for i in range(n):
        mi = adj_mask[i]
        # For each j > i that is adjacent to i, check if any k > j adjacent to both
        # Simple O(n^3 / 64) via bit ops
        for j in range(i + 1, n):
            if (mi >> j) & 1:
                # common neighbors of i and j above j
                common = mi & adj_mask[j]
                # only care about bits > j (to avoid duplicates and self)
                common &= ~((1 << (j + 1)) - 1)
                if common:
                    return True
    return False


def _max_independent_set(adj_mask, n):
    """Exact branch and bound for maximum independent set size."""
    # Greedy coloring for upper bound (DSATUR-like simple version)
    # We use a simple recursive B&B with bitmasks.
    
    best = [0]
    
    # Precompute: for each vertex, its non-neighbors (independent candidates)
    # Actually, we'll do standard B&B: 
    # State: candidate set (bitmask), current set size
    # Upper bound: candidate set size + current size
    # Prune if upper bound <= best
    
    # Better: use the classic algorithm with coloring bound
    # For n <= 41, a simple recursive B&B with good ordering works.
    
    # Order vertices by degree (descending) for better pruning
    degs = [bin(adj_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degs[x])
    
    # Build new adjacency in order
    pos = {v: i for i, v in enumerate(order)}
    new_adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (adj_mask[i] >> j) & 1:
                new_adj[i] |= (1 << pos[j])
    
    # Simple B&B
    def bb(cand_mask, cur_size):
        if cand_mask == 0:
            if cur_size > best[0]:
                best[0] = cur_size
            return
        # Upper bound
        if cur_size + bin(cand_mask).count('1') <= best[0]:
            return
        # Pick a vertex with max degree in candidate set
        # Find a vertex v in cand_mask
        # Try including v
        v = (cand_mask & -cand_mask).bit_length() - 1
        # Include v: remove v and its neighbors from candidates
        new_cand = cand_mask & ~new_adj[v] & ~(1 << v)
        bb(new_cand, cur_size + 1)
        # Exclude v
        bb(cand_mask & ~(1 << v), cur_size)
    
    bb((1 << n) - 1, 0)
    return best[0]


def _verify(n, edges):
    """Check triangle-free and alpha < 10."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    if _has_triangle(adj, n):
        return False
    alpha = _max_independent_set(adj, n)
    return alpha < 10


def _cayley_graph_cyclic(n, S):
    """Construct Cayley graph of Z_n with connection set S (symmetric)."""
    edges = []
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if i < j:
                edges.append([i, j])
    return n, edges


def _andrasfai_variant(n, k, offset):
    """Generalized Andrasfai: i~j iff (j-i) mod n in S where S = {offset, offset+1, ..., offset+k-1} mod n
       but we need triangle-free. For Z_n, if S has 3 consecutive, triangle exists.
       Use S = {a, b} or specific sets."""
    # Standard Andrasfai: Z_{3m}, S = {1, 3, 5, ..., 3m-2} (all odd)
    # This gives triangle-free with alpha = m
    # For n=3m, alpha = m. We need alpha <= 9, so m <= 9, n <= 27.
    # Not enough for n=39.
    pass


def _known_constructions():
    """Generate candidate graphs from known constructions."""
    candidates = []
    
    # 1. Andrasfai And(9): n=26, but that's only alpha=9 with n=26.
    #    Actually And(m) has n=3m, alpha=m. For alpha=9, n=27.
    #    Wait, the original says n=26. Let me re-check.
    #    Andrasfai graph And(m): vertices Z_{3m}, i~j if (j-i) mod 3m is odd.
    #    Triangle-free, alpha = m.
    #    For m=9: n=27, alpha=9.
    
    # 2. Clebsch graph: n=16, triangle-free, alpha=5
    #    Not large enough.
    
    # 3. Chvatal graph: n=12, triangle-free, alpha=4
    #    Not large enough.
    
    # 4. Schläfli graph: n=27, triangle-free, alpha=9? No, Schläfli has alpha=9? 
    #    Actually Schläfli graph is 16-regular on 27 vertices, triangle-free, alpha=9.
    
    # 5. For n=39: We need a triangle-free graph on 39 vertices with alpha <= 9.
    #    Known: The largest known triangle-free graph with alpha <= 9 has 39 vertices.
    #    This is related to the Ramsey number R(3,10) > 39.
    #    
    #    One construction: Use a Paley-type or other algebraic construction.
    #    
    #    Let's try: Z_39 with connection set S.
    #    For triangle-free: S must not contain x, y, x+y (mod 39) for any x,y in S.
    #    This means S is sum-free in Z_39.
    #    
    #    Max sum-free subset of Z_n has size ~n/3 for odd n.
    #    For n=39, max sum-free set size is 13 (e.g., {14,15,...,26} or {1,2,...,13} with care).
    #    If |S|=13, degree=26, and we need alpha <= 9.
    #    
    #    Let's try S = {1, 2, ..., 13} mod 39. Check triangle-free:
    #    1+2=3 in S, so triangle (0,1,2). Not triangle-free.
    #    
    #    Try S = {14, 15, ..., 26}. Check: 14+14=28 not in S, 14+15=29 not in S, etc.
    #    Max sum: 26+26=52 mod 39 = 13, not in S. 
    #    Min sum: 14+14=28, not in S.
    #    So S={14,...,26} is sum-free in Z_39. Triangle-free!
    #    Degree = 13. By Ramsey/Turán, alpha >= 39/14 ≈ 2.8. 
    #    But we need alpha <= 9. With degree 13, alpha could be as low as ~39/(13+1) ≈ 2.8.
    #    Let's verify computationally.
    
    candidates.append(("S_14_26", 39, _cayley_graph_cyclic(39, list(range(14, 27)))))
    
    # Try other sum-free sets
    # S = {1, 2, ..., 13} is not sum-free. 
    # S = {1, 3, 5, ..., 25} (odd numbers 1 to 25): 13 elements.
    # 1+1=2 not in S, 1+3=4 not in S, ... 25+25=50 mod 39=11 not in S.
    # But 1+5=6 not in S, 3+5=8 not in S. Seems sum-free.
    candidates.append(("S_odd_1_25", 39, _cayley_graph_cyclic(39, list(range(1, 26, 2)))))
    
    # S = {1, 2, 4, 8, 16, 32, 25, 11, 22, 5, 10, 20, 1} - multiplicative? No.
    
    # Let's also try n=40
    # Z_40, need sum-free S. Max size ~13.
    # S = {14,15,...,26}: 14+14=28 not in S, 26+26=52 mod 40=12 not in S.
    # 14+26=40 mod 40=0 not in S. Seems sum-free.
    candidates.append(("S_14_26_40", 40, _cayley_graph_cyclic(40, list(range(14, 27)))))
    
    # Try n=38
    candidates.append(("S_13_25_38", 38, _cayley_graph_cyclic(38, list(range(13, 26)))))
    
    return candidates


def construct():
    """Main entry point. Returns best valid graph found."""
    best_n = 0
    best_edges = []
    
    # First, try known constructions
    candidates = _known_constructions()
    
    for name, n, edges in candidates:
        if _time_left() < 5:
            break
        # Verify
        if _verify(n, edges):
            if n > best_n:
                best_n = n
                best_edges = edges
                print(f"Found valid graph: {name}, n={n}")
    
    # If we found n=39 or 40, we're done (40 is the max possible)
    if best_n >= 39:
        return {'n': best_n, 'edges': best_edges}
    
    # Try more constructions if time permits
    # Search over different connection sets for Z_39
    if _time_left() > 10:
        n = 39
        # Try various sum-free sets
        # A set S is sum-free if for all a,b in S, a+b not in S
        # We'll try random-ish sets and verify
        import random
        random.seed(42)
        
        tried = set()
        for attempt in range(500):
            if _time_left() < 5:
                break
            # Generate a random sum-free set
            # Start with a base sum-free set and perturb
            S = list(range(14, 27))
            # Randomly swap some elements
            for _ in range(3):
                i = random.randint(0, len(S)-1)
                new_val = random.randint(1, n-1)
                if new_val not in S and new_val != n - new_val:  # avoid symmetric issues
                    S[i] = new_val
            # Make symmetric: if x in S, n-x in S
            S = set(S)
            S = S | {(n - x) % n for x in S}
            S.discard(0)
            S = sorted(S)
            
            if tuple(S) in tried:
                continue
            tried.add(tuple(S))
            
            # Check sum-free
            sum_free = True
            for a in S:
                for b in S:
                    if (a + b) % n in S:
                        sum_free = False
                        break
                if not sum_free:
                    break
            
            if not sum_free:
                continue
            
            edges = _cayley_graph_cyclic(n, S)
            if _verify(n, edges):
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    print(f"Found valid graph: random S={S}, n={n}")
                break  # Found one, move on
    
    # Fallback: return the original Andrasfai if nothing better
    if best_n == 0:
        n = 26
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}
