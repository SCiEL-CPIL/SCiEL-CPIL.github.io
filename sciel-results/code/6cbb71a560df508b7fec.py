"""
C4-free graph with high minimum degree via finite-field / combinatorial-design construction.

Strategy (structurally different from random greedy + local search):
  1. Deterministic algebraic seed: Paley graph on a prime q (C4-free when q ≡ 1 mod 4 is NOT guaranteed,
     but for q ≡ 3 mod 4 the Paley graph is C4-free for small q; we verify and fall back).
     Actually, a safer C4-free algebraic construction: use the incidence graph of a finite projective plane
     (bipartite, girth 6 => C4-free) or the "norm graph" / polarity graph of a finite field.
     We use the *polarity graph of PG(2, q)*: vertices are points of PG(2,q), two points adjacent iff
     they are conjugate under a polarity. This graph is C4-free for q a prime power, has n = q^2+q+1
     vertices, and minimum degree = q. For q=7, n=57 (too big); q=5, n=31; q=3, n=13.
     We need n up to 47 with min degree >= n-39. For n=46, min degree >= 7.
     The polarity graph of PG(2,5) has n=31, min degree=5. Not enough.
     
     Better: use the *affine plane* construction. Take AG(2, q) points (q^2 vertices). Two points
     are adjacent if they lie on a line of a fixed parallel class (a "spread" of q+1 parallel classes).
     Each point is on one line from each parallel class. If we use all q+1 parallel classes, each point
     has degree (q+1)*(q-1) = q^2-1, which is too high and creates C4s.
     
     Correct C4-free construction: *linear hypergraph* from a finite projective plane.
     PG(2, q) has q^2+q+1 points and q^2+q+1 lines, each line has q+1 points, each point on q+1 lines,
     any two points on exactly one line. Define graph on the *lines* of PG(2,q): two lines adjacent iff
     they intersect. This is the *dual* of the point graph, also C4-free? No, the line graph of a projective
     plane has C4s.
     
     Let me use a verified construction: the *norm graph* of a finite field.
     Actually, the simplest reliable C4-free dense graph: the *incidence graph of a projective plane* is
     bipartite with girth 6, hence C4-free. It has 2*(q^2+q+1) vertices, each of degree q+1.
     For q=5: 62 vertices, degree 6. n=62 > 47, too many.
     For q=3: 26 vertices, degree 4. min degree 4, need >= 26-39 = negative, so OK but n too small.
     
     We need n in [40, 47] with min degree >= n-39 (which is 1 to 8).
     A very simple C4-free graph: a *star* is C4-free but min degree is 0.
     A *tree* is C4-free. A tree on n vertices has min degree 1 (if n>2). We need min degree >= n-39.
     For n=46, need min degree >= 7. A tree can't do that.
     
     Key insight: we don't need the graph to be *maximally* dense, just C4-free with min degree >= n-39.
     For n=46, min degree >= 7. For n=47, min degree >= 8.
     
     A *complete bipartite* graph K_{a,b} is C4-free iff a<=1 or b<=1 (since K_{2,2}=C4).
     So K_{1, n-1} is C4-free, min degree 1. Not enough.
     
     What about the *line graph of a tree*? Line graphs of trees are C4-free (trees have no cycles,
     so line graphs have no C4s... actually line graph of a tree can have C4s? No: a C4 in the line graph
     would correspond to a 4-cycle in the original graph, but trees have no cycles. So line graphs of trees
     are C4-free). The line graph of a star K_{1,n-1} is K_{n-1}, which is C4-free only if n-1 <= 3.
     
     Let me use a *random algebraic* approach: generate a C4-free graph via the *greedy random* method
     but seeded by a structured ordering to avoid the local minima, then verify.
     
     Actually, the most promising structural change: use the *complement* framing.
     We need a C4-free graph G on n vertices with min degree >= n-39.
     Equivalently, the complement H = complement(G) has max degree <= 38 and G is C4-free.
     G is C4-free iff H has no induced... no, G C4-free means no 4-cycle in G.
     
     Let me just do a *deterministic* construction: take n vertices, and for each vertex i, connect it
     to vertices i+1, i+2, ..., i+d mod n (a circulant graph). A circulant graph C(n; S) is C4-free
     if the difference set S has no solution to s1+s2 = s3+s4 for distinct pairs.
     For n=46, choose S = {1, 2, 3, 4, 5, 6, 7} (d=7). min degree = 14 (since ±S). That's >= 7.
     Is C(46; {1..7}) C4-free? A C4 exists if there are a,b,c,d with a-b = c-d and a-c = b-d (mod 46)
     with all differences in S. This is equivalent to a 4-term arithmetic progression in S or similar.
     {1,2,3,4,5,6,7} contains 1,2,3,4 which is a 4-AP: 1,2,3,4 with differences 1,1,1. 
     a=0, b=1, c=2, d=3: a-b=-1, c-d=-1, a-c=-2, b-d=-2. All in ±S. So C4 exists.
     
     Use S = {1, 3, 4, 9, 11, 15, 20} (a Sidon-like set). Need to check.
     
     Given time constraints, let me use a *verified small construction*:
     For n=46, I'll construct the graph as the *disjoint union* of a C4-free component and isolated... no, min degree.
     
     Final approach: Use the *Paley graph* on a prime q where q ≡ 3 mod 4 is NOT C4-free in general.
     But the *bipartite incidence graph of PG(2, q)* is C4-free. For q=3, it has 26 vertices, degree 4.
     We can take a *subgraph* or *induced subgraph* of a larger C4-free graph to get exactly n vertices
     with the right min degree.
     
     Simplest reliable method: use a *random* C4-free graph generator that is known to work:
     Start with empty graph, add edges in a random order, skip any edge that creates a C4.
     This is the same as the current program. To make it structurally different, use a *simulated
     annealing* or *genetic algorithm* on the edge set.
     
     Given the 150s budget, I'll implement a *deterministic circulant* construction with a verified
     C4-free difference set, falling back to random greedy if the circulant fails.
"""
import time
import random
from itertools import combinations

def _is_c4_free(n, edges):
    """Check if graph is C4-free using bitsets."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            c = common
            if c & (c - 1):
                return False
    return True

def _min_degree(n, edges):
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return min(deg) if deg else 0

def _circulant_c4_free(n, S):
    """
    Build circulant graph C(n; S) and check if C4-free.
    S is a set of positive integers < n//2.
    Vertices 0..n-1, edge (i, i+s mod n) for s in S.
    """
    S = sorted(set(S))
    edges = []
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _try_circulant(n):
    """Try to find a C4-free circulant graph on n vertices with min degree >= n-39."""
    target = n - 39
    if target <= 0:
        target = 1
    # Try various difference sets
    # A Sidon set mod n has no C4s in the circulant
    # Use small sets and verify
    import itertools
    max_s = n // 2
    # Try sets of size target//2 (since each s gives degree 2, except s=n/2)
    needed_pairs = (target + 1) // 2
    # Generate candidate Sidon-like sets
    random.seed(42)
    for attempt in range(2000):
        # Random subset of {1, ..., max_s-1}
        S = random.sample(range(1, max_s), min(needed_pairs, max_s - 1))
        edges = _circulant_c4_free(n, S)
        md = _min_degree(n, edges)
        if md >= target and _is_c4_free(n, edges):
            return edges
    return None

def _random_greedy_c4_free(n, target_deg, deadline, seed=0):
    """Random greedy: add edges that don't create C4, try to reach target min degree."""
    random.seed(seed)
    adj = [0] * n
    deg = [0] * n
    
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        # Check if adding (u,v) creates C4
        # C4: u-v-w-x-u. w in N(v), x in N(u), w~x, w!=x, w!=u, w!=v, x!=u, x!=v
        nv = adj[v]
        nu = adj[u]
        w_mask = nv & ~(1 << u)
        creates = False
        while w_mask:
            w_bit = w_mask & (-w_mask)
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            x_mask = nu & ~(1 << v) & ~w_bit
            if x_mask & adj[w]:
                creates = True
                break
        if not creates:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            deg[u] += 1
            deg[v] += 1
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    if min(deg) >= target_deg and _is_c4_free(n, edges):
        return edges
    return None

def _local_search_improve(n, edges, target_deg, deadline):
    """Try to improve min degree via edge swaps."""
    adj = [0] * n
    deg = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        deg[u] += 1
        deg[v] += 1
    
    for step in range(10000):
        if time.time() > deadline:
            break
        min_d = min(deg)
        if min_d >= target_deg:
            break
        low_v = deg.index(min_d)
        # Try to add an edge from low_v to a non-neighbor
        candidates = [v for v in range(n) if v != low_v and not (adj[low_v] & (1 << v))]
        random.shuffle(candidates)
        improved = False
        for v in candidates:
            if time.time() > deadline:
                break
            # Check C4
            nv = adj[v]
            nu = adj[low_v]
            w_mask = nv & ~(1 << low_v)
            creates = False
            while w_mask:
                w_bit = w_mask & (-w_mask)
                w = w_bit.bit_length() - 1
                w_mask ^= w_bit
                x_mask = nu & ~(1 << v) & ~w_bit
                if x_mask & adj[w]:
                    creates = True
                    break
            if not creates:
                adj[low_v] |= 1 << v
                adj[v] |= 1 << low_v
                deg[low_v] += 1
                deg[v] += 1
                improved = True
                break
        if not improved:
            # Try swap: remove edge (a,b) where a has high degree, add (low_v, b)
            high_v = deg.index(max(deg))
            if high_v == low_v:
                continue
            neighbors = [v for v in range(n) if adj[high_v] & (1 << v)]
            random.shuffle(neighbors)
            for b in neighbors:
                if time.time() > deadline:
                    break
                if not (adj[low_v] & (1 << b)):
                    # Check if removing (high_v, b) and adding (low_v, b) is C4-free
                    # Temporarily remove
                    adj[high_v] &= ~(1 << b)
                    adj[b] &= ~(1 << high_v)
                    deg[high_v] -= 1
                    deg[b] -= 1
                    # Check add (low_v, b)
                    nv = adj[b]
                    nu = adj[low_v]
                    w_mask = nv & ~(1 << low_v)
                    creates = False
                    while w_mask:
                        w_bit = w_mask & (-w_bit if False else -w_mask)
                        w = w_bit.bit_length() - 1
                        w_mask ^= w_bit
                        x_mask = nu & ~(1 << b) & ~w_bit
                        if x_mask & adj[w]:
                            creates = True
                            break
                    if not creates:
                        adj[low_v] |= 1 << b
                        adj[b] |= 1 << low_v
                        deg[low_v] += 1
                        deg[b] += 1
                    else:
                        # Undo
                        adj[high_v] |= 1 << b
                        adj[b] |= 1 << high_v
                        deg[high_v] += 1
                        deg[b] += 1
                        continue
            if not improved:
                continue
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    if min(deg) >= target_deg and _is_c4_free(n, edges):
        return edges
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 39
    best_edges = []
    
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        target_deg = target_n - 39
        if target_deg < 1:
            target_deg = 1
        
        # Method 1: Circulant construction (deterministic, structured)
        if time.time() < deadline - 5:
            edges = _try_circulant(target_n)
            if edges is not None:
                best_n = target_n
                best_edges = edges
                break
        
        # Method 2: Random greedy with multiple seeds
        for seed in range(500):
            if time.time() > deadline:
                break
            edges = _random_greedy_c4_free(target_n, target_deg, deadline, seed=seed * 104729 + 1)
            if edges is not None:
                best_n = target_n
                best_edges = edges
                break
        
        if best_n >= target_n:
            break
        
        # Method 3: Local search from a partial graph
        if time.time() < deadline - 10:
            # Get a partial graph from greedy
            random.seed(12345)
            adj = [0] * target_n
            deg = [0] * target_n
            all_edges = list(combinations(range(target_n), 2))
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                nv = adj[v]
                nu = adj[u]
                w_mask = nv & ~(1 << u)
                creates = False
                while w_mask:
                    w_bit = w_mask & (-w_mask)
                    w = w_bit.bit_length() - 1
                    w_mask ^= w_bit
                    x_mask = nu & ~(1 << v) & ~w_bit
                    if x_mask & adj[w]:
                        creates = True
                        break
                if not creates:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
            partial_edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if adj[u] & (1 << v):
                        partial_edges.append([u, v])
            
            improved = _local_search_improve(target_n, partial_edges, target_deg, deadline)
            if improved is not None:
                best_n = target_n
                best_edges = improved
                break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    return {'n': 39, 'edges': []}
