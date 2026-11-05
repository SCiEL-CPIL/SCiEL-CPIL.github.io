"""Stronger design: constructive + local search for a C4-free graph on n≈39 with alpha<11."""
import time
import random
import sys

# EVOLVE-BLOCK-START
_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _check_c4(adj):
    """adj: list of sets. Return True if no C4 (no two vertices with >=2 common neighbours)."""
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _alpha_greedy_lb(adj):
    """Greedy lower bound on independence number."""
    n = len(adj)
    remaining = set(range(n))
    alpha = 0
    while remaining:
        # pick vertex with smallest degree within remaining
        best_v = None
        best_deg = 10**9
        for v in remaining:
            d = len(adj[v] & remaining)
            if d < best_deg:
                best_deg = d
                best_v = v
        alpha += 1
        remaining.discard(best_v)
        remaining -= adj[best_v]
    return alpha

def _max_independent_set(adj, limit=11):
    """Return True if there exists an independent set of size >= limit."""
    n = len(adj)
    # Use branch and bound with coloring upper bound
    # Convert to bitmasks for speed
    masks = []
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks.append(m)

    sys.setrecursionlimit(10000)

    def branch(remaining_mask, depth, current_size):
        if current_size + depth < limit - 1:
            return False
        if current_size >= limit:
            return True
        if remaining_mask == 0:
            return False
        # Pick a vertex
        # Choose vertex with highest degree in remaining
        rem = remaining_mask
        best_v = -1
        best_deg = -1
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            deg = bin(remaining_mask & masks[v]).count("1")
            if deg > best_deg:
                best_deg = deg
                best_v = v
        # Branch: include best_v
        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, depth - 1, current_size + 1):
            return True
        # Branch: exclude best_v
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, depth - 1, current_size):
            return True
        return False

    full = (1 << n) - 1
    return branch(full, n, 0)

def _has_indep_11(adj):
    """Check if there is an independent set of size 11."""
    n = len(adj)
    if n < 11:
        return False
    # Quick greedy check
    alpha_lb = _alpha_greedy_lb(adj)
    if alpha_lb < 11:
        # Not conclusive, do exact
        pass
    # Exact check using the branch and bound
    return _max_independent_set(adj, 11)

def _build_polarity_graph_q5():
    """Polarity graph of PG(2,5): 31 vertices, C4-free."""
    # Points of PG(2,5) up to scalar: representatives (x,y,z) with x=1 or x=0,y=1 etc.
    # Simpler: enumerate all nonzero triples in F5^3, quotient by scalars {1,2,3,4}
    # Represent each projective point by canonical form.
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize: divide by first nonzero component
                if x != 0:
                    inv = pow(x, 3, 5)  # x^3 = x^{-1} mod 5
                    cx, cy, cz = 1, (y*inv) % 5, (z*inv) % 5
                elif y != 0:
                    inv = pow(y, 3, 5)
                    cx, cy, cz = 0, 1, (z*inv) % 5
                else:
                    cx, cy, cz = 0, 0, 1
                key = (cx, cy, cz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    npts = len(points)  # should be 31
    # Bipolarity: x*x' + y*y' + z*z' = 0 mod 5 (this is a symmetric polarity? No, need a proper one)
    # Standard: use quadratic form x^2 + y^2 + z^2 = 0? That's not a polarity in the right sense.
    # Let's use: two points P=(x,y,z), Q=(x',y',z') are adjacent if x*x' + y*y' + z*z' = 0 mod 5
    # This gives a C4-free graph if the form is anisotropic enough.
    adj = [set() for _ in range(npts)]
    for i in range(npts):
        for j in range(i+1, npts):
            a = points[i]
            b = points[j]
            s = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) % 5
            if s == 0:
                adj[i].add(j)
                adj[j].add(i)
    # Make sure no self-loops and it's C4-free
    # Check C4
    if not _check_c4(adj):
        # Fallback: use a known smaller construction
        adj = []
        for t in range(10):
            base = len(adj)
            adj.append(set())
            adj.append(set())
            adj.append(set())
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            adj[a].add(b); adj[b].add(a)
            adj[a].add(c); adj[c].add(a)
            adj[b].add(c); adj[c].add(b)
        return adj
    return adj

def _local_search(adj, target_n, max_iters=2000):
    """Try to grow or modify the graph to have target_n vertices with alpha < 11."""
    n = len(adj)
    # Ensure adj is list of sets
    adj = [set(s) for s in adj]
    best_adj = [set(s) for s in adj]
    best_score = 0  # higher is better: n - penalty for alpha
    
    def score(a):
        nn = len(a)
        # Penalize if alpha >= 11
        al = _alpha_greedy_lb(a)
        if al >= 11:
            # Exact check is expensive; use greedy as proxy for speed in local search
            return nn - 100
        return nn
    
    # Add vertices one by one
    for _ in range(target_n - n):
        if _deadline_hit():
            break
        # Try to add a new vertex with edges to existing ones, maintaining C4-free
        added = False
        for attempt in range(50):
            new_idx = n
            # Choose random subset of existing vertices to connect to
            new_neigh = set()
            for v in range(n):
                if random.random() < 0.3:
                    new_neigh.add(v)
            # Check C4 condition: for any two existing neighbors a,b of new vertex,
            # they must have at most 1 common neighbor among existing vertices (since new vertex is added,
            # it would be a second common neighbor if |N(a)∩N(b)| >= 1 already)
            # Actually: new vertex u connected to a and b. C4 would be u-a-?-b-u or similar.
            # The condition is: for any two neighbors a,b of u, |N(a) ∩ N(b) ∩ existing| must be < 1? 
            # No: u-a-w-b-u is a C4 if w is a common neighbor of a and b. So we need |N(a)∩N(b)| <= 0 for all pairs a,b in new_neigh? 
            # That's too strict. Actually C4 is a 4-cycle: u-a-w-b-u. So a and b must share a common neighbor w.
            # So for the graph to remain C4-free, for any two vertices a,b that both connect to u, 
            # they must NOT have any common neighbor w (other than u, which isn't in the graph yet).
            # Wait, u is new, so w is in existing graph. So N(a)∩N(b) must be empty? 
            # That seems very restrictive. Let me re-examine.
            # C4: vertices v1-v2-v3-v4-v1. If u=v1, a=v2, w=v3, b=v4. Then u~a, a~w, w~b, b~u.
            # So a and b are both neighbors of u, and a~w~b. So a and b have a common neighbor w.
            # Thus, to avoid C4, for any two neighbors a,b of the new vertex u, we must have N(a)∩N(b) = empty? 
            # Not exactly: w must be adjacent to both a and b. So yes, a and b cannot share any common neighbor in the existing graph.
            # This is indeed very restrictive.
            
            valid = True
            neigh_list = list(new_neigh)
            for i in range(len(neigh_list)):
                for j in range(i+1, len(neigh_list)):
                    a, b = neigh_list[i], neigh_list[j]
                    if adj[a] & adj[b]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Also need to check that adding u doesn't create C4 through paths not involving two neighbors of u
            # The only new edges are u-a for a in new_neigh. Any C4 must use at least one of these new edges.
            # A C4 using edge u-a: u-a-x-y-u where y is another neighbor of u and x is neighbor of a and y.
            # So x is in N(a) ∩ N(y). This is the same condition: for any two neighbors a,y of u, N(a)∩N(y) must be empty.
            # So the check above is sufficient.
            
            # Now add the vertex
            for v in range(n):
                if v in new_neigh:
                    adj[v].add(new_idx)
            adj.append(set(new_neigh))
            n = new_idx + 1
            added = True
            break
        
        if not added:
            # Try with fewer neighbors
            for attempt in range(30):
                new_idx = n
                new_neigh = set()
                for v in range(n):
                    if random.random() < 0.15:
                        new_neigh.add(v)
                valid = True
                neigh_list = list(new_neigh)
                for i in range(len(neigh_list)):
                    for j in range(i+1, len(neigh_list)):
                        a, b = neigh_list[i], neigh_list[j]
                        if adj[a] & adj[b]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                for v in range(n):
                    if v in new_neigh:
                        adj[v].add(new_idx)
                adj.append(set(new_neigh))
                n = new_idx + 1
                added = True
                break
    
    # Now try to improve by adding edges between existing vertices
    for it in range(max_iters):
        if _deadline_hit():
            break
        # Pick random pair of non-adjacent vertices
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v or v in adj[u]:
            continue
        # Check if adding (u,v) creates C4
        if len(adj[u] & adj[v]) >= 2:
            continue
        # Add the edge
        adj[u].add(v)
        adj[v].add(u)
        # Check alpha
        al = _alpha_greedy_lb(adj)
        if al < 11:
            s = n
            if s > best_score:
                best_score = s
                best_adj = [set(x) for x in adj]
        else:
            # Revert
            adj[u].discard(v)
            adj[v].discard(u)
    
    return best_adj, best_score

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 118.0
    
    # Start with polarity graph or triangles
    adj = _build_polarity_graph_q5()
    n0 = len(adj)
    
    # Try to grow to 39
    target = 39
    best_adj, best_score = _local_search(adj, target, max_iters=5000)
    
    # Verify the best result
    if len(best_adj) >= 11:
        if not _check_c4(best_adj):
            # Invalid, fall back
            best_adj = []
            for t in range(10):
                base = len(best_adj)
                best_adj.append(set())
                best_adj.append(set())
                best_adj.append(set())
            for t in range(10):
                a, b, c = 3*t, 3*t+1, 3*t+2
                best_adj[a].add(b); best_adj[b].add(a)
                best_adj[a].add(c); best_adj[c].add(a)
                best_adj[b].add(c); best_adj[c].add(b)
        else:
            # Check alpha exactly
            if _has_indep_11(best_adj):
                # Has independent set of size 11, not valid for our purpose
                # But the task says "there must be no independent set of 11 vertices"
                # So we need alpha <= 10
                pass
    
    # Convert to edges list
    n = len(best_adj)
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    
    # Validate final result
    if n >= 1 and n <= 43 and _check_c4(best_adj) and not _has_indep_11(best_adj):
        pass
    else:
        # Fallback to seed
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
