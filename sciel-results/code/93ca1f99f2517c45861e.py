"""C6-free graph on 40 vertices, maximize edges. Optimized random init + greedy + local search."""
import time
import random

def _has_c6_after_add(adj, u, v):
    """Return True if adding edge (u,v) (currently absent) creates a C6.
    adj: list of int bitsets of neighbors.
    A new C6 is u-a-b-c-d-e-v-u where a in N(u), e in N(v), a!=e, and there's a path of length 3 from a to e.
    """
    Nu = adj[u]
    Nv = adj[v]
    # For each a in N(u), e in N(v), a != e, check if there's a path of length 3 from a to e
    # Path length 3: a-b-c-e where b in N(a)\{u}, c in N(b)\{a,e}, e in N(c)
    # Equivalent: exists b, c such that b in N(a), c in N(b) ∩ N(e), b != e, c != a
    # Even faster: for each a in N(u), compute the set of vertices at distance exactly 2 from a (excluding u)
    # and check if any of those are in N(e) for e in N(v)
    
    # Precompute for each a in N(u): N2(a) = {x : x not in N(a), x != a, and exists y in N(a) with y in N(x)}
    # But we need to be careful about the path not using u or v
    
    # Simpler approach: for each a in N(u), for each e in N(v) with a != e,
    # check if there exists a path a-b-c-e of length 3 where b,c are distinct from a,e,u,v
    # This is equivalent to: (N(a) & N2_from_e) != 0 where N2_from_e is vertices at distance 2 from e
    
    # Let's use a different approach: 
    # A C6 u-a-b-c-d-e-v-u means:
    # a in N(u), e in N(v), a != e
    # b in N(a) \ {u}, c in N(b) \ {a}, d in N(c) \ {b}, e in N(d) \ {c}
    # So we need: exists a in N(u), e in N(v), a!=e, and a path of length 3 from a to e avoiding u,v
    
    # For efficiency, for each a in N(u), compute the set of vertices reachable from a in exactly 2 steps (avoiding u)
    # Then for each e in N(v), check if e is in N2(a) and a != e
    
    # N2_from_a = {x : x != a, x != u, and exists y in N(a)\{u} with x in N(y)}
    # = union over y in N(a)\{u} of N(y), minus {a, u}
    
    a_bits = Nu
    while a_bits:
        lb = a_bits & (-a_bits)
        a = lb.bit_length() - 1
        a_bits ^= lb
        
        # Compute N2_from_a: vertices at distance 2 from a, avoiding u
        # N2 = union of N(y) for y in N(a)\{u}, then remove a and u
        N2 = 0
        y_bits = adj[a] & ~(1 << u)
        while y_bits:
            ly = y_bits & (-y_bits)
            y = ly.bit_length() - 1
            y_bits ^= ly
            N2 |= adj[y]
        N2 &= ~(1 << a)
        N2 &= ~(1 << u)
        
        # Now check if any e in N(v) is in N2, with e != a
        e_bits = Nv & N2 & ~(1 << a)
        if e_bits:
            return True
    
    return False

def _add(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _edge_count(adj, n):
    return sum(b.bit_count() for b in adj) // 2

def construct():
    n = 40
    t0 = time.time()
    deadline = time.time() + 105  # Safety margin: return by 105s, well before 120s limit

    random.seed(12345)

    def time_ok():
        return time.time() < deadline

    best_adj = None
    best_count = -1

    def update_best(adj):
        nonlocal best_adj, best_count
        c = _edge_count(adj, n)
        if c > best_count:
            best_count = c
            best_adj = [b for b in adj]

    # Phase 1: Multiple random restarts with fewer iterations but faster checks
    for restart in range(50):  # Reduced from 2000 to 50
        if not time_ok():
            break
        
        # Random initial graph with edge probability p
        p = 0.3
        adj = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1<<j)
                    adj[j] |= (1<<i)
        
        # Prune: remove edges that are part of a C6 until C6-free
        # Use a faster C6 finding approach
        changed = True
        prune_iter = 0
        while changed and prune_iter < 200 and time_ok():
            changed = False
            prune_iter += 1
            # Find a C6 and remove an edge
            found = False
            # Try to find a C6 by checking all 6-cycles more efficiently
            for u in range(n):
                if found: break
                Nu = adj[u]
                if not Nu: continue
                for a in range(n):
                    if found: break
                    if a == u or not ((Nu >> a) & 1): continue
                    Na = adj[a] & ~(1 << u)
                    for b in range(n):
                        if found: break
                        if b == u or b == a or not ((Na >> b) & 1): continue
                        Nb = adj[b] & ~(1 << a)
                        for c in range(n):
                            if found: break
                            if c == u or c == a or c == b or not ((Nb >> c) & 1): continue
                            Nc = adj[c] & ~(1 << b)
                            for d in range(n):
                                if found: break
                                if d == u or d == a or d == b or d == c or not ((Nc >> d) & 1): continue
                                Nd = adj[d] & ~(1 << c)
                                for e in range(n):
                                    if found: break
                                    if e == u or e == a or e == b or e == c or e == d: continue
                                    if not ((Nd >> e) & 1): continue
                                    if not ((adj[e] >> u) & 1): continue
                                    # C6 found: u-a-b-c-d-e-u
                                    # Remove a random edge
                                    edges = [(u,a),(a,b),(b,c),(c,d),(d,e),(e,u)]
                                    ru,rv = random.choice(edges)
                                    _remove(adj, ru, rv)
                                    found = True
                                    changed = True
                                    break
        
        if not time_ok():
            break
        
        # Greedy add edges: try to add non-edges in random order
        for _ in range(2):  # Reduced from 3 to 2 passes
            if not time_ok(): break
            order = [(i,j) for i in range(n) for j in range(i+1,n) if not ((adj[i]>>j)&1)]
            random.shuffle(order)
            added = False
            for (i,j) in order:
                if not time_ok(): break
                if (adj[i]>>j)&1: continue
                if not _has_c6_after_add(adj, i, j):
                    _add(adj, i, j)
                    added = True
            if not added:
                break
        
        update_best(adj)
        
        # Local search: try edge swaps
        ls_iter = 0
        while ls_iter < 50 and time_ok():  # Reduced from 150 to 50
            ls_iter += 1
            trial = [b for b in adj]
            all_edges = [(i,j) for i in range(n) for j in range(i+1,n) if (trial[i]>>j)&1]
            if not all_edges: break
            u1,v1 = random.choice(all_edges)
            _remove(trial, u1, v1)
            before = _edge_count(trial, n)
            order = [(i,j) for i in range(n) for j in range(i+1,n) if not ((trial[i]>>j)&1)]
            random.shuffle(order)
            for (i,j) in order:
                if not time_ok(): break
                if (trial[i]>>j)&1: continue
                if not _has_c6_after_add(trial, i, j):
                    _add(trial, i, j)
            after = _edge_count(trial, n)
            if after > before:
                adj = trial
                update_best(adj)
            else:
                pass

    # Final: try to add any remaining safe edges to best_adj
    if best_adj is not None and time_ok():
        order = [(i,j) for i in range(n) for j in range(i+1,n) if not ((best_adj[i]>>j)&1)]
        random.shuffle(order)
        for (i,j) in order:
            if not time_ok(): break
            if (best_adj[i]>>j)&1: continue
            if not _has_c6_after_add(best_adj, i, j):
                _add(best_adj, i, j)
        update_best(best_adj)

    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i]>>j)&1:
                edges.append([i,j])
    return edges
