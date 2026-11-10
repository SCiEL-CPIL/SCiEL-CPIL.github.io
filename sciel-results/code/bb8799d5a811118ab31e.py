"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    n = 40
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def neighbors_mask(u):
        return adj_mask[u]
    
    def degree(u):
        return bin(adj_mask[u]).count('1')
    
    def total_edges():
        return sum(degree(u) for u in range(n)) // 2
    
    def has_c6_fast(u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks.
        
        A C6 through edge (u,v) means there's a path of length 5 from u to v:
        u - a - b - c - d - e - v (6 edges, 7 vertices, all distinct)
        Wait, that's 7 vertices. Let me recount.
        
        C6 has 6 vertices and 6 edges. If we add edge (u,v), the other 4 edges
        form a path of length 4 from u to v: u - a - b - c - d - v
        So we need a path of length 4 from u to v in the current graph.
        
        Actually, adding edge (u,v) to create a C6 means there exists a path
        of length 5 from u to v in the graph WITHOUT the edge (u,v).
        Path: u - x1 - x2 - x3 - x4 - x5 - v (6 edges total including new one)
        So in the current graph, we need a path of length 5 from u to v.
        
        Let me use a different approach: check all possible 4-tuples (a,b,c,d)
        where u-a, a-b, b-c, c-d, d-v are edges, and all 6 vertices u,a,b,c,d,v
        are distinct.
        """
        # For each a in N(u), for each b in N(a)\{u}, for each c in N(b)\{u,a},
        # for each d in N(c)\{u,a,b} ∩ N(v)\{u,a,b,c}, check if d != v and all distinct
        
        nu = adj_mask[u] & ~(1 << v)  # neighbors of u except v
        nv = adj_mask[v] & ~(1 << u)  # neighbors of v except u
        
        # Iterate through neighbors of u
        a = nu
        while a:
            la = a & (-a)
            a ^= la
            ai = la.bit_length() - 1
            
            # b in N(a) \ {u, v, ai}
            nb = adj_mask[ai] & ~(1 << u) & ~(1 << v) & ~(1 << ai)
            b = nb
            while b:
                lb = b & (-b)
                b ^= lb
                bi = lb.bit_length() - 1
                
                # c in N(b) \ {u, v, ai, bi}
                nc = adj_mask[bi] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << bi)
                c = nc
                while c:
                    lc = c & (-c)
                    c ^= lc
                    ci = lc.bit_length() - 1
                    
                    # d in N(c) ∩ N(v) \ {u, ai, bi, ci}
                    # d must be neighbor of both ci and v, and not u, ai, bi, ci
                    nd = adj_mask[ci] & nv & ~(1 << u) & ~(1 << ai) & ~(1 << bi) & ~(1 << ci)
                    if nd:
                        return True
        
        return False
    
    # Start with 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(5 * k + a, 5 * k + b)
    
    # Track best state
    best_adj_mask = list(adj_mask)
    best_count = total_edges()
    
    # Greedy edge addition with multiple passes
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        added_any = False
        
        for (u, v) in all_pairs:
            if time.time() > deadline - 3:
                break
            if (1 << v) & adj_mask[u]:
                continue
            if not has_c6_fast(u, v):
                add_edge(u, v)
                added_any = True
        
        current_count = total_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj_mask = list(adj_mask)
        
        if not added_any:
            # Try perturbation: remove a random edge and try to re-add different ones
            if time.time() > deadline - 3:
                break
            # Pick a random edge to remove
            all_current_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (1 << v) & adj_mask[u]]
            if all_current_edges:
                ru, rv = random.choice(all_current_edges)
                remove_edge(ru, rv)
                # Now try to add edges again
                random.shuffle(all_pairs)
                for (u, v) in all_pairs:
                    if time.time() > deadline - 3:
                        break
                    if (1 << v) & adj_mask[u]:
                        continue
                    if not has_c6_fast(u, v):
                        add_edge(u, v)
                current_count = total_edges()
                if current_count > best_count:
                    best_count = current_count
                    best_adj_mask = list(adj_mask)
            else:
                break
    
    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (1 << v) & best_adj_mask[u]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
