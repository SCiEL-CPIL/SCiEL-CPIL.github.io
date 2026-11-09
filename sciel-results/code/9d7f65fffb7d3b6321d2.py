import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit
    n = 20
    
    def time_left():
        return deadline - time.time()
    
    # Precompute all pairs for faster checking
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def edge_count(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2
    
    def find_violation(adj):
        """Find a K3,3 violation. Returns (a,b,c, common_mask) or None."""
        for a in range(n):
            for b in range(a+1, n):
                ab = adj[a] & adj[b]
                if popcount(ab) < 3:
                    continue
                for c in range(b+1, n):
                    common = ab & adj[c]
                    if popcount(common) >= 3:
                        return (a, b, c, common)
        return None
    
    def can_add_fast(adj, pair_counts, u, v):
        """Check if adding edge (u,v) creates a K3,3.
        
        A K3,3 using new edge (u,v): one part {u, x, y}, other {v, p, q}.
        x,y must be neighbors of v; p,q must be neighbors of u.
        Need edges x-p, x-q, y-p, y-q to exist.
        
        Equivalently: exists x,y in N(v)\{u} with |N(x) ∩ N(y) ∩ N(u)| >= 2.
        """
        # Get neighbors of u and v (excluding each other)
        nu_mask = adj[u] & ~(1 << v)
        nv_mask = adj[v] & ~(1 << u)
        
        # For each pair (x,y) in N(v), count common neighbors in N(u)
        nv_list = []
        mask = nv_mask
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            nv_list.append(w)
            mask ^= lb
        
        if len(nv_list) < 2:
            return True  # can add
        
        for i in range(len(nv_list)):
            for j in range(i+1, len(nv_list)):
                x, y = nv_list[i], nv_list[j]
                # Count neighbors of both x and y that are also neighbors of u
                common = adj[x] & adj[y] & nu_mask
                if popcount(common) >= 2:
                    return False
        return True
    
    def update_pair_counts(pair_counts, adj, u, v, added):
        """Update pair_counts when adding/removing edge (u,v).
        pair_counts[a,b] = popcount(adj[a] & adj[b]) for a < b
        """
        bit_u = 1 << u
        bit_v = 1 << v
        if added:
            adj[u] |= bit_v
            adj[v] |= bit_u
        else:
            adj[u] &= ~bit_v
            adj[v] &= ~bit_u
        
        # Update pair_counts for all pairs involving u or v
        for w in range(n):
            if w == u or w == v:
                continue
            # Pair (min(u,w), max(u,w))
            a, b = min(u, w), max(u, w)
            pair_counts[a*n+b] = popcount(adj[a] & adj[b])
            # Pair (min(v,w), max(v,w))
            a, b = min(v, w), max(v, w)
            pair_counts[a*n+b] = popcount(adj[a] & adj[b])
        # Pair (u, v)
        pair_counts[u*n+v] = popcount(adj[u] & adj[v])
    
    def make_pair_counts(adj):
        pc = [0] * (n*n)
        for i in range(n):
            for j in range(i+1, n):
                pc[i*n+j] = popcount(adj[i] & adj[j])
        return pc
    
    # === Structured initialization: polarity graph of PG(2,4) ===
    # PG(2,4) has 21 points. We'll use 20 of them.
    # Points of PG(2,4): 21 points, each line has 5 points, each point on 5 lines.
    # Polarity: point p is NOT adjacent to p, and p adjacent to q iff p is on line through q (in the polarity sense).
    # Actually for a polarity graph: p ~ q iff p lies on the polar line of q.
    # This gives a graph where each vertex has degree 5 (since each line has 5 points, 
    # and the polar line of a point contains 5 points, but the point itself is on its own polar line 
    # for a symplectic polarity... let me use a simpler construction).
    
    # Alternative: use a known K3,3-free graph structure.
    # The complete bipartite graph K_{4,4} minus a perfect matching is K3,3-free? No, it contains K3,3.
    
    # Let's use: start with a random dense graph and repair.
    # But use a better initial structure: take two copies of K_{10} with controlled cross-edges.
    
    # Actually, let's try: start from empty, greedily add edges in a smart order.
    
    random.seed(123)
    
    best_adj = None
    best_count = 0
    best_pair_counts = None
    
    for attempt in range(30):
        if time_left() < 30:
            break
        
        if attempt == 0:
            # Start with a structured base: cycle + chords
            adj = [0] * n
            for i in range(n):
                for d in [1, 2, 3]:
                    j = (i + d) % n
                    if j > i:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        else:
            # Random initialization
            p = 0.4 + 0.5 * random.random()
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Repair: remove edges to fix violations
        for _ in range(300):
            if time_left() < 25:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            a, b, c, common = viol
            # Remove an edge that breaks the K3,3
            # The K3,3 has parts {a,b,c} and 3 vertices from common
            common_list = []
            mask = common
            while mask:
                lb = mask & (-mask)
                w = lb.bit_length() - 1
                common_list.append(w)
                mask ^= lb
            
            # Remove edge from a to first common vertex
            remove_target = common_list[0]
            if (adj[a] >> remove_target) & 1:
                adj[a] &= ~(1 << remove_target)
                adj[remove_target] &= ~(1 << a)
            else:
                # Try removing from b
                if (adj[b] >> remove_target) & 1:
                    adj[b] &= ~(1 << remove_target)
                    adj[remove_target] &= ~(1 << b)
                elif common_list[1] if len(common_list) > 1 else None:
                    adj[a] &= ~(1 << common_list[1])
                    adj[common_list[1]] &= ~(1 << a)
        
        # Build pair counts
        pc = make_pair_counts(adj)
        
        # Greedy add: try to add all possible edges
        changed = True
        while changed:
            if time_left() < 20:
                break
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] >> j) & 1:
                        if can_add_fast(adj, pc, i, j):
                            update_pair_counts(pc, adj, i, j, True)
                            changed = True
        
        # Local search with kicks
        for kick in range(50):
            if time_left() < 15:
                break
            
            # Try to add random non-edges
            for _ in range(200):
                if time_left() < 10:
                    break
                i, j = random.sample(range(n), 2)
                if not (adj[i] >> j) & 1:
                    if can_add_fast(adj, pc, i, j):
                        update_pair_counts(pc, adj, i, j, True)
            
            # Kick: remove a random edge and re-optimize
            if random.random() < 0.3:
                # Find a random edge to remove
                edges_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges_list.append((i, j))
                if edges_list:
                    i, j = random.choice(edges_list)
                    update_pair_counts(pc, adj, i, j, False)
                    # Re-greedy
                    changed2 = True
                    while changed2:
                        if time_left() < 10:
                            break
                        changed2 = False
                        for a in range(n):
                            for b in range(a+1, n):
                                if not (adj[a] >> b) & 1:
                                    if can_add_fast(adj, pc, a, b):
                                        update_pair_counts(pc, adj, a, b, True)
                                        changed2 = True
        
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = adj[:]
            best_pair_counts = pc[:]
    
    # Final greedy pass on best
    if best_adj is not None and time_left() > 10:
        pc = make_pair_counts(best_adj)
        changed = True
        while changed:
            if time_left() < 5:
                break
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (best_adj[i] >> j) & 1:
                        if can_add_fast(best_adj, pc, i, j):
                            update_pair_counts(pc, best_adj, i, j, True)
                            changed = True
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
