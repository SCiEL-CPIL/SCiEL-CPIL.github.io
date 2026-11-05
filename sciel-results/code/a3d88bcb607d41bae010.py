import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n = 20
    # Use bitmasks for adjacency
    # adj[i] is a bitmask of neighbors of vertex i
    
    def is_valid(adj):
        """Check if graph is K3,3-free: no 3 vertices have 3+ common neighbors."""
        for a in range(n):
            for b in range(a+1, n):
                common_ab = adj[a] & adj[b]
                for c in range(b+1, n):
                    common_abc = common_ab & adj[c]
                    # Count bits
                    if (common_abc & (common_abc - 1)) != 0 and (common_abc & (common_abc - 1) & (common_abc - 1)) != 0:
                        # At least 3 bits set
                        return False
        return True
    
    def count_common3(adj):
        """Count number of triples with 3+ common neighbors."""
        cnt = 0
        for a in range(n):
            for b in range(a+1, n):
                common_ab = adj[a] & adj[b]
                for c in range(b+1, n):
                    common_abc = common_ab & adj[c]
                    # Count bits in common_abc
                    bits = common_abc
                    while bits:
                        bits &= bits - 1
                        cnt += 1
        return cnt
    
    def edge_count(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v), return True if valid."""
        if u == v or (adj[u] >> v) & 1:
            return False
        # Adding edge (u,v): check if this creates a K3,3
        # New K3,3 would involve u or v as part of a triple with 3 common neighbors
        # Actually, we need to check: after adding (u,v), is there any triple with 3+ common neighbors?
        # The only new triples involving the new edge are those where u and v are in the triple,
        # or where the new edge creates a new common neighbor for some triple.
        # 
        # More precisely: after adding (u,v), a K3,3 exists iff there's a triple {a,b,c} 
        # with |N(a)∩N(b)∩N(c)| >= 3.
        # The new edge can only affect triples that include u or v.
        # 
        # Case 1: u and v are both in the triple: {u,v,c} for some c
        #   N(u)∩N(v)∩N(c) must have size >= 3
        #   N(u)∩N(v) now includes the other's... wait, N(u) includes v now, N(v) includes u now.
        #   So N(u)∩N(v) = (old N(u)∩N(v)) ∪ {nothing new from the edge itself since u∉N(u)}
        #   Actually u is not in N(u), v is not in N(v). The new edge adds v to N(u) and u to N(v).
        #   So N(u)∩N(v) gains: v is in N(u) but is v in N(v)? No. u is in N(v) but is u in N(u)? No.
        #   So N(u)∩N(v) doesn't change! The common neighbors of u and v are unchanged.
        #   Wait, that's not right. N(u)∩N(v) = vertices adjacent to both u and v.
        #   Before: a vertex w is in N(u)∩N(v) iff w~u and w~v.
        #   After adding (u,v): w~u and w~v still the same condition. u~v doesn't affect whether w is adjacent to u or v.
        #   So N(u)∩N(v) is unchanged.
        #   
        #   But wait, for triple {u,v,c}: common neighbors = N(u)∩N(v)∩N(c). This is unchanged.
        #   
        # Case 2: u is in the triple but v is not: {u,c,d}
        #   Common neighbors = N(u)∩N(c)∩N(d). 
        #   N(u) gained v. So if v ∈ N(c)∩N(d), then v is a new common neighbor.
        #   So |N(u)∩N(c)∩N(d)| increases by 1 iff v~c and v~d.
        #   So we need: for no pair c,d (c≠d, c≠u, d≠u, c≠v, d≠v) do we have 
        #   v~c and v~d and |old N(u)∩N(c)∩N(d)| >= 2.
        #   
        # Case 3: v is in the triple but u is not: symmetric to Case 2.
        #
        # So the check is:
        # 1. For all c: |N(u)∩N(v)∩N(c)| <= 2 (but this is unchanged, so just check current)
        #    Actually we should verify the current graph is valid first.
        # 2. For all pairs c,d (c<d, c≠u, d≠u, c≠v, d≠v): if v~c and v~d, 
        #    then |N(u)∩N(c)∩N(d)| <= 1 (since adding v would make it 2+1=3 if it was 2)
        #    Wait: if |old N(u)∩N(c)∩N(d)| = k, and v~c and v~d, then new count = k + 1 (v is added to N(u)).
        #    We need new count <= 2, so k <= 1.
        # 3. Symmetric: for all pairs c,d: if u~c and u~d, then |N(v)∩N(c)∩N(d)| <= 1.
        
        # Check case 1: triples containing both u and v
        common_uv = adj[u] & adj[v]
        for c in range(n):
            if c == u or c == v:
                continue
            common_abc = common_uv & adj[c]
            if bin(common_abc).count('1') >= 3:
                return False
        
        # Check case 2: triples containing u but not v
        for c in range(n):
            if c == u or c == v:
                continue
            if not ((adj[v] >> c) & 1):
                continue
            for d in range(c+1, n):
                if d == u or d == v:
                    continue
                if not ((adj[v] >> d) & 1):
                    continue
                common = adj[u] & adj[c] & adj[d]
                if bin(common).count('1') >= 2:
                    return False
        
        # Check case 3: triples containing v but not u
        for c in range(n):
            if c == u or c == v:
                continue
            if not ((adj[u] >> c) & 1):
                continue
            for d in range(c+1, n):
                if d == u or d == v:
                    continue
                if not ((adj[u] >> d) & 1):
                    continue
                common = adj[v] & adj[c] & adj[d]
                if bin(common).count('1') >= 2:
                    return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    best_adj = None
    best_edges = 0
    
    # Strategy: start with various initial graphs, then do local search
    # Initial: random graphs with varying density
    
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random graph
        if attempt == 0:
            # Start with complete bipartite K10,10
            adj = [0] * n
            for i in range(10):
                for j in range(10, 20):
                    add_edge(adj, i, j)
        elif attempt == 1:
            # Start with random 0.5 density
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        add_edge(adj, i, j)
        elif attempt == 2:
            # Start with random 0.6 density
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.6:
                        add_edge(adj, i, j)
        else:
            # Random with varying density
            density = 0.3 + 0.4 * random.random()
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < density:
                        add_edge(adj, i, j)
        
        # Repair: remove edges until valid
        max_repair_iters = 10000
        iters = 0
        while iters < max_repair_iters:
            if time.time() > deadline:
                break
            # Find a violation
            violation_found = False
            for a in range(n):
                for b in range(a+1, n):
                    common_ab = adj[a] & adj[b]
                    for c in range(b+1, n):
                        common_abc = common_ab & adj[c]
                        if bin(common_abc).count('1') >= 3:
                            # Remove a random edge from the K3,3
                            # The K3,3 involves {a,b,c} and 3 of their common neighbors
                            neighbors = []
                            bits = common_abc
                            while bits:
                                lsb = bits & (-bits)
                                v = lsb.bit_length() - 1
                                neighbors.append(v)
                                bits ^= lsb
                            # Pick 3 common neighbors
                            if len(neighbors) >= 3:
                                # Remove edge between a and a random common neighbor
                                v = random.choice(neighbors[:3])
                                remove_edge(adj, a, v)
                                violation_found = True
                                break
                    if violation_found:
                        break
                if violation_found:
                    break
            if not violation_found:
                break
            iters += 1
        
        # Now adj should be K3,3-free. Do local search to add edges.
        if time.time() > deadline:
            break
        
        # Local search: repeatedly try to add random edges
        no_improve = 0
        max_no_improve = 500
        while no_improve < max_no_improve:
            if time.time() > deadline:
                break
            
            # Try to add a random non-edge
            improved = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or (adj[u] >> v) & 1:
                    continue
                if try_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
                    break
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        
        # Track best
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
        
        # Also try a polish: systematic edge addition
        if time.time() > deadline:
            break
        
        polish_adj = best_adj[:]
        for u in range(n):
            for v in range(u+1, n):
                if time.time() > deadline:
                    break
                if (polish_adj[u] >> v) & 1:
                    continue
                if try_add_edge(polish_adj, u, v):
                    add_edge(polish_adj, u, v)
        
        ec2 = edge_count(polish_adj)
        if ec2 > best_edges:
            best_edges = ec2
            best_adj = polish_adj
    
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
