import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 20
    
    def bipartite_spectral_init(seed):
        """
        Create a bipartite graph using spectral methods, then add intra-partition edges.
        This naturally limits common neighbors across the partition.
        """
        rng = np.random.RandomState(seed)
        
        # Split into two parts of size 10
        part_a = list(range(10))
        part_b = list(range(10, 20))
        
        # Create a random bipartite adjacency matrix
        # Use a regular-like structure to balance degrees
        B = np.zeros((10, 10), dtype=int)
        
        # Start with a random permutation-based structure (like a Latin square)
        for i in range(10):
            # Each vertex in A connects to ~5 vertices in B
            targets = rng.choice(10, size=5, replace=False)
            for t in targets:
                B[i, t] = 1
        
        # Build adjacency from bipartite part
        adj = [0] * n
        for i in range(10):
            for j in range(10):
                if B[i, j]:
                    u = part_a[i]
                    v = part_b[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Add some intra-partition edges (randomly)
        p_intra = 0.3
        for i in range(10):
            for j in range(i + 1, 10):
                if rng.random() < p_intra:
                    adj[part_a[i]] |= (1 << part_a[j])
                    adj[part_a[j]] |= (1 << part_a[i])
                if rng.random() < p_intra:
                    adj[part_b[i]] |= (1 << part_b[j])
                    adj[part_b[j]] |= (1 << part_b[i])
        
        return adj
    
    def get_violating_triple(adj):
        """
        Returns a triple (i, j, k) such that common neighbors >= 3, or None.
        """
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = adj[i] & adj[j]
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        return (i, j, k, common_ijk)
        return None
    
    def repair(adj):
        """
        Removes edges until the graph is valid.
        """
        while True:
            v = get_violating_triple(adj)
            if v is None:
                break
            i, j, k, common = v
            # Remove an edge to break the K3,3
            candidates = []
            for m in range(n):
                if common & (1 << m):
                    candidates.append(m)
            
            m = random.choice(candidates)
            # Remove edge (i, m)
            adj[i] &= ~(1 << m)
            adj[m] &= ~(1 << i)
    
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) keeps the graph K3,3-free.
        After adding (u,v), for any pair (a,b) where a,b != u,v:
        common(a,b) gains u if both a,b were neighbors of v, and gains v if both were neighbors of u.
        Actually simpler: after adding (u,v), the only new common neighbor contributions are:
        - u becomes a common neighbor of any pair that includes v and another neighbor of v
        - v becomes a common neighbor of any pair that includes u and another neighbor of u
        
        More precisely: for any triple (i,j,k), the common neighbors can only increase if one of i,j,k is u or v.
        So we only need to check triples containing u or v.
        
        Triples containing u: (u, a, b) where a,b are other vertices.
        Common neighbors of (u,a,b) after adding (u,v) = (N(u) ∪ {v}) ∩ N(a) ∩ N(b)
        = (N(u) ∩ N(a) ∩ N(b)) ∪ ({v} ∩ N(a) ∩ N(b))
        = old_common(u,a,b) ∪ ({v} if v in N(a) and v in N(b) else empty)
        
        So the count increases by 1 if v is in N(a) and N(b).
        We need old_count + [v in N(a) and v in N(b)] < 3 for all pairs (a,b) with a,b != u.
        
        Similarly for triples (a, u, b): same thing.
        And triples (a, b, u): same.
        
        Also triples containing v: (v, a, b) where a,b != v.
        Common neighbors = (N(v) ∪ {u}) ∩ N(a) ∩ N(b)
        = old_common(v,a,b) ∪ ({u} if u in N(a) and u in N(b) else empty)
        
        And triples containing both u and v: (u, v, a) for any a != u,v.
        Common neighbors = (N(u) ∪ {v}) ∩ (N(v) ∪ {u}) ∩ N(a)
        = ((N(u) ∩ N(v) ∩ N(a)) ∪ (N(u) ∩ {u} ∩ N(a)) ∪ ({v} ∩ N(v) ∩ N(a)) ∪ ({v} ∩ {u} ∩ N(a)))
        Wait, this is getting complex. Let me just compute directly.
        
        For triple (u, v, a):
        N(u) after add = N(u) ∪ {v}
        N(v) after add = N(v) ∪ {u}
        Common = (N(u) ∪ {v}) ∩ (N(v) ∪ {u}) ∩ N(a)
        
        Let me just check all triples that include u or v.
        """
        # Check all triples that include u or v
        # Triples with u (but not v): (u, a, b) where a < b, a,b != u,v
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                # Triple (u, a, b)
                # New common = (adj[u] | (1<<v)) & adj[a] & adj[b]
                new_common = (adj[u] | (1 << v)) & adj[a] & adj[b]
                if new_common.bit_count() >= 3:
                    return False
                # Triple (a, u, b) is same set
                # Triple (a, b, u) is same set
        
        # Triples with v (but not u): (v, a, b) where a < b, a,b != u,v
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                # Triple (v, a, b)
                new_common = (adj[v] | (1 << u)) & adj[a] & adj[b]
                if new_common.bit_count() >= 3:
                    return False
        
        # Triples with both u and v: (u, v, a) for a != u,v
        for a in range(n):
            if a == u or a == v:
                continue
            # Triple (u, v, a)
            new_common = (adj[u] | (1 << v)) & (adj[v] | (1 << u)) & adj[a]
            if new_common.bit_count() >= 3:
                return False
        
        return True
    
    def greedy_fill(adj, current_edges, deadline):
        """
        Greedily add edges that don't violate K3,3-free property.
        Returns updated adj and edge count.
        """
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        continue
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges += 1
                        changed = True
        return adj, current_edges
    
    def kick(adj, current_edges, num_removals, deadline):
        """
        Remove some random edges to escape local optimum.
        """
        for _ in range(num_removals):
            if time.time() > deadline - 2.0:
                break
            # Pick a random existing edge
            while True:
                u = random.randint(0, n - 1)
                if adj[u] == 0:
                    continue
                neighbors = []
                for w in range(n):
                    if adj[u] & (1 << w):
                        neighbors.append(w)
                if neighbors:
                    v = random.choice(neighbors)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
                    break
        return adj, current_edges
    
    best_edges = 0
    best_adj = None
    
    num_restarts = 50
    
    for seed in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize using bipartite spectral construction
        adj = bipartite_spectral_init(seed)
        
        # Repair to valid state
        repair(adj)
        
        current_edges = sum(adj[i].bit_count() for i in range(n)) // 2
        
        # Kick + greedy fill cycles
        num_cycles = 3
        for cycle in range(num_cycles):
            if time.time() > deadline - 3.0:
                break
            
            # Greedy fill
            adj, current_edges = greedy_fill(adj, current_edges, deadline)
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [x for x in adj]
            
            # Kick: remove some edges to escape local optimum
            if cycle < num_cycles - 1:
                num_removals = random.randint(2, 5)
                adj, current_edges = kick(adj, current_edges, num_removals, deadline)
        
        # Final greedy fill
        adj, current_edges = greedy_fill(adj, current_edges, deadline)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [x for x in adj]
    
    # If we haven't found anything, try a simple random start
    if best_adj is None:
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        repair(adj)
        adj, current_edges = greedy_fill(adj, sum(adj[i].bit_count() for i in range(n)) // 2, deadline)
        best_adj = adj
        best_edges = current_edges
    
    # Construct edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
