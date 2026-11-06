import time
import random
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = 0
    best_graph = None
    
    # Precompute triples for violation check
    triples = list(itertools.combinations(range(n), 3))
    
    def is_valid(adj):
        # adj is list of int bitmasks
        for u, v, w in triples:
            # Check if there are 3 common neighbors
            common = adj[u] & adj[v] & adj[w]
            # Count bits
            if common.bit_count() >= 3:
                return False
        return True

    def find_violation(adj):
        for u, v, w in triples:
            common = adj[u] & adj[v] & adj[w]
            if common.bit_count() >= 3:
                # Get the 3 common neighbors
                neighbors = [i for i in range(n) if (common >> i) & 1]
                # We only need 3
                v_set = neighbors[:3]
                u_set = [u, v, w]
                return u_set, v_set
        return None, None

    def repair(adj):
        # Remove edges to make it valid
        while True:
            if time.time() > deadline:
                return False
            u_set, v_set = find_violation(adj)
            if u_set is None:
                return True
            # We have a K3,3 between u_set and v_set
            # Candidate edges are all (u, v) for u in u_set, v in v_set
            candidates = [(u, v) for u in u_set for v in v_set]
            
            # Choose edge to remove
            # Heuristic: Remove edge with highest sum of degrees
            best_edge = None
            max_score = -1
            for u, v in candidates:
                # Score: degree(u) + degree(v)
                score = adj[u].bit_count() + adj[v].bit_count()
                if score > max_score:
                    max_score = score
                    best_edge = (u, v)
            
            # Remove best_edge
            u, v = best_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
    def is_add_valid(adj, u, v):
        """Check if adding edge (u,v) creates a K3,3 violation.
        Only need to check triples where both u and v are common neighbors of at least 2 vertices.
        """
        # After adding (u,v), u and v become neighbors of each other.
        # A K3,3 violation would involve 3 vertices A,B,C that have 3 common neighbors.
        # The new edge (u,v) means u is now a neighbor of v and v is a neighbor of u.
        # So we need to check if there exist 3 vertices x,y,z such that
        # {u,v} plus one more vertex are all common neighbors of x,y,z.
        # Equivalently: for each pair (x,y) among the other n-2 vertices,
        # check if u, v, and some w are all neighbors of both x and y.
        # But more efficiently: for each triple (a,b,c), check if common neighbors >= 3.
        # The only new common neighbors created by adding (u,v) are:
        # - u becomes a common neighbor of any pair that both neighbor v
        # - v becomes a common neighbor of any pair that both neighbor u
        # So we only need to check triples where at least one of u,v is a "new" common neighbor.
        
        # Faster approach: check all triples, but we can optimize by noting
        # that only triples where u or v is involved in the common neighbor set matter.
        # Actually, the safest fast check: for each triple (a,b,c), compute common = adj[a]&adj[b]&adj[c],
        # but we haven't added the edge yet. After adding, common might gain u or v.
        # u is a neighbor of a,b,c iff a,b,c all neighbor v (since u-v edge added)
        # v is a neighbor of a,b,c iff a,b,c all neighbor u
        
        # Let's just do the full check but it's O(1140) which is fine
        # But we can be smarter: only check triples where the potential new common neighbor matters
        
        # For triple (a,b,c): after adding (u,v), common neighbors = old_common ∪ {u if v~a,b,c} ∪ {v if u~a,b,c}
        # u is common neighbor of a,b,c iff (adj[v] & (1<<a) & (1<<b) & (1<<c)) != 0
        # v is common neighbor of a,b,c iff (adj[u] & (1<<a) & (1<<b) & (1<<c)) != 0
        
        for a, b, c in triples:
            common = adj[a] & adj[b] & adj[c]
            count = common.bit_count()
            # Check if u becomes a new common neighbor
            if count < 3:
                if (adj[v] & (1 << a)) and (adj[v] & (1 << b)) and (adj[v] & (1 << c)):
                    count += 1
            if count < 3:
                if (adj[u] & (1 << a)) and (adj[u] & (1 << b)) and (adj[u] & (1 << c)):
                    count += 1
            if count >= 3:
                return False
        return True

    def improve(adj):
        # Try to add edges - exhaustive greedy in each pass
        improved = True
        while improved:
            if time.time() > deadline:
                return False
            improved = False
            edges_to_check = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
            random.shuffle(edges_to_check)
            
            for u, v in edges_to_check:
                if time.time() > deadline:
                    return False
                # Try adding edge (u, v)
                if is_add_valid(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    # Continue trying to add more edges in this pass
        return True

    def try_swap(adj, remove_edge, add_edge):
        """Try swapping: remove one edge, add another. Return True if successful."""
        u1, v1 = remove_edge
        u2, v2 = add_edge
        
        # Remove u1-v1
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        
        # Try to add u2-v2
        if is_add_valid(adj, u2, v2):
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            return True
        else:
            # Revert
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            return False

    def swap_improve(adj, num_swaps=200):
        """Try edge swaps to improve the graph."""
        improved = True
        while improved and num_swaps > 0:
            if time.time() > deadline:
                return False
            improved = False
            num_swaps -= 1
            
            # Get current edges
            current_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
            missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
            
            if not current_edges or not missing_edges:
                break
            
            # Try random swaps
            for _ in range(50):
                if time.time() > deadline:
                    return False
                rem = random.choice(current_edges)
                add = random.choice(missing_edges)
                if try_swap(adj, rem, add):
                    improved = True
                    break
        return True

    # Run multiple trials with different random seeds
    for seed in range(500):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Start with random graph
        p = 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        if repair(adj):
            # Improve with greedy add
            improve(adj)
            
            # Try swaps to escape local optima
            swap_improve(adj, num_swaps=100)
            
            # Final greedy add after swaps
            improve(adj)
            
            # Check validity again just in case
            if is_valid(adj):
                edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_graph = [adj[i] for i in range(n)]
        
        # If we hit a high number, we can stop early?
        # Theoretical max is ~80-90.
        if best_edges > 85:
            break

    # Convert best_graph to list of edges
    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_graph[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
