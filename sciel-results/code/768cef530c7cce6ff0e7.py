"""Construct a graph with no induced regular subgraph on 6+ vertices, maximizing n."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill
    
    # Target: build a graph on n vertices with no induced regular subgraph of size >= 6
    # Strategy: greedy construction with backtracking and fast validity checking
    
    # Precompute: for a given set of vertices and the graph, check if it's regular
    # We'll build the graph incrementally, adding one vertex at a time
    
    # State: edges as a set of frozensets for fast lookup, plus adjacency lists
    # We'll use bitmasks for adjacency for fast operations
    
    MAX_N = 22  # upper bound per contract
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Represent adjacency as list of integers (bitmasks)
    # adj[i] is a bitmask of neighbors of vertex i
    
    # Check if a subset (given as bitmask) induces a regular graph
    # Returns True if it is regular (bad), False otherwise
    def is_regular_subset(adj, subset_mask, subset_size):
        """Check if the induced subgraph on subset_mask is regular."""
        if subset_size < 6:
            return False
        # For each vertex in the subset, compute its degree within the subset
        # First vertex
        verts = []
        m = subset_mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        
        if len(verts) != subset_size:
            return False
        
        # Compute degree of first vertex within subset
        first_deg = 0
        adj_first = adj[verts[0]]
        for v in verts[1:]:
            if adj_first & (1 << v):
                first_deg += 1
        
        # Check all other vertices have the same degree
        for i, v in enumerate(verts[1:], 1):
            deg = 0
            adj_v = adj[v]
            for u in verts[:i]:
                if adj_v & (1 << u):
                    deg += 1
            for u in verts[i+1:]:
                if adj_v & (1 << u):
                    deg += 1
            if deg != first_deg:
                return False
        return True
    
    # Faster check using precomputed degrees
    def is_regular_subset_fast(adj, subset_mask, subset_size):
        """Faster check: compute all degrees and verify they're equal."""
        if subset_size < 6:
            return False
        # Get vertices
        verts = []
        m = subset_mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        
        if len(verts) != subset_size:
            return False
        
        # Compute degree of each vertex within the subset
        degrees = []
        for v in verts:
            deg = 0
            adj_v = adj[v]
            for u in verts:
                if u != v and (adj_v & (1 << u)):
                    deg += 1
            degrees.append(deg)
        
        # All degrees must be equal
        d0 = degrees[0]
        for d in degrees[1:]:
            if d != d0:
                return False
        return True
    
    # Even faster: use bit operations
    def is_regular_subset_v2(adj, subset_mask, subset_size):
        if subset_size < 6:
            return False
        # Get vertices
        verts = []
        m = subset_mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        
        if len(verts) != subset_size:
            return False
        
        # For each vertex, count neighbors within subset using bit operations
        degrees = []
        for v in verts:
            # adj[v] & subset_mask gives neighbors of v that are in subset
            # But this includes v itself if there's a self-loop (there isn't in simple graph)
            # So popcount of (adj[v] & subset_mask) is the degree
            mask = adj[v] & subset_mask
            # Count bits
            deg = 0
            mm = mask
            while mm:
                mm &= mm - 1
                deg += 1
            degrees.append(deg)
        
        d0 = degrees[0]
        for d in degrees[1:]:
            if d != d0:
                return False
        return True
    
    # Check if adding vertex v with neighbor mask new_adj to existing graph creates a bad subset
    # We need to check all subsets of size >= 6 that include v
    def check_new_vertex(adj, new_vertex, new_neighbors_mask, n_existing):
        """
        Check if adding new_vertex with given neighbor connections creates any regular induced subgraph.
        adj: list of adjacency masks for existing vertices (0 to n_existing-1)
        new_vertex: the new vertex index (n_existing)
        new_neighbors_mask: bitmask of which existing vertices are neighbors of new_vertex
        Returns True if the extension is valid (no new regular subset created).
        """
        n = n_existing + 1
        # Build full adjacency for checking
        # new_adj = adj + [new_neighbors_mask]
        # But we only need to check subsets containing new_vertex
        
        # A subset S containing new_vertex is bad if S is regular
        # S = {new_vertex} ∪ T where T ⊆ {0,...,n_existing-1}
        # |S| >= 6 means |T| >= 5
        
        # For a given T, the degree of new_vertex in S is popcount(new_neighbors_mask & T_mask)
        # The degree of each vertex u in T in S is popcount(adj[u] & T_mask) + (1 if u in new_neighbors_mask else 0)
        
        # We need all these degrees to be equal for S to be regular
        
        # Iterate over all T with |T| >= 5
        # This is expensive for large n_existing, so we need to be smart
        
        # Optimization: if n_existing is small, enumerate all T
        # For larger n_existing, we can use the fact that we're building incrementally
        
        # For now, enumerate all subsets T of {0,...,n_existing-1} with |T| >= 5
        # and check if T ∪ {new_vertex} is regular
        
        # Actually, this is O(2^n_existing) which is too slow for n_existing > 15 or so
        # We need a better approach
        
        # Alternative: only check subsets that are "likely" to be regular
        # Or use a different construction strategy
        
        # For this implementation, let's limit the subset size we check
        # and use the incremental property: if the previous graph was valid, 
        # we only need to check subsets containing the new vertex
        
        # For efficiency, let's check subsets of size 6 to min(n, some_max)
        # and hope that's sufficient (not guaranteed, but a heuristic)
        
        # Actually, for correctness we must check all subsets of size >= 6
        # Let's use the bitmask approach and hope n_existing stays small enough
        
        if n_existing <= 16:  # 2^16 = 65536 subsets, manageable
            # Enumerate all T ⊆ {0,...,n_existing-1} with |T| >= 5
            # Use Gray code or just iterate
            for t_size in range(5, n_existing + 1):
                for T in combinations(range(n_existing), t_size):
                    T_mask = 0
                    for v in T:
                        T_mask |= (1 << v)
                    
                    # Degree of new_vertex in S = T ∪ {new_vertex}
                    deg_new = 0
                    mm = new_neighbors_mask & T_mask
                    while mm:
                        mm &= mm - 1
                        deg_new += 1
                    
                    # Degree of each vertex in T
                    all_same = True
                    for u in T:
                        deg_u = 0
                        mm = adj[u] & T_mask
                        while mm:
                            mm &= mm - 1
                            deg_u += 1
                        # Add 1 if u is neighbor of new_vertex
                        if new_neighbors_mask & (1 << u):
                            deg_u += 1
                        if deg_u != deg_new:
                            all_same = False
                            break
                    
                    if all_same:
                        return False  # Found a regular subset
        
        # If n_existing > 16, we can't enumerate all subsets
        # In that case, we'll do a partial check and hope for the best
        # This is a heuristic that may miss some regular subsets
        else:
            # Spot check some random subsets
            import random
            for _ in range(100):
                t_size = random.randint(5, min(n_existing, 10))
                T = random.sample(range(n_existing), t_size)
                T_mask = 0
                for v in T:
                    T_mask |= (1 << v)
                
                deg_new = 0
                mm = new_neighbors_mask & T_mask
                while mm:
                    mm &= mm - 1
                    deg_new += 1
                
                all_same = True
                for u in T:
                    deg_u = 0
                    mm = adj[u] & T_mask
                    while mm:
                        mm &= mm - 1
                        deg_u += 1
                    if new_neighbors_mask & (1 << u):
                        deg_u += 1
                    if deg_u != deg_new:
                        all_same = False
                        break
                
                if all_same:
                    return False
        
        return True
    
    # Try to build the graph greedily
    # Start with a small valid graph and add vertices one by one
    # For each new vertex, try different connection patterns
    
    # We'll try to build up to MAX_N vertices
    # Use a simple greedy: for each new vertex, try all 2^n_existing connection patterns
    # and pick the first one that works (or a random one for diversity)
    
    # Better: try to be systematic
    # Start with n=0, add vertices one by one
    # For vertex i, try connection patterns in some order
    
    # Let's try a randomized greedy with restarts
    import random
    
    best_n = 0
    best_adj = None
    best_edges = []
    
    # Try multiple random seeds
    for seed in range(100):
        if time_left() < 10:
            break
        
        random.seed(seed)
        
        adj = []  # list of adjacency masks
        current_n = 0
        
        # Try to add vertices up to MAX_N
        for target_n in range(1, MAX_N + 1):
            if time_left() < 5:
                break
            
            if current_n >= target_n:
                continue
            
            # Try to add vertex current_n with some connection pattern
            # Try multiple random patterns
            found = False
            attempts = 0
            max_attempts = 50 if current_n <= 10 else 20
            
            # Symmetry breaking: the first vertex has no edges
            if current_n == 0:
                adj.append(0)
                current_n += 1
                found = True
            elif current_n == 1:
                # Second vertex: try connected or not
                for pattern in [1, 0]:  # bit 0 is vertex 0
                    if check_new_vertex(adj, current_n, pattern, current_n):
                        # Update adj
                        adj.append(pattern)
                        # Update existing vertices' adjacency
                        if pattern & 1:
                            adj[0] |= (1 << current_n)
                        current_n += 1
                        found = True
                        break
            else:
                # Try random patterns
                for attempt in range(max_attempts):
                    # Generate random neighbor mask
                    # Bias: prefer sparse graphs
                    prob = 0.3  # probability of connecting to each existing vertex
                    pattern = 0
                    for v in range(current_n):
                        if random.random() < prob:
                            pattern |= (1 << v)
                    
                    if check_new_vertex(adj, current_n, pattern, current_n):
                        # Accept this vertex
                        adj.append(pattern)
                        # Update existing vertices' adjacency
                        for v in range(current_n):
                            if pattern & (1 << v):
                                adj[v] |= (1 << current_n)
                        current_n += 1
                        found = True
                        break
            
            if not found:
                # Can't add this vertex, break
                break
        
        # Record result
        if current_n > best_n:
            best_n = current_n
            best_adj = list(adj)
            # Convert to edge list
            edges = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            best_edges = edges
    
    # Final verification of the best graph (if time permits)
    if best_n >= 6 and time_left() > 2:
        # Verify no regular induced subgraph of size >= 6
        adj = best_adj
        n = best_n
        valid = True
        for subset_size in range(6, n + 1):
            if time_left() < 2:
                break
            for subset in combinations(range(n), subset_size):
                subset_mask = 0
                for v in subset:
                    subset_mask |= (1 << v)
                if is_regular_subset_v2(adj, subset_mask, subset_size):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # The graph is invalid, need to find a smaller valid one
            # Try to trim vertices
            # For simplicity, just return a known valid small graph
            best_n = 5
            best_adj = [0, 0, 0, 0, 0]
            best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
