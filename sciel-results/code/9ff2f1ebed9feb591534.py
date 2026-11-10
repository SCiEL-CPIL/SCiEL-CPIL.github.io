import time
import random
import itertools

def construct():
    """Construct a graph with no regular induced subgraph on 6+ vertices, maximizing n."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def time_left():
        return deadline - time.time()

    def is_regular_subgraph(adj, subset):
        """
        Check if the induced subgraph on 'subset' is regular.
        adj: list of int bitmasks
        subset: tuple of vertex indices
        """
        if len(subset) < 2:
            return True # Trivially regular? Task says 6+. 
        # Build mask for the subset
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        # Calculate degree of first vertex in the subset
        # Degree is count of neighbors in 'mask'
        first_deg = (adj[subset[0]] & mask).bit_count()
        
        # Check if all other vertices have the same degree
        for v in subset[1:]:
            if (adj[v] & mask).bit_count() != first_deg:
                return False
        return True

    def find_violation(adj, n):
        """
        Find any subset of size >= 6 that induces a regular graph.
        Returns the subset (tuple) or None if valid.
        Prioritizes smaller subsets as they are more common and faster to check.
        """
        # We check sizes 6, 7, ..., n
        # For performance, we can stop checking larger sizes if we are confident 
        # or if time is running out, but correctness requires checking all.
        # However, if a graph has no regular 6-subsets, it's *likely* to have no 
        # larger ones, but not guaranteed. We must check all.
        
        # Optimization: Pre-compute combinations? 
        # itertools.combinations is implemented in C and is fast.
        
        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                if is_regular_subgraph(adj, subset):
                    return subset
        return None

    def try_solve(n, time_budget):
        """
        Attempt to find a valid graph on n vertices using local search.
        Returns edges list if found, else None.
        """
        end_time = time.time() + time_budget
        
        # Try multiple random restarts
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Initialize random graph
            adj = [0] * n
            # Use a probability that might avoid regular structures? 
            # 0.5 is standard. 
            p = 0.5
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local Search
            # We will iterate until we find a valid graph or run out of time/iterations
            max_iters = 10000
            for it in range(max_iters):
                if time.time() > end_time:
                    return None
                
                # Check for violation
                bad_subset = find_violation(adj, n)
                
                if bad_subset is None:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return edges
                
                # Flip a random edge within the bad subset
                # Picking an edge within the subset ensures we disrupt the local structure
                u, v = random.sample(list(bad_subset), 2)
                
                # Toggle edge (u, v)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        return None

    # Target sizes in descending order
    # We allocate more time to larger n
    target_ns = [21, 20, 19]
    
    for n in target_ns:
        if time_left() < 5:
            break
            
        # Allocate remaining time minus a buffer for the next iterations
        # Give 30s to n=21, 20s to n=20, etc.
        budget = min(30, time_left() - 5)
        
        edges = try_solve(n, budget)
        
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback: Return a small valid graph
    # A graph with n < 6 is trivially valid.
    # Let's return a random graph on 5 vertices to be safe and fast.
    n = 5
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
