import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    needed_in_B = target_edges - (a_size * b_size)  # 122
    
    def try_construct_B(max_deg, num_restarts=50):
        """Try to construct triangle-free G_B with max degree constraint."""
        best_result = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            B_edges = []
            degree = {v: 0 for v in B}
            adj = {v: set() for v in B}
            
            # Generate all possible pairs
            pairs = list(itertools.combinations(B, 2))
            random.shuffle(pairs)
            
            for u, v in pairs:
                if len(B_edges) == needed_in_B:
                    break
                
                # Check triangle constraint: no common neighbor
                if adj[u] & adj[v]:
                    continue
                    
                # Check degree constraint
                if degree[u] < max_deg and degree[v] < max_deg:
                    B_edges.append((u, v))
                    degree[u] += 1
                    degree[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
            
            if len(B_edges) == needed_in_B:
                # Verify max degree
                max_d = max(degree.values())
                if max_d <= max_deg:
                    return B_edges
                    
        return None
    
    # Try to get max degree 8 first (score = 8)
    B_edges = try_construct_B(8, num_restarts=100)
    
    if B_edges is None and time.time() < deadline - 10.0:
        # Fall back to max degree 9 (score = 9)
        B_edges = try_construct_B(9, num_restarts=20)
    
    if B_edges is None:
        # Last resort: just pick any 122 edges, no constraints
        B_edges = []
        pairs = list(itertools.combinations(B, 2))
        random.shuffle(pairs)
        for u, v in pairs[:needed_in_B]:
            B_edges.append((u, v))
    
    # Add B edges to main set
    for u, v in B_edges:
        edges.add((u, v))
        
    # Format output
    result = [[u, v] for u, v in edges]
    
    return result
