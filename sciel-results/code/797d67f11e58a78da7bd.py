import itertools
import time
import random

def construct():
    n = 20
    # Use a list of lists for adjacency, or a set of edges
    # For speed, use a 20x20 matrix or bitsets
    # Let's use a set of tuples for edges, and a 2D list for adjacency
    adj = [[1]*n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 0
        
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            edges.add((u, v))
            
    # Precompute triples
    triples = list(itertools.combinations(range(n), 3))
    
    # A[B] will be a list of sets, or we can just compute on the fly
    # Since n is small, we can maintain a list of common neighbors for each triple
    # Let's maintain common_neighbors[B] as a bitmask or list
    # Using integers as bitmasks for speed
    
    # adj_mask[i] is a bitmask of neighbors of i
    adj_mask = [0]*n
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)
                
    # common_neighbors for each triple
    # We can compute this on the fly or update it
    # Let's compute the count of K3,3 for each edge
    # cnt[e] = number of K3,3 containing e
    
    # Initialize counts
    # For each triple B, let A_B = {u | u connected to all b in B}
    # For each u in A_B, and each b in B, the edge (u,b) is in C(A_B \ {u}, 2) K3,3s associated with B
    # Wait, if A_B has size k, then for a specific u in A_B, the number of K3,3s containing (u,b) and B is C(k-1, 2).
    # This is because we need to choose 2 other vertices from A_B \ {u} to form the other part of the K3,3.
    
    edge_cnt = {}
    for e in edges:
        edge_cnt[e] = 0
        
    # Function to compute counts from scratch (for initialization or restart)
    def compute_counts():
        for e in edge_cnt:
            edge_cnt[e] = 0
        for B in triples:
            b1, b2, b3 = B
            # Find A_B
            # u is in A_B if adj_mask[u] has bits b1, b2, b3 set
            # We can iterate u from 0 to n-1
            A_B = []
            for u in range(n):
                if (adj_mask[u] >> b1) & 1 and (adj_mask[u] >> b2) & 1 and (adj_mask[u] >> b3) & 1:
                    A_B.append(u)
            k = len(A_B)
            if k >= 3:
                # For each u in A_B, for each b in B, add C(k-1, 2) to edge_cnt[(u,b)]
                # Note: u cannot be in B because B vertices are not connected to themselves (adj[i][i]=0)
                # So u is always distinct from b1, b2, b3
                add_val = (k-1)*(k-2)//2
                for u in A_B:
                    for b in B:
                        e = (min(u,b), max(u,b))
                        edge_cnt[e] += add_val
                        
    compute_counts()
    
    # Greedy removal
    # To avoid infinite loops or getting stuck, we can limit iterations or use a deadline
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds safety margin
    
    while time.time() < deadline:
        # Find edge with max count
        max_cnt = 0
        max_edge = None
        for e, c in edge_cnt.items():
            if c > max_cnt:
                max_cnt = c
                max_edge = e
                
        if max_cnt == 0:
            break
            
        # Remove max_edge
        u, v = max_edge
        adj[u][v] = 0
        adj[v][u] = 0
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove(max_edge)
        del edge_cnt[max_edge]
        
        # Update counts
        # Removing (u,v) affects all triples B that contain v (or u)
        # Let's just recompute counts for simplicity, since n=20 is small
        # But recomputing is O(n^4) = 160k, which is fast.
        # Number of removals might be up to 100. 100 * 160k = 16M ops. Very fast.
        compute_counts()
        
        # Optional: Random restarts if we get stuck? 
        # If max_cnt == 0, we are done.
        
    # Convert edges set to list of lists
    result = [[u, v] for u, v in edges]
    return result

# To improve, we can try multiple random initializations or random removal orders
# But the greedy max-count removal is a strong heuristic.
# Let's add a few random restarts to escape local optima.
# We can modify construct to run this process a few times with slight variations.

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_count = -1
    
    # Try a few strategies
    # Strategy 1: Start with complete graph, greedy removal
    # Strategy 2: Start with random graph p=0.8, greedy addition/removal? No, just removal from dense.
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Initialize
        adj = [[1]*n for _ in range(n)]
        for i in range(n):
            adj[i][i] = 0
            
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                edges.add((u, v))
                
        triples = list(itertools.combinations(range(n), 3))
        
        adj_mask = [0]*n
        for u in range(n):
            for v in range(n):
                if adj[u][v]:
                    adj_mask[u] |= (1 << v)
                    
        edge_cnt = {e: 0 for e in edges}
        
        def compute_counts():
            for e in edge_cnt:
                edge_cnt[e] = 0
            for B in triples:
                b1, b2, b3 = B
                A_B = []
                for u in range(n):
                    if (adj_mask[u] >> b1) & 1 and (adj_mask[u] >> b2) & 1 and (adj_mask[u] >> b3) & 1:
                        A_B.append(u)
                k = len(A_B)
                if k >= 3:
                    add_val = (k-1)*(k-2)//2
                    for u in A_B:
                        for b in B:
                            e = (min(u,b), max(u,b))
                            if e in edge_cnt:
                                edge_cnt[e] += add_val
                                
        compute_counts()
        
        # Greedy removal
        while time.time() < deadline:
            max_cnt = 0
            max_edge = None
            for e, c in edge_cnt.items():
                if c > max_cnt:
                    max_cnt = c
                    max_edge = e
                    
            if max_cnt == 0:
                break
                
            u, v = max_edge
            adj[u][v] = 0
            adj[v][u] = 0
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            edges.remove(max_edge)
            del edge_cnt[max_edge]
            
            compute_counts()
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
            
        # If we have time, we can try a different randomization
        # For now, just repeat the same deterministic process? 
        # It will yield the same result. 
        # To vary, we can break ties randomly.
        
    if best_edges is None:
        return []
        
    return [[u, v] for u, v in best_edges]
