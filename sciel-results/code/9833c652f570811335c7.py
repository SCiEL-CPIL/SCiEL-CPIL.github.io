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
            
    def improve(adj):
        # Try to add edges
        improved = True
        while improved:
            if time.time() > deadline:
                return False
            improved = False
            # Shuffle order to avoid bias?
            edges_to_check = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
            random.shuffle(edges_to_check)
            
            for u, v in edges_to_check:
                if time.time() > deadline:
                    return False
                # Try adding edge (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not is_valid(adj):
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    improved = True
                    # If we added one, we can break and restart the loop to re-shuffle
                    # Or continue? Continuing might be faster but less random.
                    # Let's break to re-evaluate all possibilities in new state
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
            # Improve
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
