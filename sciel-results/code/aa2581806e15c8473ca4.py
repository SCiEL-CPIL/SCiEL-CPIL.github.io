import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    N = 16
    subsets = list(itertools.combinations(range(N), 8))
    
    # Precompute for each subset, the list of (u, v) pairs in the subset
    # And the bipartitions
    # To speed up, we can precompute the "local" bipartitions
    
    bipartitions_local = []
    for A in itertools.combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        bipartitions_local.append((A, B))
        
    def is_matching_local(missing_indices):
        # missing_indices is a list of (i, j) in 0..3
        seen_a = [0]*4
        seen_b = [0]*4
        for i, j in missing_indices:
            if seen_a[i] or seen_b[j]:
                return False
            seen_a[i] = 1
            seen_b[j] = 1
        return True

    def subset_has_q3(adj, subset):
        # adj is 16x16
        # subset is tuple of 8 vertices
        # Map to local 0..7
        # We need to check if any bipartition of the 8 vertices works
        
        # Optimization: If total edges in induced subgraph < 12, return False
        # Sum of adj[subset[i]][subset[j]] for i<j
        # This is O(28).
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[subset[i]][subset[j]]:
                    edge_count += 1
        if edge_count < 12:
            return False
            
        for A, B in bipartitions_local:
            A_global = [subset[i] for i in A]
            B_global = [subset[j] for j in B]
            
            missing = []
            for i in range(4):
                for j in range(4):
                    if not adj[A_global[i]][B_global[j]]:
                        missing.append((i, j))
                        
            if len(missing) > 4:
                continue
                
            if is_matching_local(missing):
                return True
            # If missing is empty, it's K4,4, which contains Q3
            # is_matching_local([]) returns True (vacuously)
            # Wait, if missing is empty, len(missing)=0, which is <= 4.
            # is_matching_local([]) is True.
            # So it returns True. Correct.
            
        return False

    def find_q3(adj):
        # Returns a subset that contains Q3, or None
        # Shuffle subsets for randomization
        shuffled = subsets[:]
        random.shuffle(shuffled)
        for subset in shuffled:
            if subset_has_q3(adj, subset):
                return subset
        return None

    def try_add_edge(adj, u, v):
        # Check if adding (u,v) creates a Q3
        # Only check subsets containing u and v
        # Precompute list of subsets containing u and v?
        # Or iterate all and filter?
        # Filtering 12870 subsets is fast.
        
        for subset in subsets:
            if u in subset and v in subset:
                # Temporarily add
                adj[u][v] = 1
                adj[v][u] = 1
                if subset_has_q3(adj, subset):
                    adj[u][v] = 0
                    adj[v][u] = 0
                    return False
                adj[u][v] = 0
                adj[v][u] = 0
        return True

    best_edges = 0
    best_adj = None
    
    while time.time() < deadline:
        # Random init
        p = random.uniform(0.6, 0.9)
        adj = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    
        # Repair phase
        while time.time() < deadline:
            q3_subset = find_q3(adj)
            if q3_subset is None:
                break
            # Remove a random edge from the q3_subset
            edges_in_subset = []
            for i in range(8):
                for j in range(i+1, 8):
                    u = q3_subset[i]
                    v = q3_subset[j]
                    if adj[u][v]:
                        edges_in_subset.append((u, v))
            if edges_in_subset:
                u, v = random.choice(edges_in_subset)
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                break # Should not happen
                
        # Improvement phase
        # Try to add all missing edges
        missing_edges = [(i, j) for i in range(N) for j in range(i+1, N) if not adj[i][j]]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if try_add_edge(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                
        current_edges = sum(sum(row) for row in adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj]
            
        if time.time() > deadline - 10:
            break

    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges

# EVOLVE-BLOCK-END
