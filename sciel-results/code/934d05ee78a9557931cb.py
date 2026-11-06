import time
import random

def construct():
    """
    Constructs a K3,3-free graph on 20 vertices (10 in U, 10 in V) 
    maximizing the number of edges.
    Returns a list of edges [u, v].
    """
    n_u = 10
    n_v = 10
    deadline = time.time() + 110  # 110s safety margin
    
    best_edges = 0
    best_matrix = None
    best_edge_list = None
    
    # Helper to check if adding edge (u, v) creates a K3,3
    # Assumes matrix M is current state (before adding)
    # u in [0, n_u), v in [0, n_v)
    def can_add(M, u, v):
        if M[u][v]:
            return True # Already exists
        # Get neighbors of u (cols)
        N_u = [j for j in range(n_v) if M[u][j]]
        
        # Candidates for u1, u2 are rows connected to v
        S_v = [i for i in range(n_u) if i != u and M[i][v]]
        
        if len(S_v) < 2:
            return True # Can't form K3,3 if less than 2 other rows connected to v
        
        # Check all pairs in S_v
        # Precompute neighbor lists for rows in S_v?
        # Or just use bitmasks for speed? 
        # With n=10, lists are fine.
        
        for i in range(len(S_v)):
            u1 = S_v[i]
            N_u1 = [j for j in range(n_v) if M[u1][j]]
            for k in range(i+1, len(S_v)):
                u2 = S_v[k]
                N_u2 = [j for j in range(n_v) if M[u2][j]]
                
                # Find common neighbors of u, u1, u2
                # We need 2 common neighbors OTHER than v
                # Note: v is in N_u1 and N_u2.
                # v is NOT in N_u (since we are checking if we CAN add it, M[u][v] is 0).
                # So common neighbors of u, u1, u2 are subset of N_u.
                
                # Efficient intersection
                set_u = set(N_u)
                set_u1 = set(N_u1)
                set_u2 = set(N_u2)
                
                common = set_u.intersection(set_u1).intersection(set_u2)
                
                # We need at least 2 elements in common
                # Wait, do we need to exclude v?
                # v is in N_u1 and N_u2.
                # Is v in N_u? No, M[u][v] is 0.
                # So v is NOT in common.
                # So we just need len(common) >= 2.
                
                if len(common) >= 2:
                    return False
        return True

    # Helper to get edge list from matrix
    def mat_to_edges(M):
        edges = []
        for u in range(n_u):
            for v in range(n_v):
                if M[u][v]:
                    edges.append([u, v])
        # Map to 0..19
        # U: 0..9, V: 10..19
        mapped = []
        for u, v in edges:
            mapped.append([u, v + n_u])
        return mapped

    # Run multiple restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Initialize matrix
        # Strategy: Start with random density, or empty, or structured
        # Let's try random density 0.5
        M = [[random.randint(0, 1) for _ in range(n_v)] for _ in range(n_u)]
        
        # Greedy removal of K3,3s
        # We can just try to remove edges that are part of K3,3
        # Or simpler: iterate through all edges, if adding them back would be bad, keep them removed?
        # Better: Start with all 1s, remove edges that cause K3,3?
        # Or start with random, then "repair" by removing edges.
        
        # Let's do a "Clean" phase:
        # Iterate through all possible edges (u,v).
        # If M[u][v] is 1, check if removing it is necessary?
        # Actually, let's just use the `can_add` logic in reverse or a global check.
        
        # Global check for K3,3:
        # For each triple of rows, check if they have >= 3 common neighbors.
        def is_k33_free(M):
            for r1 in range(n_u):
                for r2 in range(r1+1, n_u):
                    for r3 in range(r2+1, n_u):
                        # Common neighbors
                        common = []
                        for c in range(n_v):
                            if M[r1][c] and M[r2][c] and M[r3][c]:
                                common.append(c)
                        if len(common) >= 3:
                            return False
            return True

        # Repair loop: While not free, remove a random edge from a violation
        # Find a violation
        def find_violation(M):
            for r1 in range(n_u):
                for r2 in range(r1+1, n_u):
                    for r3 in range(r2+1, n_u):
                        common = []
                        for c in range(n_v):
                            if M[r1][c] and M[r2][c] and M[r3][c]:
                                common.append(c)
                        if len(common) >= 3:
                            # Return one edge to remove, e.g., (r1, common[0])
                            return (r1, common[0])
            return None

        while not is_k33_free(M):
            if time.time() > deadline:
                break
            viol = find_violation(M)
            if viol:
                u, v = viol
                M[u][v] = 0
            else:
                break

        # Now M is K3,3-free. Try to add edges.
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias?
            order = [(u, v) for u in range(n_u) for v in range(n_v)]
            random.shuffle(order)
            for u, v in order:
                if not M[u][v]:
                    if can_add(M, u, v):
                        M[u][v] = 1
                        improved = True
                        if time.time() > deadline:
                            break
            if time.time() > deadline:
                break

        # Local Search: Try to swap edges to increase density
        # This is harder. Let's just rely on multiple restarts with different initial densities.
        
        current_edges = sum(sum(row) for row in M)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_matrix = [row[:] for row in M]
            best_edge_list = mat_to_edges(M)
            
        # If we found a very good solution, we can stop early?
        # Theoretical max for 10x10 K3,3-free is around 40-50?
        # If best_edges >= 50, break.
        if best_edges >= 50:
            break

    # Fallback to seed if no better found or timeout
    if best_edge_list is None or best_edges < 36:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edge_list
