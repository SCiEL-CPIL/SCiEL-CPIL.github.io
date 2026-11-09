import time
import random

def construct():
    """
    Construct a 5-coloring of K_n with no monochromatic C4.
    Uses a greedy cost-minimization approach to add vertices,
    preferring colors that create fewer new at-risk pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with a known valid n=11 coloring (cyclic construction)
    n_init = 11
    best_n = n_init
    best_colors = []
    cm = [[-1] * n_init for _ in range(n_init)]
    idx = 0
    for i in range(n_init):
        for j in range(i + 1, n_init):
            c = min(j - i, n_init - j + i) - 1
            cm[i][j] = c
            cm[j][i] = c
            best_colors.append(c)
            idx += 1
    
    n = n_init
    
    def flatten_cm(cm, n):
        """Flatten upper triangle of cm to a list."""
        result = []
        for i in range(n):
            for j in range(i + 1, n):
                result.append(cm[i][j])
        return result
    
    def try_add_vertex(cm, n, deadline):
        """
        Try to add a new vertex (n) with backtracking.
        Returns (new_cm, new_n) if successful, else (None, None).
        Uses cost-minimization: prefer colors that create fewer new at-risk pairs.
        """
        new_n = n + 1
        new_cm = [[-1] * new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(n):
                new_cm[i][j] = cm[i][j]
        
        # Build adjacency sets for each color
        adj_sets = [[set() for _ in range(new_n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                adj_sets[c][i].add(j)
                adj_sets[c][j].add(i)
        
        # Track assigned colors for the new vertex's edges
        assigned = [-1] * n  # assigned[u] = color of edge (new_v, u)
        
        # Track which vertices have been assigned to new_v in each color
        new_v_neighbors = [set() for _ in range(5)]
        
        # Precompute: for each color c and pair (u, v), whether u and v have a common neighbor in color c
        # We'll compute this on the fly using adj_sets
        
        # Order of vertices to assign: try to minimize conflicts
        # Order vertices by how constrained they are (higher degree in any color first)
        def vertex_constraint(u):
            max_deg = 0
            for c in range(5):
                max_deg = max(max_deg, len(adj_sets[c][u]))
            return max_deg
        
        order = sorted(range(n), key=vertex_constraint, reverse=True)
        
        def get_cost(u, c):
            """
            Compute the cost of assigning color c to edge (new_v, u).
            Cost = number of new at-risk pairs created, plus a small penalty
            for high existing degree in color c.
            """
            cost = 0
            for v in new_v_neighbors[c]:
                # Check if u and v have a common neighbor in color c in the existing graph
                # If they do, adding new_v would create 2 common neighbors -> C4
                common = adj_sets[c][u] & adj_sets[c][v]
                if len(common) >= 1:
                    return float('inf')  # Invalid
                else:
                    cost += 1  # New at-risk pair
            # Small penalty for high existing degree in this color
            deg_u_c = len(adj_sets[c][u])
            cost += 0.05 * deg_u_c
            return cost
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                # Success: write assigned colors to new_cm
                for i in range(n):
                    c = assigned[i]
                    new_cm[new_n - 1][i] = c
                    new_cm[i][new_n - 1] = c
                return True
            
            u = order[idx]
            
            # Try each color, ordered by cost (lowest first)
            colors_with_cost = []
            for c in range(5):
                cost = get_cost(u, c)
                if cost < float('inf'):
                    colors_with_cost.append((cost, c))
            
            # Sort by cost (ascending), then by color
            colors_with_cost.sort()
            
            for cost, c in colors_with_cost:
                assigned[u] = c
                new_v_neighbors[c].add(u)
                
                if backtrack(idx + 1):
                    return True
                
                assigned[u] = -1
                new_v_neighbors[c].remove(u)
            
            return False
        
        if backtrack(0):
            return new_cm, new_n
        return None, None
    
    # Try to add vertices iteratively
    attempts = 0
    max_attempts_per_n = 100
    
    while n < 29 and time.time() < deadline:
        success = False
        attempts = 0
        while attempts < max_attempts_per_n and not success and time.time() < deadline:
            new_cm, new_n = try_add_vertex(cm, n, deadline)
            if new_cm is not None:
                cm = new_cm
                n = new_n
                success = True
                if n > best_n:
                    best_n = n
                    best_colors = flatten_cm(cm, n)
            else:
                attempts += 1
                # If too many failures, break to try a different approach or stop
                if attempts >= max_attempts_per_n:
                    break
        
        if not success:
            # Can't add another vertex with this approach
            break
    
    return {'n': best_n, 'colors': best_colors}
