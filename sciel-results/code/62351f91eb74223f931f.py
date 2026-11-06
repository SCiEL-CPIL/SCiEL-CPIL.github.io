import time
import sys

# EVOLVE-BLOCK-START
def construct():
    # We will try to find a coloring for n=17, 16, ... down to 10.
    # If none found within time, return a known valid small coloring (e.g., n=9 from seed).
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s for overhead
    
    # Try decreasing n
    for n in range(17, 9, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Calculate remaining time for this n
        # If we just started, give it more time. If we've spent a lot, give it less.
        # Simple heuristic: if n==17, allow up to 100s. If n<17, allow 5s each.
        if n == 17:
            limit_for_n = 100.0
        else:
            limit_for_n = 5.0
            
        sol = solve_c6(n, limit_for_n)
        if sol is not None:
            return {'n': n, 'colors': sol}
            
    # Fallback: Return the seed solution (n=9)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_c6(n, time_limit):
    """
    Backtracking solver for 4-coloring K_n without monochromatic C6.
    Returns list of colors for edges (i,j) i<j in lex order, or None if timeout/fail.
    """
    start_time = time.time()
    
    # adj[i][j] = color, -1 if unassigned
    adj = [[-1] * n for _ in range(n)]
    
    # List of edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Symmetry breaking: First edge (0,1) must be 0
    # Second edge (0,2) can be 0, 1, or 2? 
    # Let's just use standard backtracking with pruning.
    
    def is_valid_assignment(i, j, c):
        """
        Check if assigning color c to edge (i,j) creates a monochromatic C6.
        Assumes all other edges in the potential cycle are already assigned.
        """
        # A C6 containing (i,j) has form i-v1-v2-v3-v4-j-i
        # Edges: (i,v1), (v1,v2), (v2,v3), (v3,v4), (v4,j) must be color c
        # Note: Vertices must be distinct from i, j and each other.
        
        # Optimization: Iterate v1 neighbors of i with color c
        # Iterate v4 neighbors of j with color c
        # Check if there is a path of length 2 between v1 and v4 with color c
        
        # Get neighbors of i with color c
        # Note: adj is symmetric. We only store i<j in adj? 
        # Let's keep adj symmetric for easy lookup.
        
        neigh_i = []
        for v in range(n):
            if v == i or v == j:
                continue
            # Check if edge (i,v) is assigned and has color c
            if i < v:
                if adj[i][v] == c:
                    neigh_i.append(v)
            else:
                if adj[v][i] == c:
                    neigh_i.append(v)
                    
        neigh_j = []
        for v in range(n):
            if v == i or v == j:
                continue
            if j > v:
                if adj[v][j] == c:
                    neigh_j.append(v)
            else:
                if adj[j][v] == c:
                    neigh_j.append(v)
        
        # For each v1 in neigh_i and v4 in neigh_j, check if path v1-v2-v4 exists with color c
        # v1, v4 must be distinct from each other and from i,j (already handled by skip i,j)
        # v1 and v4 must be distinct? Yes, in a simple cycle.
        
        for v1 in neigh_i:
            for v4 in neigh_j:
                if v1 == v4:
                    continue
                # Check for v2 such that (v1,v2) is c and (v2,v4) is c
                # v2 must be distinct from i, j, v1, v4
                for v2 in range(n):
                    if v2 == i or v2 == j or v2 == v1 or v2 == v4:
                        continue
                    # Check edge (v1, v2)
                    if v1 < v2:
                        e1 = adj[v1][v2]
                    else:
                        e1 = adj[v2][v1]
                    if e1 != c:
                        continue
                    # Check edge (v2, v4)
                    if v2 < v4:
                        e2 = adj[v2][v4]
                    else:
                        e2 = adj[v4][v2]
                    if e2 != c:
                        continue
                    # Found a cycle!
                    return False
        
        return True

    def backtrack(idx):
        if time.time() - start_time > time_limit:
            return False
        
        if idx == len(edges):
            return True
            
        i, j = edges[idx]
        
        # Try colors 0, 1, 2, 3
        # Symmetry breaking for the first few edges?
        # For edge (0,1), only try 0.
        # For edge (0,2), try 0, 1.
        # For edge (0,3), try 0, 1, 2.
        # This reduces the initial branching significantly.
        
        if i == 0 and j == 1:
            colors_to_try = [0]
        elif i == 0 and j == 2:
            colors_to_try = [0, 1]
        elif i == 0 and j == 3:
            colors_to_try = [0, 1, 2]
        else:
            colors_to_try = [0, 1, 2, 3]
            
        for c in colors_to_try:
            if is_valid_assignment(i, j, c):
                # Assign
                adj[i][j] = c
                adj[j][i] = c
                
                if backtrack(idx + 1):
                    return True
                    
                # Unassign
                adj[i][j] = -1
                adj[j][i] = -1
                
        return False

    if backtrack(0):
        # Extract colors in lex order
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(adj[i][j])
        return colors
    else:
        return None
# EVOLVE-BLOCK-END
