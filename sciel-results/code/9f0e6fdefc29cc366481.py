import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def check_c6_in_color(adj_matrix, n, color):
        """Check if the subgraph of edges with given color contains a C6."""
        # Build adjacency for this color class
        A = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i][j] == color:
                    A[i][j] = 1
                    A[j][i] = 1
        # Count C6: trace(A^6)/12
        if n < 6:
            return False
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A4 @ A2
        return np.trace(A6) > 0

    def has_monochromatic_c6(adj_matrix, n):
        """Check if any color class contains a C6."""
        for c in range(4):
            if check_c6_in_color(adj_matrix, n, c):
                return True
        return False

    def backtrack_search(n, adj_matrix, next_edge_idx, edge_list, remaining_time_fn):
        """
        Backtracking search to find a valid 4-coloring of K_n with no monochromatic C6.
        adj_matrix: n x n matrix where adj_matrix[i][j] = -1 if uncolored, else 0-3
        next_edge_idx: index into edge_list of next edge to color
        edge_list: list of (i, j) pairs in lexicographic order
        Returns True if a valid coloring is found, modifying adj_matrix in place.
        """
        if time.time() > remaining_time_fn():
            return False
        
        if next_edge_idx == len(edge_list):
            # All edges colored, final check
            return not has_monochromatic_c6(adj_matrix, n)
        
        i, j = edge_list[next_edge_idx]
        
        # Try each color
        for c in range(4):
            adj_matrix[i][j] = c
            adj_matrix[j][i] = c
            
            # Forward check: see if any color class now has a C6
            # Only check colors that could be affected (all 4, but we can be smarter)
            valid = True
            for cc in range(4):
                if check_c6_in_color(adj_matrix, n, cc):
                    valid = False
                    break
            
            if valid:
                if backtrack_search(n, adj_matrix, next_edge_idx + 1, edge_list, remaining_time_fn):
                    return True
            
            adj_matrix[i][j] = -1
            adj_matrix[j][i] = -1
        
        return False

    def randomized_backtrack(n, adj_matrix, next_edge_idx, edge_list, remaining_time_fn, max_nodes=50000):
        """
        Randomized backtracking with restarts.
        Returns the adj_matrix if successful, else None.
        """
        # Shuffle colors to break symmetry
        color_order = list(range(4))
        random.shuffle(color_order)
        
        def bt(idx):
            nonlocal max_nodes
            if time.time() > remaining_time_fn():
                return False
            if idx >= max_nodes:
                return False
            
            if next_edge_idx == len(edge_list):
                return not has_monochromatic_c6(adj_matrix, n)
            
            i, j = edge_list[next_edge_idx]
            
            for c in color_order:
                adj_matrix[i][j] = c
                adj_matrix[j][i] = c
                
                valid = True
                for cc in range(4):
                    if check_c6_in_color(adj_matrix, n, cc):
                        valid = False
                        break
                
                if valid:
                    if bt(idx + 1):
                        return True
                
                adj_matrix[i][j] = -1
                adj_matrix[j][i] = -1
            
            return False
        
        return bt(0)

    def extract_colors(adj_matrix, n):
        """Extract color list in lexicographic order."""
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(adj_matrix[i][j])
        return colors

    # Try n from 17 down to 10
    best_result = None
    
    for n in range(17, 9, -1):
        if time.time() > deadline:
            break
        
        edge_list = [(i, j) for i in range(n) for j in range(i+1, n)]
        found = False
        
        # Try multiple randomized restarts
        for restart in range(200):
            if time.time() > deadline:
                break
            
            adj_matrix = np.full((n, n), -1, dtype=np.int8)
            
            # Use a node limit per restart to encourage diversity
            success = randomized_backtrack(n, adj_matrix, 0, edge_list, lambda: deadline, max_nodes=10000)
            
            if success:
                # Verify
                if not has_monochromatic_c6(adj_matrix, n):
                    colors = extract_colors(adj_matrix, n)
                    best_result = {'n': n, 'colors': colors}
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: K9 cyclic construction
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
