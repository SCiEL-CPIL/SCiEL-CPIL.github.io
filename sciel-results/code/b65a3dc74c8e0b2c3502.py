import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try to find valid colorings for increasing n
    # Start with known constructions and build up
    
    best_n = 1
    best_colors = None
    
    # Known small constructions
    # n=2: trivial
    if 2 <= 29:
        best_n = 2
        best_colors = [0]
    
    # n=3: trivial (no C4 possible)
    if 3 <= 29:
        best_n = 3
        best_colors = [0, 0, 0]
    
    # n=4: need to avoid monochromatic C4
    # K4 has 6 edges. A C4 in K4 would be 4 vertices with 4 edges forming a cycle.
    # Actually in K4, any 4 vertices form K4 which contains C4s.
    # We need no monochromatic C4.
    # Color edges of K4: use 2 colors, e.g., 0-1:0, 0-2:1, 0-3:0, 1-2:0, 1-3:1, 2-3:1
    # Check: color 0 edges: (0,1),(0,3),(1,2) - no C4
    # color 1 edges: (0,2),(1,3),(2,3) - no C4
    if 4 <= 29:
        best_n = 4
        best_colors = [0, 1, 0, 0, 1, 1]
    
    # For larger n, use backtracking with constraint propagation
    # Try n from 5 up to 29, but focus on getting good n
    
    # Efficient violation counting using adjacency matrices
    def check_valid(n, colors):
        """Check if coloring has no monochromatic C4"""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        # Build adjacency matrices for each color
        # A_c[i,j] = 1 if edge (i,j) has color c
        matrices = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                matrices[c][i, j] = 1
                matrices[c][j, i] = 1
                idx += 1
        
        # For each color, check if any pair of vertices has >= 2 common neighbors
        # This means (A_c^2)[i,j] >= 2 for some i != j
        for c in range(5):
            A = matrices[c]
            A2 = A @ A
            # Check off-diagonal entries
            np.fill_diagonal(A2, 0)
            if np.any(A2 >= 2):
                return False
        
        return True
    
    # Backtracking search for n vertices
    def backtrack_search(n, time_limit):
        """Find a valid 5-coloring of K_n using backtracking with constraint propagation"""
        num_edges = n * (n - 1) // 2
        colors = [-1] * num_edges
        
        # Precompute edge indices
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                edge_idx[(j, i)] = idx
                idx += 1
        
        # For constraint propagation, track for each color and pair of vertices,
        # how many common neighbors they have
        # common_count[c][u][v] = number of w such that (u,w) and (v,w) both have color c
        
        def get_common_count(c, u, v, colors):
            """Count common neighbors of u and v in color c"""
            count = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                cuw = colors[edge_idx[(u, w)]]
                cvw = colors[edge_idx[(v, w)]]
                if cuw == c and cvw == c:
                    count += 1
            return count
        
        # Recursive backtracking
        def backtrack(pos, violations):
            if time.time() > time_limit:
                return False
            
            if pos == num_edges:
                return violations == 0
            
            i = pos // (n - 1 - pos // (n - 1))  # This is wrong, let me fix
            # Actually, let's use a simpler mapping
            # pos-th edge in lexicographic order
            # We need to find which (i,j) this is
            # Number of edges with first vertex < i: i*(n-1) - i*(i-1)//2
            # Let me precompute this
            
            return False
        
        # Simpler approach: use random initialization + local search with better heuristics
        return None
    
    # Better approach: use the Lazebnik-Woldar construction or similar algebraic construction
    # For now, try random + local search with efficient violation counting
    
    def local_search(n, max_iterations, time_limit):
        """Local search to find valid coloring"""
        num_edges = n * (n - 1) // 2
        colors = np.random.randint(0, 5, size=num_edges)
        
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                edge_idx[(j, i)] = idx
                idx += 1
        
        def count_violations():
            """Count total number of monochromatic C4 violations"""
            total = 0
            for c in range(5):
                # Build adjacency matrix
                A = np.zeros((n, n), dtype=np.int8)
                for (u, v), idx in edge_idx.items():
                    if u < v and colors[idx] == c:
                        A[u, v] = 1
                        A[v, u] = 1
                
                A2 = A @ A
                np.fill_diagonal(A2, 0)
                # Count entries >= 2
                total += np.sum(A2 >= 2)
            return total
        
        # Local search
        best_colors = colors.copy()
        best_violations = count_violations()
        
        for iteration in range(max_iterations):
            if time.time() > time_limit:
                break
            
            if best_violations == 0:
                break
            
            # Pick a random edge to change
            edge_pos = random.randint(0, num_edges - 1)
            old_color = colors[edge_pos]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[edge_pos] = new_color
            new_violations = count_violations()
            
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors.copy()
            else:
                # Accept with some probability to escape local minima
                if random.random() < 0.1:
                    best_colors = colors.copy()
                else:
                    colors[edge_pos] = old_color
        
        if best_violations == 0:
            return best_colors.tolist()
        return None
    
    # Try increasing n from 5 to 29
    # Allocate time budget
    time_per_n = 8.0  # seconds per n value
    current_time = start_time
    
    for n in range(5, 30):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts
        found = None
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            result = local_search(n, 5000, deadline - 5)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_n = n
            best_colors = found
            # If we found n=26 or higher, we can stop or continue trying for higher
            if n >= 26:
                # Try a bit longer for higher n
                continue
        else:
            # If we can't find valid coloring for this n, try next
            # But keep best_n as is
            pass
    
    # Final check
    if best_colors is not None and check_valid(best_n, best_colors):
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback to smallest valid
    return {'n': best_n, 'colors': best_colors if best_colors else []}
