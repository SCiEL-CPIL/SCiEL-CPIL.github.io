import time
import random
import numpy as np

def _base32_colors():
    """Return the 32-vertex base coloring as a list of (i,j,color) in lex order."""
    # Re-implementing the Clebsch-based 32-vertex construction
    # Vertices 0-15 and 16-31
    # Inside blocks: Clebsch graph coloring
    # Between blocks: Color 3
    
    # GF(2^4) multiplication table for Clebsch
    def _build_mul_table():
        log = {}
        x = 1
        log[1] = 0
        for e in range(15):
            x2 = 0
            for i in range(4):
                if (x >> i) & 1:
                    x2 ^= (1 << (i + 1))
            if x2 & 0b10000:
                x2 ^= 0b10011
            x = x2
            log[x] = e + 1
        return log

    log_table = _build_mul_table()
    
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                # Same block: Clebsch
                # Map to 0-15
                ui = i % 16
                uj = j % 16
                # Clebsch graph edges are based on XOR
                # Color 0,1,2 based on log_table[ui ^ uj] % 3
                c = log_table[ui ^ uj] % 3
            else:
                c = 3
            colors.append((i, j, c))
    return colors

def _build_adj_matrices(colors, n):
    """Build adjacency matrices for fast triangle checking."""
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for i, j, c in colors:
        adj[c][i, j] = True
        adj[c][j, i] = True
    return adj

def _is_triangle_free_numpy(adj, n):
    """Check triangle-freeness using numpy matrix operations."""
    for c in range(4):
        A = adj[c]
        # Triangle exists if (A @ A) has any entry > 0 where A is also 1
        # common[i,j] = number of common neighbors
        common = A @ A.T
        if np.any(common * A):
            return False
    return True

def _extend_vertex(adj, n):
    """
    Try to extend the coloring to n+1 vertices.
    adj: list of 4 (n,n) boolean numpy arrays.
    Returns: (success, new_adj, assignment)
    """
    new_n = n + 1
    deadline = time.time() + 10 # 10s per vertex
    
    # Precompute neighbor counts for fast cost update
    # For each color c, and each vertex u, we need to know
    # how many neighbors v of u in color c are currently assigned to color c
    # This depends on the assignment of the new vertex.
    
    # We will use local search on the assignment of the new vertex.
    # Assignment: array of length n, assign[i] in {0,1,2,3}
    
    best_cost = float('inf')
    best_assign = None
    
    for trial in range(100):
        if time.time() > deadline:
            break
            
        assign = np.random.randint(0, 4, size=n)
        
        # Calculate initial cost
        # Cost = sum over c of edges in G_c with both endpoints in S_c
        # S_c = {i | assign[i] == c}
        
        # We can compute this by iterating
        # Or using numpy
        # For each c, mask = (assign == c)
        # cost_c = sum of A_c[mask, :][mask, :] / 2
        
        total_cost = 0
        for c in range(4):
            mask = (assign == c)
            if np.sum(mask) > 1:
                # Extract submatrix
                # A_c is boolean
                sub = adj[c][np.ix_(mask, mask)]
                # Count edges
                total_cost += np.sum(sub) // 2
        
        # Local search
        stuck = 0
        for it in range(5000):
            if time.time() > deadline:
                break
                
            if total_cost == 0:
                best_cost = 0
                best_assign = assign.copy()
                break
                
            # Pick random vertex to flip
            u = random.randint(0, n - 1)
            old_c = assign[u]
            
            # Find best new color
            min_new_cost = total_cost
            best_new_c = old_c
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                # Calculate cost change
                # Remove u from S_old:
                # Decrease by number of neighbors of u in S_old (in G_old)
                # Add u to S_new:
                # Increase by number of neighbors of u in S_new (in G_new)
                
                # Neighbors of u in G_old that are in S_old
                # adj[old_c][u, :] & (assign == old_c)
                # Exclude u itself (diagonal is 0)
                
                nbrs_old_in_set = np.sum(adj[old_c][u, :] & (assign == old_c))
                nbrs_new_in_set = np.sum(adj[new_c][u, :] & (assign == new_c))
                
                # Note: adj is symmetric.
                # When we move u from old to new:
                # The edges (u, v) where v in S_old are removed from cost_old.
                # The edges (u, v) where v in S_new are added to cost_new.
                
                delta = nbrs_new_in_set - nbrs_old_in_set
                new_cost = total_cost + delta
                
                if new_cost < min_new_cost:
                    min_new_cost = new_cost
                    best_new_c = new_c
            
            if best_new_c == old_c:
                stuck += 1
                if stuck > 100:
                    break
            else:
                stuck = 0
                # Update assignment
                assign[u] = best_new_c
                total_cost = min_new_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_assign = assign.copy()
            
        if best_cost == 0:
            break
            
    if best_cost == 0 and best_assign is not None:
        # Construct new adj matrices
        new_adj = [np.zeros((new_n, new_n), dtype=bool) for _ in range(4)]
        for c in range(4):
            new_adj[c][:n, :n] = adj[c]
            # Add new vertex edges
            # new vertex is index new_n - 1
            for i in range(n):
                if best_assign[i] == c:
                    new_adj[c][i, new_n - 1] = True
                    new_adj[c][new_n - 1, i] = True
        return True, new_adj, best_assign
    else:
        return False, None, None

def construct():
    deadline = time.time() + 110
    
    # Start with 32 vertices
    base_colors = _base32_colors()
    n = 32
    adj = _build_adj_matrices(base_colors, n)
    
    # Verify base
    if not _is_triangle_free_numpy(adj, n):
        return {'n': 0, 'colors': []}
        
    # Try to extend
    while n < 50 and time.time() < deadline - 10:
        success, new_adj, _ = _extend_vertex(adj, n)
        if success:
            n += 1
            adj = new_adj
        else:
            # If we fail, try a few more times with fresh random seeds?
            # The _extend_vertex already does 100 trials.
            # If it fails, it's likely stuck.
            # We could try to modify the existing graph, but that's complex.
            # For now, break.
            break
            
    # Reconstruct color list from adj matrices
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            for c in range(4):
                if adj[c][i, j]:
                    colors.append((i, j, c))
                    break
                    
    out_colors = [c for _, _, c in colors]
    
    return {'n': n, 'colors': out_colors}
