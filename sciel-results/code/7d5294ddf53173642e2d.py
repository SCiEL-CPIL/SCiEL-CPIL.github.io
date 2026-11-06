import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    def build_paley(p):
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return p, orientation

    def verify(n, adj):
        """
        Verify if every 4-set has a common dominator.
        adj: numpy boolean array (n, n), adj[i,j]=True if i->j
        """
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Check if there is any u such that u->a, u->b, u->c, u->d
            # This is equivalent to checking if the bitwise AND of the columns a,b,c,d in adj is non-zero
            # adj[:, a] is the column of vertices that point TO a? No.
            # adj[i, j] is True if i -> j.
            # So adj[:, a] is the vector of vertices i such that i -> a.
            # We want u such that u -> a, u -> b, u -> c, u -> d.
            # So we need adj[u, a] & adj[u, b] & adj[u, c] & adj[u, d] to be True for some u.
            # This is the element-wise AND of the columns a, b, c, d.
            # If any element is True, we are good.
            
            # Note: u can be any vertex, including those in the quad?
            # "Every set of 4 vertices must have a vertex with arcs to all four."
            # If u is in the quad, say u=a, then we need a->a (False). So u cannot be in the quad.
            # The check adj[:, a] & ... will naturally handle this because adj[a, a] is False.
            
            dominators = adj[:, a] & adj[:, b] & adj[:, c] & adj[:, d]
            if not np.any(dominators):
                return False
        return True

    def get_critical_vertices(n, adj):
        """
        Returns a boolean array 'critical' where critical[v] is True if v is the unique dominator
        of some 4-set.
        """
        critical = np.zeros(n, dtype=bool)
        
        # To speed up, we can iterate over quads.
        # If a quad has exactly 1 dominator, that dominator is critical.
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            dominators = adj[:, a] & adj[:, b] & adj[:, c] & adj[:, d]
            count = np.sum(dominators)
            if count == 1:
                idx = np.where(dominators)[0][0]
                critical[idx] = True
            
            # Early exit if all are critical?
            if np.all(critical):
                break
                
        return critical

    # Initial construction: Paley 67
    p_init = 67
    _, orientation = build_paley(p_init)
    
    # Build initial adjacency matrix
    adj = np.zeros((p_init, p_init), dtype=bool)
    idx = 0
    for i in range(p_init):
        for j in range(i+1, p_init):
            if orientation[idx] == 0:
                adj[i, j] = True
            else:
                adj[j, i] = True
            idx += 1
            
    # Verify initial
    if not verify(p_init, adj):
        # Should not happen for Paley 67
        return {'n': p_init, 'orientation': orientation}

    active = list(range(p_init))
    current_n = p_init
    best_n = p_init
    best_orientation = orientation
    best_adj = adj.copy()

    # Optimization loop
    while elapsed() < DEADLINE:
        if current_n < 5:
            break
            
        # Get critical vertices in the current subgraph defined by 'active'
        # We need to map the active vertices to a contiguous 0..m-1 space for efficient NumPy ops?
        # Or just use the original indices.
        # Using original indices in a boolean mask is fine, but slicing might be slower.
        # Let's extract the submatrix for active vertices.
        
        m = len(active)
        if m == current_n:
             # First iteration, we already have adj
             sub_adj = adj
        else:
             sub_adj = adj[np.ix_(active, active)]
        
        critical = get_critical_vertices(m, sub_adj)
        
        # Find non-critical vertices
        non_critical_indices = np.where(~critical)[0]
        
        if len(non_critical_indices) == 0:
            break
            
        # Remove one non-critical vertex
        # Heuristic: Remove the one with the smallest out-degree within the subgraph?
        # Or just the first one.
        # Let's pick the first one for simplicity, or maybe randomize to avoid local minima?
        # Given the time, deterministic is safer.
        
        # Pick the first non-critical index in the local space
        local_idx_to_remove = non_critical_indices[0]
        global_idx_to_remove = active[local_idx_to_remove]
        
        # Remove from active list
        active.remove(global_idx_to_remove)
        current_n -= 1
        
        # Update best if improved
        if current_n < best_n:
            best_n = current_n
            # Reconstruct orientation for the current active set
            sub_adj_current = adj[np.ix_(active, active)]
            final_orientation = []
            for i in range(current_n):
                for j in range(i+1, current_n):
                    if sub_adj_current[i, j]:
                        final_orientation.append(0)
                    else:
                        final_orientation.append(1)
            best_orientation = final_orientation
            best_adj = sub_adj_current.copy()
            
    # Final verification of the best found
    # We need to ensure the best_orientation corresponds to best_n
    # The loop above updates best_orientation only when current_n < best_n.
    # If we break early, best_orientation is the last valid one.
    
    # One final check on the best result
    # Reconstruct adj from best_orientation to be sure
    if best_n > 0:
        adj_check = np.zeros((best_n, best_n), dtype=bool)
        idx_c = 0
        for i in range(best_n):
            for j in range(i+1, best_n):
                if best_orientation[idx_c] == 0:
                    adj_check[i, j] = True
                else:
                    adj_check[j, i] = True
                idx_c += 1
        
        if verify(best_n, adj_check):
            return {'n': best_n, 'orientation': best_orientation}
    
    # Fallback to initial if something went wrong
    return {'n': p_init, 'orientation': orientation}
