"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _has_k222_fast(adj, n):
    """
    Check if graph with adjacency matrix adj (numpy 2D array of 0/1) on n vertices
    contains K2,2,2.
    Uses the property: K2,2,2 exists iff there is a pair (i,j) such that
    the subgraph induced by N(i) ∩ N(j) contains a K2,2.
    K2,2 exists in a graph iff there are two vertices with at least 2 common neighbors.
    """
    # adj is n x n numpy array of int8
    for i in range(n):
        for j in range(i + 1, n):
            # Common neighbors of i and j
            S = adj[i] & adj[j]
            # Get indices of vertices in S
            verts = np.nonzero(S)[0]
            if len(verts) < 4:
                continue
            
            # Check if subgraph induced by verts contains K2,2
            # K2,2 exists if there are u, v in verts such that 
            # |N(u) ∩ N(v) ∩ S| >= 2
            
            # Optimization: iterate over pairs in verts
            # To speed up, we can precompute adj[u] & S for all u in verts
            # But verts size is small (max 28).
            
            # We can vectorize the check for all pairs (u,v) in verts
            # sub_adj = adj[np.ix_(verts, verts)]
            # But we need intersection with S.
            # Since u, v are in S, adj[u] & S is the neighbors of u within S.
            
            # Let's just loop, it's small
            m = len(verts)
            for a_idx in range(m):
                u = verts[a_idx]
                # Neighbors of u in S
                N_u = adj[u] & S
                # We need to find v in verts (v != u) such that N_u & adj[v] & S has >= 2 bits
                # We can check all v > u
                for b_idx in range(a_idx + 1, m):
                    v = verts[b_idx]
                    # Common neighbors of u and v in S
                    common = N_u & adj[v] & S
                    if np.count_nonzero(common) >= 2:
                        return True
    return False

def _build_adj_np(colors, n):
    """Build adjacency matrix as numpy 2D array of 0/1 from color list."""
    adj = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj[i, j] = 1
                adj[j, i] = 1
            idx += 1
    return adj

def _is_valid_np(colors, n):
    """Check if coloring is valid (no K2,2,2 in either color class) using numpy."""
    if len(colors) != n * (n - 1) // 2:
        return False
    
    adj1 = _build_adj_np(colors, n)
    if _has_k222_fast(adj1, n):
        return False
    
    # Complement graph
    adj0 = (1 - adj1).astype(np.int8)
    np.fill_diagonal(adj0, 0)
    
    if _has_k222_fast(adj0, n):
        return False
    return True

def _paley_colors(n, p):
    """
    Generate colors for the first n vertices of the Paley graph on p vertices.
    Color 1 if (j-i) mod p is a quadratic residue.
    Color 0 otherwise.
    """
    if p < n:
        return None
    squares = set((x * x) % p for x in range(1, p))
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            colors.append(1 if diff in squares else 0)
    return colors

def _count_violations(colors, n):
    """
    Count the number of monochromatic K2,2,2 subgraphs.
    Returns (count_color1, count_color0).
    """
    adj1 = _build_adj_np(colors, n)
    adj0 = (1 - adj1).astype(np.int8)
    np.fill_diagonal(adj0, 0)
    
    # We just need to check existence for SA, but counting helps with delta.
    # For simplicity and speed, let's just check existence and return 0 or 1+
    # Actually, for SA, we need a cost function.
    # Let's return the number of "bad" pairs (i,j) that lead to a K2,2 in S.
    # This is a proxy for the number of K2,2,2s.
    
    def count_proxy(adj):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                S = adj[i] & adj[j]
                verts = np.nonzero(S)[0]
                if len(verts) < 4:
                    continue
                m = len(verts)
                found = False
                for a_idx in range(m):
                    u = verts[a_idx]
                    N_u = adj[u] & S
                    for b_idx in range(a_idx + 1, m):
                        v = verts[b_idx]
                        common = N_u & adj[v] & S
                        if np.count_nonzero(common) >= 2:
                            count += 1
                            found = True
                            break
                    if found:
                        break
        return count

    c1 = count_proxy(adj1)
    c0 = count_proxy(adj0)
    return c1 + c0

def _sa_extend(base_colors, base_n, target_n, deadline):
    """
    Use Simulated Annealing to extend a valid coloring of base_n vertices to target_n.
    """
    if base_n >= target_n:
        return base_colors, True
    
    # Initialize with base colors + random edges for new vertices
    cur_colors = list(base_colors)
    for i in range(base_n, target_n):
        for j in range(i + 1, target_n):
            cur_colors.append(np.random.randint(2))
        # Edges from new vertex i to old vertices 0..i-1
        # Wait, the order is lexicographic (i,j) with i<j.
        # If we add vertex i, the new edges are (0,i), (1,i), ..., (i-1,i).
        # These are NOT contiguous at the end of the list if we just append.
        # The list is ordered by (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # So adding a new vertex n-1 appends (0,n-1), (1,n-1), ..., (n-2,n-1) at the end.
        # This is correct for sequential addition.
        
    # But if we start with n=29 and want n=30, we append 29 edges.
    # The indices for these edges are:
    # idx for (i, 29) is sum_{k=0}^{i-1} (n-k-1) + i ?
    # Let's just rebuild the adj matrix for cost calculation.
    
    current_cost = _count_violations(cur_colors, target_n)
    best_colors = list(cur_colors)
    best_cost = current_cost
    
    T = 1.0
    cooling_rate = 0.99
    steps = 5000
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Pick a random edge to flip
        edge_idx = np.random.randint(len(cur_colors))
        cur_colors[edge_idx] = 1 - cur_colors[edge_idx]
        
        new_cost = _count_violations(cur_colors, target_n)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = list(cur_colors)
                if best_cost == 0:
                    return best_colors, True
        else:
            # Revert
            cur_colors[edge_idx] = 1 - cur_colors[edge_idx]
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return best_colors, (best_cost == 0)

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # 1. Try Paley 29
    c29 = _paley_colors(29, 29)
    if c29:
        if _is_valid_np(c29, 29):
            # Try to extend to 30
            if time.time() < deadline - 20:
                ext_colors, success = _sa_extend(c29, 29, 30, deadline - 5)
                if success and _is_valid_np(ext_colors, 30):
                    return {'n': 30, 'colors': ext_colors}
            return {'n': 29, 'colors': c29}
            
    # 2. Try other constructions if 29 failed (unlikely)
    # Try Paley 37 subset
    c37_30 = _paley_colors(30, 37)
    if c37_30:
        if _is_valid_np(c37_30, 30):
            return {'n': 30, 'colors': c37_30}
            
    # Fallback
    if c29:
        return {'n': 29, 'colors': c29}
        
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
