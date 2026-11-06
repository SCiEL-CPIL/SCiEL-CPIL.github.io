"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _has_k222_np(adj, n):
    """
    Check if graph with adjacency matrix adj (numpy 2D array of 0/1) on n vertices
    contains K2,2,2 using vectorized NumPy operations.
    """
    # Pre-compute pairings of indices 0..5
    pairings = [
        ((0,1),(2,3),(4,5)), ((0,1),(2,4),(3,5)), ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)), ((0,2),(1,4),(3,5)), ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)), ((0,3),(1,4),(2,5)), ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)), ((0,4),(1,3),(2,5)), ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)), ((0,5),(1,3),(2,4)), ((0,5),(1,4),(2,3)),
    ]
    
    # For each 6-subset, check all 15 pairings
    for verts in itertools.combinations(range(n), 6):
        v = np.array(verts)
        # Vectorize over all 15 pairings at once
        # For each pairing, we need:
        #   part1: v[p1[0]], v[p1[1]]
        #   part2: v[p2[0]], v[p2[1]]
        #   part3: v[p3[0]], v[p3[1]]
        # Check: all edges between part1-part2, part1-part3, part2-part3
        
        # Build arrays for vectorized check
        p1a = np.array([p[0][0] for p in pairings])
        p1b = np.array([p[0][1] for p in pairings])
        p2a = np.array([p[1][0] for p in pairings])
        p2b = np.array([p[1][1] for p in pairings])
        p3a = np.array([p[2][0] for p in pairings])
        p3b = np.array([p[2][1] for p in pairings])
        
        # Get vertex indices for each pairing
        a_arr = v[p1a]  # shape (15,)
        b_arr = v[p1b]
        c_arr = v[p2a]
        d_arr = v[p2b]
        e_arr = v[p3a]
        f_arr = v[p3b]
        
        # Check edges between parts
        # Part1-Part2: adj[a,c], adj[a,d], adj[b,c], adj[b,d]
        ab = adj[a_arr, c_arr] & adj[a_arr, d_arr] & adj[b_arr, c_arr] & adj[b_arr, d_arr]
        # Part1-Part3: adj[a,e], adj[a,f], adj[b,e], adj[b,f]
        ac = adj[a_arr, e_arr] & adj[a_arr, f_arr] & adj[b_arr, e_arr] & adj[b_arr, f_arr]
        # Part2-Part3: adj[c,e], adj[c,f], adj[d,e], adj[d,f]
        bc = adj[c_arr, e_arr] & adj[c_arr, f_arr] & adj[d_arr, e_arr] & adj[d_arr, f_arr]
        
        if np.any(ab & ac & bc):
            return True
    return False

def _build_adj_np(colors, n):
    """Build adjacency matrix as numpy 2D array of 0/1 from color list."""
    adj = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == 1:
                adj[i, j] = 1
                adj[j, i] = 1
            idx += 1
    return adj

def _is_valid_np(colors, n):
    """Check if coloring is valid (no K2,2,2 in either color class) using numpy."""
    if len(colors) != n*(n-1)//2:
        return False
    
    adj1 = _build_adj_np(colors, n)
    if _has_k222_np(adj1, n):
        return False
    
    # Complement graph
    adj0 = (1 - adj1).astype(np.int8)
    np.fill_diagonal(adj0, 0)
    
    if _has_k222_np(adj0, n):
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
    squares = set((x*x) % p for x in range(1, p))
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % p
            colors.append(1 if diff in squares else 0)
    return colors

def _find_k222_violations(adj, n):
    """
    Find all K2,2,2 violations in the graph. Returns list of 6-tuples (sorted vertex sets).
    """
    pairings = [
        ((0,1),(2,3),(4,5)), ((0,1),(2,4),(3,5)), ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)), ((0,2),(1,4),(3,5)), ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)), ((0,3),(1,4),(2,5)), ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)), ((0,4),(1,3),(2,5)), ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)), ((0,5),(1,3),(2,4)), ((0,5),(1,4),(2,3)),
    ]
    
    p1a = np.array([p[0][0] for p in pairings])
    p1b = np.array([p[0][1] for p in pairings])
    p2a = np.array([p[1][0] for p in pairings])
    p2b = np.array([p[1][1] for p in pairings])
    p3a = np.array([p[2][0] for p in pairings])
    p3b = np.array([p[2][1] for p in pairings])
    
    violations = []
    for verts in itertools.combinations(range(n), 6):
        v = np.array(verts)
        a_arr = v[p1a]
        b_arr = v[p1b]
        c_arr = v[p2a]
        d_arr = v[p2b]
        e_arr = v[p3a]
        f_arr = v[p3b]
        
        ab = adj[a_arr, c_arr] & adj[a_arr, d_arr] & adj[b_arr, c_arr] & adj[b_arr, d_arr]
        ac = adj[a_arr, e_arr] & adj[a_arr, f_arr] & adj[b_arr, e_arr] & adj[b_arr, f_arr]
        bc = adj[c_arr, e_arr] & adj[c_arr, f_arr] & adj[d_arr, e_arr] & adj[d_arr, f_arr]
        
        if np.any(ab & ac & bc):
            violations.append(tuple(sorted(verts)))
    
    return violations

def _local_search_extend(colors, n, target_n, deadline):
    """
    Try to extend a valid coloring of n vertices to target_n vertices
    using local search. Returns (success, extended_colors) or (False, None).
    """
    if n >= target_n:
        return True, colors
    
    # Start with the existing coloring
    cur_colors = list(colors)
    cur_n = n
    
    # We'll add vertices one at a time
    while cur_n < target_n:
        if time.time() > deadline:
            return False, None
        
        new_v = cur_n
        best_assignment = None
        
        # Try multiple random assignments
        for attempt in range(200):
            if time.time() > deadline:
                break
            assignment = [np.random.randint(2) for _ in range(cur_n)]
            temp_n = cur_n + 1
            temp_colors = cur_colors + assignment
            temp_adj = _build_adj_np(temp_colors, temp_n)
            
            if not _has_k222_np(temp_adj, temp_n):
                temp_adj0 = (1 - temp_adj).astype(np.int8)
                np.fill_diagonal(temp_adj0, 0)
                if not _has_k222_np(temp_adj0, temp_n):
                    best_assignment = assignment
                    break
        
        if best_assignment is not None:
            cur_colors.extend(best_assignment)
            cur_n += 1
        else:
            # Try flipping edges to make room for new vertex
            success = False
            for flip_attempt in range(300):
                if time.time() > deadline:
                    break
                edge_idx = np.random.randint(len(cur_colors))
                old_val = cur_colors[edge_idx]
                cur_colors[edge_idx] = 1 - old_val
                
                # Try to add the new vertex with random assignment
                assignment = [np.random.randint(2) for _ in range(cur_n)]
                temp_colors = cur_colors + assignment
                temp_n = cur_n + 1
                temp_adj = _build_adj_np(temp_colors, temp_n)
                
                if not _has_k222_np(temp_adj, temp_n):
                    temp_adj0 = (1 - temp_adj).astype(np.int8)
                    np.fill_diagonal(temp_adj0, 0)
                    if not _has_k222_np(temp_adj0, temp_n):
                        cur_colors.extend(assignment)
                        cur_n += 1
                        success = True
                        break
                else:
                    cur_colors[edge_idx] = old_val
            
            if not success:
                return False, None
    
    return True, cur_colors

def construct():
    deadline = time.time() + 130  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Try known good constructions
    candidates = []
    
    # Generate Paley 29
    c29 = _paley_colors(29, 29)
    if c29:
        candidates.append((29, c29))
        
    # Generate Paley 37 and try to find a valid subset of size 30
    c37_30 = _paley_colors(30, 37)
    if c37_30:
        candidates.append((30, c37_30))
    
    # Sort by n descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for n, colors in candidates:
        if time.time() > deadline:
            break
        try:
            if _is_valid_np(colors, n):
                best_n = n
                best_colors = colors
                if n == 30:
                    return {'n': n, 'colors': colors}
        except Exception:
            continue
            
    # If 30 failed, try local search to extend 29 to 30 with multiple attempts
    if best_n == 29 and c29:
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            try:
                # Use different random seeds for diversity
                np.random.seed(attempt * 12345)
                success, ext_colors = _local_search_extend(c29, 29, 30, deadline - 10)
                if success and ext_colors:
                    if _is_valid_np(ext_colors, 30):
                        return {'n': 30, 'colors': ext_colors}
            except Exception:
                pass
    
    if best_colors is None:
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
