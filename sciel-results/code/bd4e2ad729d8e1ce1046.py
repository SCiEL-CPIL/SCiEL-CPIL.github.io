"""Improved program for R(K2,2,2) maximization via local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _has_k222(adj_mask, n):
    """
    Check if the graph defined by adj_mask (list of int bitmasks) contains a K2,2,2.
    K2,2,2 is a complete tripartite graph with parts of size 2.
    Equivalently, it is a subgraph of K6 where the complement is 3 disjoint edges.
    So we are looking for 3 disjoint non-edges (a,b), (c,d), (e,f) such that all other pairs among these 6 vertices are edges.
    """
    if n < 6:
        return False
    
    # Precompute non-edges? No, just iterate.
    # For each pair of non-edges (a,b) and (c,d) that are disjoint:
    # Find common neighbors of a,b,c,d.
    # If there is a non-edge (e,f) within those common neighbors, we have a K2,2,2.
    
    # To speed this up, we can iterate over pairs of vertices (a,b) that are NON-edges.
    # Let C_ab be the set of common neighbors of a and b.
    # For any other non-edge (c,d) disjoint from a,b, if C_ab contains c and d, 
    # we need to check if there is a non-edge (e,f) in C_ab \ {a,b,c,d} that is also a non-edge in the graph?
    # Actually, the condition is:
    # {a,b}, {c,d}, {e,f} are non-edges.
    # All cross edges exist.
    # So c,d must be in N(a) & N(b).
    # e,f must be in N(a) & N(b) & N(c) & N(d).
    # And e,f must be a non-edge.
    
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not (adj_mask[i] >> j) & 1:
                non_edges.append((i, j))
    
    # If there are fewer than 3 non-edges, no K2,2,2
    if len(non_edges) < 3:
        return False
        
    # Map non-edge to index? No, just iterate.
    # Optimization: If the graph is dense, non_edges is small. If sparse, non_edges is large.
    # K2,2,2 requires 3 non-edges.
    
    # Iterate over pairs of non-edges (ne1, ne2)
    len_ne = len(non_edges)
    for i1 in range(len_ne):
        a, b = non_edges[i1]
        # Common neighbors of a and b
        common_ab = adj_mask[a] & adj_mask[b]
        common_ab &= ~(1 << a) & ~(1 << b)
        
        for i2 in range(i1 + 1, len_ne):
            c, d = non_edges[i2]
            # Disjoint check
            if (c == a or c == b or d == a or d == b):
                continue
            
            # c and d must be in common_ab
            if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                continue
            
            # Now we have a, b, c, d.
            # e, f must be in common_ab AND common_cd (common neighbors of c and d)
            # common_cd = adj_mask[c] & adj_mask[d]
            # common_ef_candidates = common_ab & adj_mask[c] & adj_mask[d]
            # Remove a, b, c, d from candidates
            candidates = common_ab & adj_mask[c] & adj_mask[d]
            candidates &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
            
            # We need to find a non-edge (e, f) where e, f are in candidates.
            # Iterate through bits in candidates
            m = candidates
            found_candidate = False
            candidate_list = []
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                candidate_list.append(idx)
                m ^= lsb
            
            # Check if any pair in candidate_list is a non-edge
            len_cl = len(candidate_list)
            for k1 in range(len_cl):
                e = candidate_list[k1]
                for k2 in range(k1 + 1, len_cl):
                    f = candidate_list[k2]
                    if not ((adj_mask[e] >> f) & 1):
                        return True
                        
    return False

def _count_k222_fast(adj_mask, n):
    """
    Count K2,2,2 subgraphs. 
    For local search, we often just need to know if it's 0 or >0, 
    but for annealing, a count is better. 
    Given the time limit, we might just use a boolean check for validity 
    and a greedy approach for repair.
    """
    # This function is expensive. We will use _has_k222 for validity checks.
    # For local search cost, we will count the number of "bad" triples of non-edges.
    # Actually, let's just return 1 if found, 0 otherwise, to keep it fast.
    # The local search will be greedy: only accept if it reduces violations (0 or 1).
    if _has_k222(adj_mask, n):
        return 1
    return 0

def _adj_to_mask(adj, n):
    """Convert numpy boolean adjacency matrix to list of bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        row = adj[i]
        # Use numpy to get indices of True values? 
        # Or just loop. n <= 31, so loop is fine.
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _is_valid(adj, n):
    """Check that neither color class contains K2,2,2."""
    adj_mask_0 = _adj_to_mask(adj, n)
    if _has_k222(adj_mask_0, n):
        return False
    
    # Complement graph
    # Invert bits, clear diagonal
    adj_mask_1 = [0] * n
    all_ones = (1 << n) - 1
    for i in range(n):
        # ~adj_mask_0[i] & all_ones
        # Clear self-loop
        adj_mask_1[i] = (~adj_mask_0[i]) & all_ones & ~(1 << i)
        
    if _has_k222(adj_mask_1, n):
        return False
        
    return True

def _local_search(adj, n, deadline):
    """
    Local search to fix violations.
    We flip edges to minimize violations in both color classes.
    """
    adj = adj.copy()
    # We will maintain the masks for speed
    masks_0 = _adj_to_mask(adj, n)
    all_ones = (1 << n) - 1
    
    # Calculate initial violations
    def get_violations():
        v0 = 1 if _has_k222(masks_0, n) else 0
        masks_1 = [0] * n
        for i in range(n):
            masks_1[i] = (~masks_0[i]) & all_ones & ~(1 << i)
        v1 = 1 if _has_k222(masks_1, n) else 0
        return v0 + v1

    current_viol = get_violations()
    if current_viol == 0:
        return adj

    for _ in range(50000):
        if time.time() > deadline:
            break
            
        # Pick random edge
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        # Flip edge (i, j)
        # Update masks_0
        bit = 1 << j
        if masks_0[i] & bit:
            masks_0[i] &= ~bit
            masks_0[j] &= ~(1 << i)
        else:
            masks_0[i] |= bit
            masks_0[j] |= (1 << i)
            
        new_viol = get_violations()
        
        if new_viol <= current_viol:
            current_viol = new_viol
            if current_viol == 0:
                break
        else:
            # Revert
            if masks_0[i] & bit:
                masks_0[i] &= ~bit
                masks_0[j] &= ~(1 << i)
            else:
                masks_0[i] |= bit
                masks_0[j] |= (1 << i)

    return adj

def _paley_coloring(p):
    """Paley coloring of K_p where p is prime, p ≡ 1 mod 4."""
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add(x*x % p)
    colors = []
    for i in range(p):
        for j in range(i+1, p):
            diff = (j - i) % p
            colors.append(0 if diff in squares else 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 1
    best_colors = [0] # Fallback: K2 with one edge is valid? 
    # K2 has no K2,2,2. So n=2 is valid.
    # Let's use a known valid small graph.
    # Actually, any n < 6 is valid.
    # Let's start with n=5 or so to be safe, but Paley 17 is better.
    
    # Try to get a good start from Paley
    start_colors = None
    for p in [17, 29, 13, 5]:
        if p % 4 != 1:
            continue
        cols = _paley_coloring(p)
        if cols:
            # Verify validity
            n_test = p
            adj_test = np.zeros((n_test, n_test), dtype=bool)
            idx = 0
            for i in range(n_test):
                for j in range(i+1, n_test):
                    adj_test[i][j] = (cols[idx] == 0)
                    adj_test[j][i] = (cols[idx] == 0)
                    idx += 1
            if _is_valid(adj_test, n_test):
                start_colors = cols
                best_n = n_test
                best_colors = cols
                break
                
    if start_colors is None:
        # Fallback to small valid graph
        # n=5 random usually works, or just return n=1
        # Let's return a valid n=5
        n_test = 5
        adj_test = np.random.rand(n_test, n_test) > 0.5
        np.fill_diagonal(adj_test, False)
        # Make symmetric
        adj_test = (adj_test + adj_test.T) > 0
        if _is_valid(adj_test, n_test):
            best_n = n_test
            best_colors = []
            for i in range(n_test):
                for j in range(i+1, n_test):
                    best_colors.append(1 if adj_test[i][j] else 0)
        else:
            # Absolute fallback
            best_n = 1
            best_colors = []

    # Try to extend best_n
    current_n = best_n
    current_adj = np.zeros((current_n, current_n), dtype=bool)
    idx = 0
    for i in range(current_n):
        for j in range(i+1, current_n):
            current_adj[i][j] = (best_colors[idx] == 0)
            current_adj[j][i] = (best_colors[idx] == 0)
            idx += 1

    for target_n in range(current_n + 1, 31):
        if time.time() > deadline:
            break
            
        n = target_n
        # Initialize new adj
        adj_new = np.zeros((n, n), dtype=bool)
        
        # Copy old
        adj_new[:current_n, :current_n] = current_adj
        
        # Initialize new edges
        # Heuristic: random or based on symmetry
        for i in range(current_n, n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_new[i][j] = True
                    adj_new[j][i] = True
        for i in range(current_n):
            for j in range(current_n, n):
                if random.random() < 0.5:
                    adj_new[i][j] = True
                    adj_new[j][i] = True
        
        # Local search
        adj_fixed = _local_search(adj_new, n, deadline)
        
        if _is_valid(adj_fixed, n):
            current_n = n
            current_adj = adj_fixed
            # Update best
            best_n = n
            best_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    best_colors.append(0 if adj_fixed[i][j] else 1)
            
            # If we reached 29, we can stop early as it's the record? 
            # No, we want to maximize. 30 is the goal.
            # But 29 is the known record. 30 is unproven.
            # Continue until deadline.
            
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
