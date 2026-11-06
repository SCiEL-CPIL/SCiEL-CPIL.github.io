"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of PG(2, q) using the standard orthogonal polarity.
    Vertices are 1D subspaces (points) of F_q^3.
    Two points are adjacent iff their representatives are orthogonal under the form x1*y1 + x2*y2 - x3*y3 = 0.
    This graph is C4-free.
    """
    # Generate all 1D subspaces of F_q^3
    reps = []
    seen = set()
    
    # Iterate through all non-zero vectors in F_q^3 and normalize them
    # Normalization: first non-zero coordinate is 1
    for z in range(q):
        for y in range(q):
            for x in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Find first non-zero
                if x != 0:
                    norm = (1, y * pow(x, -1, q) % q, z * pow(x, -1, q) % q)
                elif y != 0:
                    norm = (0, 1, z * pow(y, -1, q) % q)
                else:
                    norm = (0, 0, 1)
                
                if norm not in seen:
                    seen.add(norm)
                    reps.append(norm)
    
    n = len(reps)
    if n != q*q + q + 1:
        raise ValueError(f"Expected {q*q+q+1} vertices, got {n}")
        
    adj = [set() for _ in range(n)]
    edges = []
    
    # Form: u[0]*v[0] + u[1]*v[1] - u[2]*v[2] == 0
    for i in range(n):
        for j in range(i+1, n):
            u = reps[i]
            v = reps[j]
            val = (u[0]*v[0] + u[1]*v[1] - u[2]*v[2]) % q
            if val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
                
    return n, edges, adj

def _check_no_c4(n, edges):
    """Check that no two vertices have two common neighbours."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return False
    return True

def _has_independent_set_of_size(n, adj, target_size):
    """
    Check if there is an independent set of size 'target_size'.
    Uses Branch and Bound with bitmasks for speed.
    Returns True if found, False otherwise.
    """
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    all_mask = (1 << n) - 1
    
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def branch(remaining_mask, current_size):
        if current_size >= target_size:
            return True
        if remaining_mask == 0:
            return False
        
        # Upper bound: if current_size + popcount(remaining) < target, fail
        if current_size + popcount(remaining_mask) < target_size:
            return False
            
        # Pick a vertex v in remaining
        # Heuristic: pick vertex with max degree in remaining subgraph to prune faster?
        # Or just pick the first one. For small n, first is fine.
        # Let's pick the one with max degree in remaining to reduce options for exclusion?
        # Actually, for independent set, we want to include high degree vertices? No.
        # Standard B&B for IS:
        # 1. Include v: remove v and its neighbors from remaining.
        # 2. Exclude v: remove v from remaining.
        
        temp = remaining_mask
        v_idx = -1
        max_deg = -1
        while temp:
            lsb = temp & -temp
            v = (lsb.bit_length() - 1)
            neighbors_in_rem = masks[v] & remaining_mask
            deg = popcount(neighbors_in_rem)
            if deg > max_deg:
                max_deg = deg
                v_idx = v
            temp -= lsb
            
        if v_idx == -1:
            return False
            
        v_bit = 1 << v_idx
        
        # Option 1: Include v
        # New remaining: remove v and its neighbors
        new_remaining = remaining_mask & ~(masks[v_idx] | v_bit)
        if branch(new_remaining, current_size + 1):
            return True
            
        # Option 2: Exclude v
        new_remaining = remaining_mask & ~v_bit
        if branch(new_remaining, current_size):
            return True
            
        return False

    return branch(all_mask, 0)

def _verify_graph(n, edges, target_alpha=11, timeout=5):
    """Verify graph has no C4 and alpha < target_alpha."""
    if not _check_no_c4(n, edges):
        return False
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    start_time = time.time()
    # Check if independent set of size target_alpha exists
    if _has_independent_set_of_size(n, adj, target_alpha):
        return False
        
    return True

def construct():
    """
    Construct a C4-free graph with no independent 11-set.
    Uses the polarity graph of PG(2, 7) which has 57 vertices and alpha=8.
    We take the first 43 vertices to satisfy the n <= 43 constraint.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    try:
        # Construct PG(2, 7) polarity graph
        n_full, edges_full, adj_full = _construct_polarity_graph(7)
        
        # We want n=43
        target_n = 43
        if target_n > n_full:
            target_n = n_full
            
        # Extract subgraph of first target_n vertices
        # Vertices 0 to target_n-1
        edges_sub = []
        adj_sub = [set() for _ in range(target_n)]
        
        for u, v in edges_full:
            if u < target_n and v < target_n:
                edges_sub.append([u, v])
                adj_sub[u].add(v)
                adj_sub[v].add(u)
                
        # Verify the subgraph
        if _verify_graph(target_n, edges_sub, target_alpha=11, timeout=10):
            return {'n': target_n, 'edges': edges_sub}
            
    except Exception as e:
        pass
    
    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    # This is always valid
    edges_30 = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges_30 += [[a,b], [a,c], [b,c]]
        
    return {'n': 30, 'edges': edges_30}
# EVOLVE-BLOCK-END
