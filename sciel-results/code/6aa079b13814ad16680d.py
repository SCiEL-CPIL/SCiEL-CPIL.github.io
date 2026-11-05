"""Construct a C5-free graph with no independent set of size 9, maximizing n."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _has_c5_fast(adj_matrix, n):
    """Check if graph has a 5-cycle using numpy for speed.
    adj_matrix is a (n,n) boolean numpy array.
    A 5-cycle exists iff there exist distinct vertices v0,v1,v2,v3,v4
    such that all edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v0) exist.
    We use the approach: for each pair of non-adjacent vertices (a,c) with dist=2,
    check if there's a path of length 3 from a to c through two other vertices.
    Actually, let's use a more direct but vectorized approach.
    
    For a 5-cycle v0-v1-v2-v3-v4-v0:
    - v0 and v2 are non-adjacent (otherwise we'd have a triangle within the 5-cycle)
    - v1 is a common neighbor of v0 and v2
    - v0 and v3 are non-adjacent  
    - v4 is a common neighbor of v0 and v3... 
    
    Let me use a cleaner approach:
    For each ordered pair (u,v) where u<v and u,v are non-adjacent:
      Let N(u) ∩ N(v) = common neighbors
      For each pair (a,b) in common neighbors with a<b:
        Check if there exists w such that w is adjacent to both a and b, 
        and w is not u or v, and the path u-a-w-b-v-u forms a 5-cycle.
        Actually this is getting complicated.
    
    Let me just use the direct enumeration but with numpy for the inner checks.
    """
    # adj_matrix[i,j] is True if edge (i,j) exists
    # For 5-cycle: v0-v1-v2-v3-v4-v0
    # We can fix v0 and v1 (adjacent), then v2 must be adjacent to v1 but not v0
    # v3 must be adjacent to v2 but not v0 or v1
    # v4 must be adjacent to v3 and v0, but not v1 or v2
    
    for v0 in range(n):
        # v1 is a neighbor of v0
        for v1 in np.where(adj_matrix[v0])[0]:
            if v1 <= v0:
                continue
            # v2 is a neighbor of v1, not v0
            for v2 in np.where(adj_matrix[v1])[0]:
                if v2 <= v1 or v2 == v0:
                    continue
                # v3 is a neighbor of v2, not v0 or v1
                for v3 in np.where(adj_matrix[v2])[0]:
                    if v3 <= v2 or v3 == v0 or v3 == v1:
                        continue
                    # v4 must be adjacent to v3 and v0, not v1 or v2
                    # Check if there's any such v4
                    candidates = np.where(adj_matrix[v3] & adj_matrix[v0])[0]
                    for v4 in candidates:
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        return True
    return False

def _has_c5_sets(adj_sets, n):
    """Check C5 using adjacency sets - optimized version."""
    for v0 in range(n):
        n0 = adj_sets[v0]
        for v1 in n0:
            if v1 <= v0:
                continue
            n1 = adj_sets[v1] - {v0}
            for v2 in n1:
                if v2 <= v1 or v2 == v0:
                    continue
                n2 = adj_sets[v2] - {v0, v1}
                for v3 in n2:
                    if v3 <= v2 or v3 == v0 or v3 == v1:
                        continue
                    n3 = adj_sets[v3] - {v0, v1, v2}
                    # Check if any vertex in n3 is also in n0 (adjacent to v0)
                    if n3 & n0:
                        return True
    return False

def _has_indep_9(adj_sets, n):
    """Check if there's an independent set of size 9.
    Use backtracking with pruning."""
    # Independent set: no two vertices are adjacent
    # Use branch and bound
    
    best = [0]
    
    def backtrack(remaining, current_size, chosen):
        if current_size >= 9:
            return True
        if len(remaining) < 9 - current_size:
            return False
        if not remaining:
            return False
        
        # Choose vertex with minimum degree in remaining (greedy heuristic)
        # Actually, for independent set, we want vertices with few neighbors in remaining
        v = min(remaining, key=lambda x: len(adj_sets[x] & remaining))
        
        # Option 1: include v
        new_remaining = remaining - adj_sets[v] - {v}
        if backtrack(new_remaining, current_size + 1, chosen | {v}):
            return True
        
        # Option 2: exclude v
        if backtrack(remaining - {v}, current_size, chosen):
            return True
        
        return False
    
    return backtrack(set(range(n)), 0, set())

def _independence_number(adj_sets, n):
    """Compute independence number using branch and bound."""
    best = [0]
    
    def backtrack(remaining, current_size):
        if current_size > best[0]:
            best[0] = current_size
        if not remaining:
            return
        if len(remaining) + current_size <= best[0]:
            return
        
        # Choose vertex with minimum degree in remaining
        v = min(remaining, key=lambda x: len(adj_sets[x] & remaining))
        
        # Include v
        new_remaining = remaining - adj_sets[v] - {v}
        backtrack(new_remaining, current_size + 1)
        
        # Exclude v
        backtrack(remaining - {v}, current_size)
    
    backtrack(set(range(n)), 0)
    return best[0]

def _make_adj_sets(edges, n):
    """Create adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _make_adj_matrix(edges, n):
    """Create numpy adjacency matrix from edge list."""
    mat = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        mat[u, v] = True
        mat[v, u] = True
    return mat

def construct():
    """
    Returns a simple graph on n vertices (0..n-1) that is:
    - C5-free (no 5-cycle as a subgraph)
    - has no independent set of size 9
    
    Strategy: Start with 8 disjoint K4s (n=32, alpha=8),
    then try to add vertices and edges to reach n=33,34,35.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    def base_k4_edges(n_blocks=8):
        edges = []
        for b in range(n_blocks):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return edges
    
    best_result = {'n': 32, 'edges': base_k4_edges(8)}
    
    for target_n in [35, 34, 33]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            random.seed(12345 + attempt * 7919 + target_n)
            
            # Start with 8 K4s for first 32 vertices
            edges = base_k4_edges(8)
            adj = _make_adj_sets(edges, target_n)
            
            # Add new vertices 32..target_n-1
            for nv in range(32, target_n):
                # Connect to 1-2 vertices in each K4 block
                for b in range(8):
                    base = b * 4
                    num_conn = random.randint(1, 2)
                    targets = random.sample(range(base, base + 4), num_conn)
                    for t in targets:
                        if t not in adj[nv]:
                            adj[nv].add(t)
                            adj[t].add(nv)
                            edges.append([nv, t])
            
            # Check C5 after initial construction
            if _has_c5_sets(adj, target_n):
                continue
            
            # Check if we already have no independent set of size 9
            if not _has_indep_9(adj, target_n):
                best_result = {'n': target_n, 'edges': edges}
                break
            
            # Try to add more edges to reduce independence number
            # Generate candidate cross-edges
            cross_edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if j not in adj[i]:
                        block_i = i // 4 if i < 32 else 8
                        block_j = j // 4 if j < 32 else 8
                        if block_i != block_j:
                            cross_edges.append((i, j))
            
            random.shuffle(cross_edges)
            
            added = 0
            for u, v in cross_edges:
                if time.time() > deadline - 3:
                    break
                if v in adj[u]:
                    continue
                
                # Try adding edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
                # Check C5
                if _has_c5_sets(adj, target_n):
                    # Remove this edge
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edges.pop()
                else:
                    added += 1
                    # Periodically check independence
                    if added % 3 == 0:
                        if not _has_indep_9(adj, target_n):
                            break
            
            # Final validation
            if not _has_c5_sets(adj, target_n) and not _has_indep_9(adj, target_n):
                best_result = {'n': target_n, 'edges': edges}
                break
    
    return best_result
# EVOLVE-BLOCK-END
