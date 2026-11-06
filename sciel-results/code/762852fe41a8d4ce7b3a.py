"""Combined program: deterministic C4-free construction + fast verification."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _popcount(x):
    """Count set bits in integer x."""
    return bin(x).count('1')

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if _popcount(common) >= 2:
                return True
    return False

def _has_independent_set(n, adj, target):
    """
    Check if there exists an independent set of size >= target in C4-free graph.
    Uses branch-and-bound with tight upper bounds.
    Returns True if independent set of size >= target exists.
    """
    if target <= 0:
        return True
    if n < target:
        return False
    
    # Build complement adjacency: comp_adj[v] = set of vertices NOT adjacent to v (excluding v itself)
    # In an independent set, all vertices must be pairwise non-adjacent in original graph,
    # i.e., pairwise adjacent in complement.
    # So we're looking for a clique of size >= target in the complement graph.
    
    all_mask = (1 << n) - 1
    
    # For each vertex, compute its "non-neighbors" mask (vertices it could be paired with in independent set)
    non_neighbors = [0] * n
    for v in range(n):
        non_neighbors[v] = (all_mask ^ (1 << v)) & ~adj[v]
    
    # Branch and bound for finding clique in complement (independent set in original)
    # Use coloring-based upper bound (Tomita algorithm idea)
    
    import functools
    
    @functools.lru_cache(maxsize=None)
    def max_clique_size(candidates_mask):
        """Find max clique size in complement subgraph induced by candidates_mask."""
        if not candidates_mask:
            return 0
        
        count = _popcount(candidates_mask)
        if count == 0:
            return 0
        if count <= 1:
            return count
        
        # Upper bound: count of candidates
        if count < 2:
            return count
        
        # Pick vertex with max degree in candidate subgraph (greedy heuristic)
        best_v = -1
        best_deg = -1
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(non_neighbors[v] & candidates_mask)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v < 0:
            return 0
        
        v = best_v
        
        # Option 1: include v - recurse on non_neighbors[v] & candidates_mask without v
        mask_without_v = candidates_mask ^ (1 << v)
        new_candidates = non_neighbors[v] & mask_without_v
        include_val = 1 + max_clique_size(new_candidates)
        
        # Option 2: exclude v
        exclude_val = max_clique_size(mask_without_v)
        
        return max(include_val, exclude_val)
    
    result = max_clique_size(all_mask)
    return result >= target

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _has_independent_set(n, adj, 11):
        return False
    
    return True

def _estimate_alpha(n, adj):
    """Greedy estimate of independence number for ranking."""
    remaining = (1 << n) - 1
    greedy_alpha = 0
    while remaining:
        best_v = -1
        best_deg = float('inf')
        temp = remaining
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(adj[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        greedy_alpha += 1
        remaining &= ~(1 << best_v) & ~adj[best_v]
    return greedy_alpha

def _construct_39_vertex_graph():
    """
    Construct a C4-free graph on 39 vertices with independence number < 11.
    Uses 13 triangles with carefully chosen inter-triangle edges.
    """
    n = 39
    deadline = time.time() + 115  # Leave margin for verification
    
    best_result = None
    best_alpha = float('inf')
    
    for seed in range(2000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Start with 13 triangles
        adj = [0] * n
        edges = []
        for t in range(13):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges.append([a, b])
            edges.append([a, c])
            edges.append([b, c])
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
        
        # Generate all possible inter-triangle edges
        inter_edges = []
        for i in range(13):
            for j in range(i+1, 13):
                for ui in range(3*i, 3*i+3):
                    for vj in range(3*j, 3*j+3):
                        inter_edges.append((ui, vj))
        
        random.shuffle(inter_edges)
        
        # Try to add edges greedily, checking C4 constraint
        added_count = 0
        for ui, vj in inter_edges:
            if time.time() > deadline:
                break
            
            # Check if adding (ui, vj) creates a C4
            # C4 check: for any u' in N(ui) and v' in N(vj), if u'!=vj and v'!=ui 
            # and u' is adjacent to v', then C4: ui-vj-v'-u'-ui
            # Actually: C4 is ui - vj - x - y - ui where x adjacent to vj, y adjacent to ui, x!=ui, y!=vj
            # So: common neighbors of ui and vj in the current graph must be <= 1
            
            common = adj[ui] & adj[vj]
            if _popcount(common) >= 2:
                continue
            
            # Add the edge
            edges.append([ui, vj])
            adj[ui] |= (1 << vj)
            adj[vj] |= (1 << ui)
            added_count += 1
        
        # Verify C4-freeness (should be guaranteed by construction but double-check)
        if _has_c4_bitmask(n, adj):
            continue
        
        # Quick alpha estimate for ranking
        est_alpha = _estimate_alpha(n, adj)
        
        if est_alpha < best_alpha:
            best_alpha = est_alpha
            best_result = {'n': n, 'edges': edges.copy()}
            
            # If we found alpha < 11, verify exactly and return
            if est_alpha < 11:
                if not _has_independent_set(n, adj, 11):
                    return best_result
                # If exact check says alpha >= 11, don't use this result
                best_result = None
                best_alpha = float('inf')
    
    if best_result:
        return best_result
    
    # Fallback: return a smaller valid graph
    # 10 disjoint triangles: 30 vertices, alpha = 10
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    # Try to construct a 39-vertex graph
    result = _construct_39_vertex_graph()
    
    # Verify the result
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
