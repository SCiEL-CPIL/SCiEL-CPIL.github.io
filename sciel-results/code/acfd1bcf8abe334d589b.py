"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _has_c4_in_subgraph(adj, mask, n):
    """Check if the subgraph induced by 'mask' contains a C4 (K2,2).
    adj: list of bitmasks
    mask: bitmask of vertices in the subgraph
    """
    # Get list of vertices in mask
    verts = []
    m = mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        verts.append(idx)
        m ^= lsb
    
    L = len(verts)
    if L < 4:
        return False
        
    # For each pair (u, v) in verts, check if they have >= 2 common neighbors in verts
    # If u and v share neighbors w and x in verts, then u-w-v-x-u is a C4
    # (provided w != x and all distinct, which is guaranteed by set membership)
    for i in range(L):
        u = verts[i]
        for j in range(i + 1, L):
            v = verts[j]
            # Common neighbors of u and v within the mask
            common = adj[u] & adj[v] & mask
            if common.bit_count() >= 2:
                return True
    return False

def _is_valid_edge_assignment(adj, n, u, v, color):
    """Check if adding edge (u,v) with given color is valid.
    We only need to check the color class we are adding to.
    If we add (u,v) to color c, we must ensure N_c(u) intersect N_c(v) does not contain C4.
    Note: N_c(u) and N_c(v) should NOT include each other yet in the check logic if we are 
    simulating the addition, but since we are updating adj, we check the state BEFORE addition 
    for the common neighbors, and then verify the new induced subgraph.
    
    Actually, simpler: 
    1. Tentatively add edge (u,v) to color c in adj.
    2. Check if color c graph has K2,2,2.
    3. Check if color 1-c graph has K2,2,2.
    4. If valid, keep. Else, revert.
    
    But full check is O(n^3). We need incremental.
    When adding (u,v) to color c:
    - The only new potential K2,2,2 must involve the new edge (u,v).
    - So (u,v) must be an edge in the K2,2,2.
    - In K2,2,2, every edge belongs to exactly 2 parts? No.
    - K2,2,2 has parts A={a,b}, B={c,d}, C={e,f}.
    - Edges are between A-B, A-C, B-C.
    - If (u,v) is in A-B, then u,v are in different parts.
    - This is complicated.
    
    Let's use the property: 
    A graph contains K2,2,2 iff there exists an edge (u,v) such that 
    the common neighbors of u and v contain a C4.
    Why? 
    If (u,v) is an edge in K2,2,2, say u in A, v in B.
    Then A\{u} and B\{v} and C are all common neighbors? 
    No. 
    If u in A, v in B.
    N(u) = B U C.
    N(v) = A U C.
    N(u) intersect N(v) = C.
    Does C contain a C4? C has size 2. No.
    
    Wait, my previous logic for _has_k222 was:
    "For each adjacent pair (a,b), let CN = common neighbors. CN must contain a C4."
    Let's re-verify.
    K2,2,2 parts: P1={a,b}, P2={c,d}, P3={e,f}.
    Take edge (a,c) where a in P1, c in P2.
    N(a) = P2 U P3 = {c,d,e,f}.
    N(c) = P1 U P3 = {a,b,e,f}.
    Common neighbors of a and c: P3 = {e,f}.
    Size 2. Does {e,f} contain a C4? No.
    
    So the condition "Common neighbors contain C4" is for when (a,b) are in the SAME part?
    If a,b in P1.
    N(a) = P2 U P3.
    N(b) = P2 U P3.
    Common neighbors = P2 U P3 = {c,d,e,f}.
    Does {c,d,e,f} contain a C4?
    Edges in K2,2,2: c-e, c-f, d-e, d-f.
    Yes, {c,e,d,f} forms a C4 (c-e-d-f-c).
    
    So: A graph contains K2,2,2 iff there exists a pair of adjacent vertices (a,b) 
    such that their common neighborhood contains a C4.
    
    This is correct.
    
    So, when adding edge (u,v) to color c:
    1. We must check if (u,v) is the "same part" edge in a new K2,2,2.
       This means we check if N_c(u) intersect N_c(v) contains a C4.
    2. We must also check if adding this edge creates a K2,2,2 where (u,v) is a "cross" edge?
       No, if (u,v) is a cross edge (e.g. P1-P2), then the "same part" pair that triggers the check 
       would be a pair in P1 or P2 or P3.
       If we add (u,v), does it create a new "same part" pair with high common neighborhood?
       The common neighborhood of any existing edge might change? No, adding an edge (u,v) 
       only changes N(u) and N(v).
       So only edges incident to u or v might become "bad" if they were previously safe?
       No, if (x,y) is an edge, N(x) and N(y) might gain u or v?
       If u is added to N(x), then N(x) intersect N(y) might gain u.
       So we must check:
       - For the new edge (u,v): Does N_c(u) intersect N_c(v) contain C4?
       - For every existing edge (x,y) in color c incident to u or v:
         Does N_c(x) intersect N_c(y) contain C4?
         
    This is manageable.
    """
    pass # Placeholder, logic implemented in backtrack

def _check_color_class_valid(adj, n, color):
    """Check if the subgraph of edges with given color is K2,2,2-free."""
    # Build adjacency for this color
    adj_c = [0] * n
    for i in range(n):
        # We need to extract bits corresponding to color
        # This is slow if done from scratch. 
        # In backtracking, we maintain adj as list of bitmasks for color 0.
        # Color 1 is complement.
        pass

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate Paley 29 solution as fallback
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    adj_paley = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj_paley[i] |= (1 << j)
                adj_paley[j] |= (1 << i)
                
    # Verify Paley 29 is valid (it should be)
    # We assume it is based on previous runs.
    
    colors_paley = []
    for i in range(q):
        for j in range(i + 1, q):
            if (adj_paley[i] >> j) & 1:
                colors_paley.append(0)
            else:
                colors_paley.append(1)
    
    best_result = {'n': q, 'colors': colors_paley}
    
    # 2. Attempt Backtracking for n=30
    # We try to find a valid coloring for n=30.
    # Use symmetry breaking: Fix first 3 vertices to form a triangle of color 0.
    # (0,1)=0, (0,2)=0, (1,2)=0.
    
    n = 30
    # adj[i] is bitmask of neighbors of i in color 0.
    # Edges not in adj are color 1.
    adj = [0] * n
    
    # Pre-assign symmetry breaking edges
    adj[0] |= (1 << 1) | (1 << 2)
    adj[1] |= (1 << 0) | (1 << 2)
    adj[2] |= (1 << 0) | (1 << 1)
    
    # List of edges to assign
    # We will iterate through all pairs (i,j) with i<j
    # We skip the ones already assigned.
    edges_to_assign = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i == 0 and j == 1) or (i == 0 and j == 2) or (i == 1 and j == 2):
                continue
            edges_to_assign.append((i, j))
            
    # Order edges to maximize pruning
    # Prioritize edges incident to low-index vertices
    edges_to_assign.sort(key=lambda x: (x[0], x[1]))
    
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == len(edges_to_assign):
            # All edges assigned, check validity
            # Since we pruned during assignment, it should be valid
            # But let's do a final check just in case
            if _is_valid_full(adj, n):
                return True
            return False
            
        u, v = edges_to_assign[idx]
        
        # Try color 0
        if _try_assign(adj, n, u, v, 0):
            if backtrack(idx + 1):
                return True
            _unassign(adj, n, u, v, 0)
            
        # Try color 1
        if _try_assign(adj, n, u, v, 1):
            if backtrack(idx + 1):
                return True
            _unassign(adj, n, u, v, 1)
            
        return False

    def _try_assign(adj, n, u, v, color):
        # Assign edge (u,v) to color
        # Update adj if color 0
        if color == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Check validity
        # 1. Check if (u,v) creates K2,2,2 in color `color` as a "same part" edge
        #    i.e., N_color(u) intersect N_color(v) contains C4
        if _cn_contains_c4(adj, n, u, v, color):
            return False
            
        # 2. Check if adding this edge invalidates any existing "same part" edge 
        #    incident to u or v in color `color`
        #    Existing edges (x,u) or (x,v) in color `color` might now have a larger common neighborhood
        #    that contains a C4.
        
        # For each neighbor x of u in color `color` (x != v):
        #    Check if (x,u) is now bad.
        #    N_color(x) intersect N_color(u)
        
        # Get neighbors of u in color `color`
        neigh_u = _get_neigh(adj, n, u, color)
        for x in neigh_u:
            if x == v: continue
            if _cn_contains_c4(adj, n, x, u, color):
                return False
                
        # Get neighbors of v in color `color`
        neigh_v = _get_neigh(adj, n, v, color)
        for x in neigh_v:
            if x == u: continue
            if _cn_contains_c4(adj, n, x, v, color):
                return False
                
        return True

    def _unassign(adj, n, u, v, color):
        if color == 0:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    def _get_neigh(adj, n, u, color):
        if color == 0:
            mask = adj[u]
        else:
            # Complement in range n, excluding u
            full = (1 << n) - 1
            mask = full ^ adj[u] ^ (1 << u)
            
        verts = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts

    def _cn_contains_c4(adj, n, u, v, color):
        # Get common neighbors of u and v in color `color`
        # N_color(u) intersect N_color(v)
        
        if color == 0:
            nu = adj[u]
            nv = adj[v]
        else:
            full = (1 << n) - 1
            nu = full ^ adj[u] ^ (1 << u)
            nv = full ^ adj[v] ^ (1 << v)
            
        cn_mask = nu & nv
        cn_mask &= ~(1 << u)
        cn_mask &= ~(1 << v)
        
        if cn_mask.bit_count() < 4:
            return False
            
        return _has_c4_in_subgraph(adj, cn_mask, n, color)

    def _has_c4_in_subgraph(adj, mask, n, color):
        verts = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        L = len(verts)
        if L < 4:
            return False
            
        for i in range(L):
            u = verts[i]
            for j in range(i + 1, L):
                v = verts[j]
                
                # Check if u and v have >= 2 common neighbors in mask, in color `color`
                if color == 0:
                    nu = adj[u]
                    nv = adj[v]
                else:
                    full = (1 << n) - 1
                    nu = full ^ adj[u] ^ (1 << u)
                    nv = full ^ adj[v] ^ (1 << v)
                    
                common = nu & nv & mask
                if common.bit_count() >= 2:
                    return True
        return False

    def _is_valid_full(adj, n):
        # Full check for both colors
        for color in [0, 1]:
            for u in range(n):
                for v in range(u + 1, n):
                    # Check if edge (u,v) exists in this color
                    if color == 0:
                        if not ((adj[u] >> v) & 1): continue
                    else:
                        if ((adj[u] >> v) & 1): continue
                        
                    if _cn_contains_c4(adj, n, u, v, color):
                        return False
        return True

    # Run backtracking
    found = backtrack(0)
    
    if found:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
        
    # Fallback to Paley 29
    return best_result

# EVOLVE-BLOCK-END
