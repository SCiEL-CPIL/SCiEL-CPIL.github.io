import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 18
    
    # Helper to check if adding edge (u, v) creates a K4
    def creates_k4(adj, u, v):
        # Check if there is an edge in the common neighborhood of u and v
        common = adj[u] & adj[v]
        # Check if common contains an edge
        # Iterate through vertices in common
        m = common
        while m:
            lb = m & (-m)
            x = lb.bit_length() - 1
            m ^= lb
            if adj[x] & common:
                return True
        return False

    # Precompute triangle masks: for each pair (u,v) that is an edge, 
    # common_neighbors[u,v] = adj[u] & adj[v]
    # A set S is triangle-free iff for every edge (u,v) in S, common_neighbors[u,v] & S == 0
    
    # Helper to evaluate largest triangle-free induced subgraph
    # Use a faster method: iterate over subsets but with early termination
    def eval_max_tf(adj):
        # Precompute for each vertex v, the set of vertices that form a triangle with v
        # tri_with[v] = set of w such that there exists u < w with u,v,w forming a triangle
        # Actually, simpler: precompute all triangles as (u,v,w) with u<v<w
        # Then a set S is triangle-free iff it contains no triangle
        
        # Precompute triangles
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # For each w in common with w > v
                    cm = common & (~((1 << (v+1)) - 1))
                    while cm:
                        lb = cm & (-cm)
                        w = lb.bit_length() - 1
                        cm ^= lb
                        triangles.append((u, v, w))
        
        # Now check subsets: a mask is triangle-free iff no triangle is subset of mask
        # Fast check: for each triangle (u,v,w), if mask has all three, not TF
        # But this is still O(#triangles * 2^n) in worst case
        
        # Better: use the fact that if we build S incrementally, we can check
        # when adding vertex v, does it create a triangle with existing vertices?
        
        # Use B&B with memoization-like pruning
        best = 0
        
        # Sort vertices by degree
        verts = list(range(n))
        verts.sort(key=lambda v: adj[v].bit_count(), reverse=True)
        
        # For each vertex v, precompute which other vertices it forms triangles with
        # tri_partners[v] = bitmask of vertices w such that v,w are in some triangle
        tri_partners = [0] * n
        for (u, v, w) in triangles:
            tri_partners[u] |= (1 << v) | (1 << w)
            tri_partners[v] |= (1 << u) | (1 << w)
            tri_partners[w] |= (1 << u) | (1 << v)
        
        def can_add(mask, v):
            # Can we add v to mask without creating a triangle?
            # v creates a triangle if there exist u,w in mask such that u,v,w is a triangle
            # Equivalent: if (mask & tri_partners[v]) contains an edge where both endpoints are neighbors of v
            # Simpler: check if there exist u in mask such that u is neighbor of v and 
            # (mask & adj[u] & adj[v]) is non-empty
            # Even simpler: for each neighbor u of v in mask, if adj[u] & mask & adj[v] has a vertex w != u,v
            # Actually: triangle (u,v,w) with u,w in mask. So if there exist u,w in mask 
            # with u~v and w~v and u~w
            # Check: for each u in (mask & adj[v]), if adj[u] & mask & adj[v] & ~(1<<u) is non-empty
            nm = mask & adj[v]
            rm = nm
            while rm:
                lb = rm & (-rm)
                u = lb.bit_length() - 1
                rm ^= lb
                if adj[u] & mask & adj[v] & ~(1 << u):
                    return False
            return True
        
        def bb(pos, current_mask, current_size):
            nonlocal best
            if pos >= n:
                if current_size > best:
                    best = current_size
                return
            
            # Upper bound
            if current_size + (n - pos) <= best:
                return
            
            v = verts[pos]
            # Try include
            if can_add(current_mask, v):
                bb(pos + 1, current_mask | (1 << v), current_size + 1)
            
            # Try exclude
            bb(pos + 1, current_mask, current_size)
        
        bb(0, 0, 0)
        return best

    best_score = n # Max possible is 18
    best_edges = []
    
    # Try a few random constructions
    # 1. Turan graph K_{6,6,6}
    adj_turan = [0] * n
    edges_turan = []
    for i in range(n):
        for j in range(i + 1, n):
            if i // 6 != j // 6:
                adj_turan[i] |= (1 << j)
                adj_turan[j] |= (1 << i)
                edges_turan.append([i, j])
    
    score_turan = eval_max_tf(adj_turan)
    if score_turan < best_score:
        best_score = score_turan
        best_edges = edges_turan.copy()
    
    # 2. Random Greedy K4-free
    for _ in range(50): # 50 restarts
        if time.time() > deadline - 10:
            break
            
        adj = [0] * n
        edges = []
        
        # Random order of edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u, v) creates K4
            if not creates_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        score = eval_max_tf(adj)
        if score < best_score:
            best_score = score
            best_edges = edges.copy()
            
        # Local search: try to remove edges to reduce triangle-free subgraphs
        # Remove an edge (u,v) if it helps reduce the max triangle-free set
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # Try removing each edge
            for ei in range(len(edges) - 1, -1, -1):
                if time.time() > deadline - 15:
                    break
                u, v = edges[ei]
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.pop(ei)
                
                new_score = eval_max_tf(adj)
                if new_score < score:
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_edges = edges.copy()
                    improved = True
                else:
                    # Restore
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.insert(ei, [u, v])
    return best_edges
