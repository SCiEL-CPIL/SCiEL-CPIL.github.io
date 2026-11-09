import random
import time
import itertools

def construct():
    n = 18
    # Bitmask representation of vertices
    # Edge (u, v) is represented by bit index u * n + v (for u < v)
    # Total edges: 18 * 17 / 2 = 153
    
    # Helper to get bit index for edge (u, v)
    def edge_idx(u, v):
        if u > v:
            u, v = v, u
        return u * n - (u * (u + 1) // 2) + v - u - 1 
        # Wait, simpler mapping:
        # idx = 0
        # for i in range(n):
        #   for j in range(i+1, n):
        #     if (i,j) == (u,v): return idx
        #     idx += 1
        # This is slow. Let's use a lookup table.
        
    # Precompute edge indices
    edge_indices = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices[(u, v)] = idx
            idx += 1
            
    # Precompute all triangles (sets of 3 vertices)
    # There are C(18, 3) = 816 triangles
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
                
    # Precompute all subsets of vertices by size
    # We need to check if a subset is triangle-free.
    # A subset S is triangle-free if it contains no triangle from the graph.
    # We can represent a subset as a bitmask of vertices.
    
    # To speed up "is S triangle-free", we can precompute for each triangle t, 
    # the bitmask of vertices it contains.
    # Then S is triangle-free iff for all t, (S & t_mask) != t_mask.
    
    tri_masks = []
    for u, v, w in triangles:
        mask = (1 << u) | (1 << v) | (1 << w)
        tri_masks.append(mask)
        
    # We also need to check K4-freeness.
    # A graph is K4-free if it contains no K4.
    # A K4 is a set of 4 vertices with all 6 edges present.
    # We can precompute all K4s (sets of 4 vertices).
    # C(18, 4) = 3060 K4s.
    
    k4s = []
    k4_edge_masks = [] # Bitmask of edges required for this K4
    for quad in itertools.combinations(range(n), 4):
        u, v, w, x = quad
        edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
        mask = 0
        for e in edges:
            mask |= (1 << edge_indices[e])
        k4s.append(quad)
        k4_edge_masks.append(mask)
        
    # Function to check if graph (edge_mask) is K4-free
    def is_k4_free(edge_mask):
        for k4_mask in k4_edge_masks:
            if (edge_mask & k4_mask) == k4_mask:
                return False
        return True
        
    # Function to find max triangle-free induced subgraph size
    def max_triangle_free_size(edge_mask):
        # We iterate subsets from largest to smallest
        # But checking all subsets is expensive.
        # We can use a heuristic or branch and bound.
        # However, for n=18, C(18, k) is manageable for k up to 12.
        # We want to minimize the score, so we hope the score is small (e.g., 8).
        # We can check k from 18 down to 1.
        
        # Optimization: 
        # If we find a triangle-free subset of size k, we can stop if we are looking for max?
        # No, we want to know the MAX size.
        # So we start from 18 and go down until we find one.
        # But most graphs will have small triangle-free subgraphs.
        # Actually, we want to MINIMIZE the max size.
        # So in the search, we evaluate a graph by finding its max triangle-free size.
        
        # To speed up, we can precompute all subsets of size k for k=1..18?
        # No, just generate on the fly or precompute lists.
        
        # Let's precompute subsets by size
        # subsets_by_size[k] = list of bitmasks
        
        # This is too much memory? 2^18 = 262144 masks. It's fine.
        # But we only need to check if ANY subset of size k is triangle-free.
        
        # Better approach for evaluation:
        # For a given edge_mask, compute the set of triangles present.
        # tri_present = [i for i, t in enumerate(triangles) if (edge_mask & t_edge_mask) == t_edge_mask]
        # Wait, we need edge mask for triangles too.
        
        tri_edge_masks = []
        for u, v, w in triangles:
            mask = 0
            mask |= (1 << edge_indices[(u,v)])
            mask |= (1 << edge_indices[(u,w)])
            mask |= (1 << edge_indices[(v,w)])
            tri_edge_masks.append(mask)
            
        # For a given graph, find which triangles are present.
        present_tris = 0 # Bitmask of triangle indices? Or list?
        # There are 816 triangles. A list of booleans is fine.
        # Or a set of triangle indices.
        
        # Actually, we can just check each subset.
        # For subset S (vertex bitmask), it is triangle-free if for all t in triangles, S does not contain t.
        # i.e., (S & tri_vertex_mask[t]) != tri_vertex_mask[t] for all t.
        
        # To make this fast, we can precompute for each vertex bitmask S, 
        # the list of triangles it contains? No, that's static.
        
        # Let's just iterate k from 18 down to 1.
        # For each k, iterate all subsets of size k.
        # If any is triangle-free, return k.
        
        # Precompute subsets of size k
        # This is done once outside? No, depends on n.
        # Let's generate them once.
        
        # Note: This evaluation function is called many times.
        # We need it to be fast.
        
        # Alternative: 
        # The score is the size of the largest triangle-free induced subgraph.
        # We can use a branch and bound to find if there is a triangle-free subgraph of size K.
        # If yes, score >= K.
        # We want to minimize score.
        
        # Let's use a simple loop for now, but optimize the inner check.
        
        # Precompute subsets by size
        subsets_by_size = [None] * (n + 1)
        for k in range(1, n + 1):
            subsets_by_size[k] = [mask for mask in range(1 << n) if bin(mask).count('1') == k]
            
        # This precomputation takes time. 2^18 * 18 operations? 
        # 262144 * 18 ~ 4.7 million. Acceptable at start.
        
        # Now, for a given edge_mask:
        # 1. Determine which triangles are present.
        #    present_tri_flags = [(edge_mask & tri_edge_masks[i]) == tri_edge_masks[i] for i in range(len(triangles))]
        # 2. For k from 18 down to 1:
        #    for S in subsets_by_size[k]:
        #       # Check if S contains any present triangle
        #       is_tf = True
        #       for i, flag in enumerate(present_tri_flags):
        #           if flag:
        #               if (S & tri_masks[i]) == tri_masks[i]:
        #                   is_tf = False
        #                   break
        #       if is_tf:
        #           return k
        # 3. If no k found (impossible, k=1 always works), return 0.
        
        # This is still potentially slow if we check many triangles for each subset.
        
        # Optimization:
        # If S is triangle-free, it must be an independent set in the 3-uniform hypergraph of triangles.
        
        # Let's try to make the search smarter.
        # We will use Simulated Annealing.
        
    # Precomputation
    subsets_by_size = [None] * (n + 1)
    for k in range(1, n + 1):
        subsets_by_size[k] = [mask for mask in range(1 << n) if bin(mask).count('1') == k]
        
    def get_score(edge_mask):
        # Check K4-free first
        if not is_k4_free(edge_mask):
            return 1000 # Penalty
            
        # Compute present triangles
        # We can represent present triangles as a bitmask of triangle indices?
        # 816 bits is 102 bytes. Python int handles this.
        present_tri_mask = 0
        for i in range(len(triangles)):
            if (edge_mask & tri_edge_masks[i]) == tri_edge_masks[i]:
                present_tri_mask |= (1 << i)
                
        # Now check subsets
        # We want to find the max k such that there exists S of size k with no present triangle.
        # For a subset S, let T_S be the set of triangles contained in S.
        # S is triangle-free iff T_S is empty.
        # i.e., T_S & present_tri_mask == 0.
        
        # Precompute T_S for all S?
        # S has 2^18 possibilities.
        # T_S can be computed as:
        # T_S = OR of (1 << i) for all i such that tri_masks[i] is subset of S.
        # This is a subset convolution or similar.
        # Since 2^18 is small, we can compute T_S for all S once?
        # No, T_S depends only on S, not on the graph!
        # The set of geometric triangles contained in S is fixed.
        # Let's call this GeomTriMask[S].
        
        # GeomTriMask[S] = bitmask of triangle indices contained in S.
        # We can precompute this array of size 2^18.
        
        # How to compute GeomTriMask[S]?
        # For each triangle i, it is contained in S if (S & tri_masks[i]) == tri_masks[i].
        # We can iterate all S and all triangles? 2^18 * 816 ~ 214 million. Too slow.
        
        # Better:
        # For each triangle i, it is contained in all supersets of tri_masks[i].
        # We can use SOS DP (Sum Over Subsets) or similar.
        # Or simply:
        # Initialize GeomTriMask[S] = 0 for all S.
        # For each triangle i:
        #   base = tri_masks[i]
        #   For each superset S of base:
        #      GeomTriMask[S] |= (1 << i)
        
        # Number of supersets of a 3-set in 18-set is 2^(18-3) = 16384.
        # 816 triangles * 16384 ~ 13 million operations.
        # This is acceptable at startup!
        
        # Let's do this precomputation.
        
    GeomTriMask = [0] * (1 << n)
    for i in range(len(triangles)):
        base = tri_masks[i]
        # Iterate all supersets of base
        # Supersets of base are base | sub where sub is subset of (all_vertices ^ base)
        complement = ((1 << n) - 1) ^ base
        sub = complement
        while True:
            S = base | sub
            GeomTriMask[S] |= (1 << i)
            if sub == 0:
                break
            sub = (sub - 1) & complement
            
    # Now get_score is fast:
    def get_score(edge_mask):
        if not is_k4_free(edge_mask):
            return 1000
            
        present_tri_mask = 0
        for i in range(len(triangles)):
            if (edge_mask & tri_edge_masks[i]) == tri_edge_masks[i]:
                present_tri_mask |= (1 << i)
                
        # Check subsets from largest to smallest
        for k in range(n, 0, -1):
            for S in subsets_by_size[k]:
                if (GeomTriMask[S] & present_tri_mask) == 0:
                    return k
        return 0
        
    # Initial graph: Random K4-free
    # Start with complete graph and remove edges to break K4s?
    # Or start with K6,6,6 and perturb.
    
    # Let's start with a random graph and fix K4s.
    # Or start with K6,6,6.
    
    # K6,6,6 edges
    def make_k666():
        mask = 0
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        for i in range(3):
            for j in range(i + 1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        mask |= (1 << edge_indices[(min(u,v), max(u,v))])
        return mask
        
    current_mask = make_k666()
    current_score = get_score(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    # Local Search
    # We will try to flip edges.
    # Flipping an edge might create a K4.
    # We need to ensure K4-freeness.
    
    # Strategy: 
    # 1. Pick a random edge.
    # 2. Flip it.
    # 3. If K4-free, compute new score.
    # 4. If new score < current score, accept.
    # 5. Else, accept with probability exp(-delta/T).
    
    # To speed up K4 check, we can check only K4s involving the flipped edge.
    
    # Precompute for each edge, the list of K4s it belongs to.
    edge_to_k4s = [[] for _ in range(153)]
    for idx, quad in enumerate(k4s):
        u, v, w, x = quad
        edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
        for e in edges:
            edge_to_k4s[edge_indices[e]].append(idx)
            
    def is_k4_free_fast(mask, edge_idx_to_check, old_val):
        # Check only K4s containing edge_idx_to_check
        # A K4 is present if all 6 edges are present.
        # If we flipped edge e, and it was 0->1, we need to check if any K4 containing e is now complete.
        # If 1->0, the graph remains K4-free if it was before (removing edges doesn't create K4s).
        # So we only need to check when adding an edge (0->1).
        
        if old_val == 1:
            return True # Removing edge cannot create K4
            
        # Adding edge
        for k4_idx in edge_to_k4s[edge_idx_to_check]:
            k4_mask = k4_edge_masks[k4_idx]
            if (mask & k4_mask) == k4_mask:
                return False
        return True

    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    T = 1.0
    cooling = 0.999
    
    # To improve, we can bias the search towards denser graphs (more triangles)
    # because dense graphs have smaller triangle-free subgraphs.
    
    while time.time() < deadline:
        # Pick random edge
        e_idx = random.randint(0, 152)
        
        # Current value
        old_val = (current_mask >> e_idx) & 1
        new_val = 1 - old_val
        
        # New mask
        new_mask = current_mask ^ (1 << e_idx)
        
        # Check K4-free
        if not is_k4_free_fast(new_mask, e_idx, old_val):
            continue
            
        # Compute score
        new_score = get_score(new_mask)
        
        if new_score < best_score:
            best_score = new_score
            best_mask = new_mask
            
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_mask = new_mask
            current_score = new_score
            
        T *= cooling
        
    # Convert best_mask to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_mask >> edge_indices[(u, v)]) & 1:
                edges.append([u, v])
                
    return edges

# Run once to test? No, construct is called by evaluator.
# The precomputation inside construct might be slow if called repeatedly?
# "Each evaluation must finish within 120s".
# Precomputation is ~13M ops + 4.7M ops. In Python, this might take a few seconds.
# We should move precomputation to module level if possible, or ensure it's fast.
# The prompt says "Keep the function names...". It doesn't forbid global variables.
# But to be safe and clean, I'll keep it inside but optimize.
# Actually, 13M operations in pure Python might take 5-10 seconds.
# 120s limit is generous.

# Let's refine the code to be robust.
