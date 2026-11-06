import random
import math
import time

def construct():
    n = 18
    
    # Precompute bits
    all_mask = (1 << n) - 1
    
    # State: adj_mask list of length n
    # adj_mask[i] has bit j set if edge (i,j) exists
    
    def is_k4_free(adj_mask):
        # Check all 4-subsets? No, check for each edge (u,v), if common[u][v] has an edge.
        # Precompute common on the fly? 
        # For each pair (u,v) that is an edge:
        #   common = adj_mask[u] & adj_mask[v]
        #   Check if any edge exists in common.
        #   To check if any edge exists in mask M:
        #     Iterate bits in M, for each bit i, check if adj_mask[i] & M has any bit other than i?
        #     Simpler: for each i in M, if (adj_mask[i] & M) != 0 (and not just self?), then edge exists.
        #     Since adj_mask[i] doesn't have bit i, (adj_mask[i] & M) != 0 implies edge.
        
        for u in range(n):
            mask_u = adj_mask[u]
            # Iterate neighbors v of u
            v = 0
            while v < n:
                if mask_u & (1 << v):
                    # Edge (u,v) exists
                    common = mask_u & adj_mask[v]
                    # Check if common contains an edge
                    # For each w in common, check if adj_mask[w] & common != 0
                    # But we only need to check if there is ANY edge in common.
                    # We can just check if for any w in common, (adj_mask[w] & common) != 0.
                    # Note: w is in common, so w is neighbor of u and v.
                    # If w is connected to x in common, then (u,v,w,x) is K4?
                    # No, (u,v,w,x) is K4 if u-v, u-w, u-x, v-w, v-x, w-x.
                    # We have u-v, u-w, u-x (if x in common), v-w, v-x (if x in common).
                    # We just need w-x edge.
                    # So yes, if common contains an edge, K4 exists.
                    
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & -c
                        w_idx = lsb.bit_length() - 1
                        if adj_mask[w_idx] & common:
                            has_edge = True
                            break
                        c ^= lsb
                    if has_edge:
                        return False
                v += 1
        return True

    def max_triangle_free_size(adj_mask):
        # Branch and Bound
        # best_size
        best_size = 0
        
        # Precompute common neighbors for all pairs?
        # common[u][v] = adj_mask[u] & adj_mask[v]
        # This is 18x18 = 324 integers. Fast to compute.
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u][v] = adj_mask[u] & adj_mask[v]
        
        def dfs(c_mask, p_mask):
            nonlocal best_size
            # c_mask: current triangle-free set
            # p_mask: potential vertices that can be added without creating triangle with c_mask
            # Actually, p_mask should be vertices that are "compatible" with c_mask.
            # If we add v to c_mask, new p_mask must be subset of p_mask \ {v} \ {incompatible with v given c_mask}
            
            # Pruning
            if bin(c_mask).count('1') + bin(p_mask).count('1') <= best_size:
                return
            
            if p_mask == 0:
                if bin(c_mask).count('1') > best_size:
                    best_size = bin(c_mask).count('1')
                return
            
            # Pick a vertex v from p_mask
            # Heuristic: pick v with fewest connections in p_mask? Or just first one.
            # Let's pick the lowest set bit.
            v_lsb = p_mask & -p_mask
            v_idx = v_lsb.bit_length() - 1
            p_mask ^= v_lsb
            
            # Option 1: Include v
            # New c_mask = c_mask | v_lsb
            # New p_mask: 
            # Vertices in p_mask that do not form a triangle with v and any u in c_mask.
            # Triangle (v, u, w) where u in c_mask, w in p_mask.
            # This happens if w is connected to v AND w is connected to u AND u-v is edge?
            # No. Triangle is (v, u, w).
            # Edges: v-u, u-w, w-v.
            # We are adding v. c_mask is triangle-free.
            # If we add v, we must ensure no triangle formed.
            # Triangles involving v and two vertices from c_mask:
            #   (v, u1, u2) where u1, u2 in c_mask.
            #   This requires u1-u2 edge (impossible, c_mask is TF) OR u1-v and u2-v and u1-u2?
            #   Wait, if c_mask is TF, no u1, u2 in c_mask are connected to each other? No, they can be connected, just no triangle.
            #   So u1-u2 can be an edge.
            #   If u1-u2 is edge, and v connects to both, then (v, u1, u2) is triangle.
            #   So v can be added only if for all u1, u2 in c_mask, it is NOT the case that (u1-u2 edge AND v-u1 AND v-u2).
            #   Actually, we check this BEFORE adding v. If valid, we add v.
            
            # If v is added, which vertices in p_mask become invalid?
            # A vertex w in p_mask becomes invalid if it forms a triangle with v and some u in c_mask.
            # Triangle (v, u, w).
            # Requires: v-u edge, u-w edge, w-v edge.
            # So w is invalid if there exists u in c_mask such that:
            #   (adj_mask[v] & (1<<u)) AND (adj_mask[w] & (1<<u)) AND (adj_mask[v] & (1<<w))
            # Since w is in p_mask, w is compatible with c_mask.
            # We need to filter p_mask.
            
            # Let's compute invalid_mask for p_mask given we add v.
            invalid = 0
            # For each u in c_mask:
            c = c_mask
            while c:
                lsb_u = c & -c
                u_idx = lsb_u.bit_length() - 1
                # w is invalid if w connects to v AND w connects to u AND u connects to v.
                # u connects to v? If not, no triangle (v,u,w) via u.
                if adj_mask[u_idx] & v_lsb:
                    # u-v is edge.
                    # w must connect to v and u.
                    # w connects to v: (adj_mask[v] & (1<<w))
                    # w connects to u: (adj_mask[u] & (1<<w))
                    # So w is in (adj_mask[v] & adj_mask[u_idx] & p_mask)
                    # But we also need w-v edge.
                    # common[v][u_idx] = adj_mask[v] & adj_mask[u_idx]
                    # So w in common[v][u_idx] & p_mask
                    invalid |= common[v_idx][u_idx]
                c ^= lsb_u
            
            # Also, w might form triangle with v and another vertex in p_mask?
            # No, we are only adding v to c_mask. The new c_mask is c_mask U {v}.
            # The new p_mask is p_mask \ {v} \ invalid.
            # But wait, if we add v, do we need to check triangles between v and two vertices in the NEW p_mask?
            # No, because those vertices are not in c_mask yet.
            # The condition for p_mask is that any subset of p_mask added to c_mask remains TF.
            # So p_mask vertices must be mutually compatible with c_mask U {v}.
            # They don't need to be compatible with each other yet?
            # If w1, w2 in new p_mask form a triangle with v, that's fine as long as we don't add both?
            # No, if w1, w2, v form a triangle, we can't add both w1 and w2 to c_mask if v is in c_mask.
            # But we can add one.
            # So the "compatibility" of p_mask is with respect to c_mask.
            # The standard B&B for Max Independent Set works this way.
            # For Max Triangle-Free Set:
            # If we add v, we remove from p_mask any w that forms a triangle with v and some u in c_mask.
            # We do NOT remove w just because w and v form a triangle with each other (impossible, 2 vertices).
            # We do NOT remove w just because w and v and some OTHER w' in p_mask form a triangle?
            # If w, v, w' form a triangle, and we add v, then we cannot add BOTH w and w'.
            # But we can add one.
            # So we don't remove them from p_mask immediately.
            # However, if w and w' are both in p_mask, and w-w' is an edge, and both connect to v, then (v,w,w') is a triangle.
            # If we add v, we must ensure we don't pick both w and w'.
            # This is handled by the recursive structure: if we pick w later, w' will be removed if it forms a triangle with w and v (which is in c_mask).
            
            new_c_mask = c_mask | v_lsb
            new_p_mask = p_mask & ~invalid & ~v_lsb # v is removed from p_mask
            
            # Check if new_c_mask is actually TF? 
            # We ensured v doesn't form triangle with c_mask.
            # And c_mask was TF.
            # So new_c_mask is TF.
            
            dfs(new_c_mask, new_p_mask)
            
            # Option 2: Exclude v
            dfs(c_mask, p_mask) # v is already removed from p_mask
            
        # Initial call
        # c_mask = 0
        # p_mask = all_mask
        # But we must ensure p_mask vertices are compatible with c_mask (empty).
        # Any set is compatible with empty.
        dfs(0, all_mask)
        
        return best_size

    # Simulated Annealing
    random.seed(42)
    
    # Initialize with Turan T(18,3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj_mask = [0]*n
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
    
    best_adj = adj_mask[:]
    best_score = max_triangle_free_size(best_adj)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    T_start = 10.0
    T_end = 0.01
    iterations = 5000
    
    for i in range(iterations):
        T = T_start * (T_end / T_start) ** (i / iterations)
        
        # Pick random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        
        # Flip edge (u,v)
        bit = (1 << v)
        adj_mask[u] ^= bit
        adj_mask[v] ^= (1 << u)
        
        # Check K4
        if not is_k4_free(adj_mask):
            # Revert
            adj_mask[u] ^= bit
            adj_mask[v] ^= (1 << u)
            continue
            
        new_score = max_triangle_free_size(adj_mask)
        
        if new_score < current_score:
            current_score = new_score
            current_adj = adj_mask[:]
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            # Accept with probability
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                    current_adj = adj_mask[:]
                else:
                    # Revert
                    adj_mask[u] ^= bit
                    adj_mask[v] ^= (1 << u)
            else:
                # Should not happen if new_score >= current_score handled above?
                # If new_score == current_score, accept?
                current_score = new_score
                current_adj = adj_mask[:]
                
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        m = best_adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
            
    return edges
