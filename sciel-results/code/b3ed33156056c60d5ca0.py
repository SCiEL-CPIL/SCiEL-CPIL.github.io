import time
import random
import math

def _paley_37_edges():
    """Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

def _build_bitmasks(n, edges):
    """Build adjacency bitmasks for fast clique/independent set checks."""
    masks = [0] * n
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _count_k5_and_ind5(n, masks):
    """Count K5s and independent 5-sets."""
    k5_count = 0
    ind5_count = 0
    
    # Count K5s
    for i in range(n):
        for j in range(i+1, n):
            if not (masks[i] >> j) & 1:
                continue
            common_ij = masks[i] & masks[j]
            for l in range(j+1, n):
                if not (common_ij >> l) & 1:
                    continue
                common_ijl = common_ij & masks[l]
                for m in range(l+1, n):
                    if not (common_ijl >> m) & 1:
                        continue
                    common_ijlm = common_ijl & masks[m]
                    for p in range(m+1, n):
                        if (common_ijlm >> p) & 1:
                            k5_count += 1
                            
    # Count Ind5s (K5s in complement)
    comp_masks = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_masks[i] = full_mask ^ masks[i] ^ (1 << i)
        
    for i in range(n):
        for j in range(i+1, n):
            if not (comp_masks[i] >> j) & 1:
                continue
            common_ij = comp_masks[i] & comp_masks[j]
            for l in range(j+1, n):
                if not (common_ij >> l) & 1:
                    continue
                common_ijl = common_ij & comp_masks[l]
                for m in range(l+1, n):
                    if not (common_ijl >> m) & 1:
                        continue
                    common_ijlm = common_ijl & comp_masks[m]
                    for p in range(m+1, n):
                        if (common_ijlm >> p) & 1:
                            ind5_count += 1
                            
    return k5_count, ind5_count

def _calculate_delta_penalty(n, masks, u, v):
    """
    Calculate change in penalty if edge (u, v) is flipped.
    Returns (delta_k5, delta_ind5).
    """
    is_edge = (masks[u] >> v) & 1
    
    delta_k5 = 0
    delta_ind5 = 0
    
    mask_u = masks[u]
    mask_v = masks[v]
    
    # Iterate over all triples (a, b, c) from V \ {u, v}
    # Optimization: iterate a < b < c, skipping u, v
    # We can just loop 0..n and skip u, v
    
    # To be faster, we can precompute the list of vertices excluding u, v
    # But creating a list every time is O(n).
    # Looping 0..n and checking is O(n^3) with small constant.
    
    # Let's use a flat loop for speed? No, nested is better for pruning.
    
    # We need to check if {a,b,c} is K3 or Ind3.
    # And check connections to u, v.
    
    # Let's iterate a, b, c.
    for a in range(n):
        if a == u or a == v: continue
        ua = (mask_u >> a) & 1
        va = (mask_v >> a) & 1
        for b in range(a+1, n):
            if b == u or b == v: continue
            ab = (masks[a] >> b) & 1
            ub = (mask_u >> b) & 1
            vb = (mask_v >> b) & 1
            for c in range(b+1, n):
                if c == u or c == v: continue
                ac = (masks[a] >> c) & 1
                bc = (masks[b] >> c) & 1
                uc = (mask_u >> c) & 1
                vc = (mask_v >> c) & 1
                
                is_k3 = ab and ac and bc
                is_ind3 = (not ab) and (not ac) and (not bc)
                
                if is_k3:
                    if ua and ub and uc and va and vb and vc:
                        if is_edge:
                            delta_k5 -= 1
                        else:
                            delta_k5 += 1
                            
                if is_ind3:
                    if (not ua) and (not ub) and (not uc) and (not va) and (not vb) and (not vc):
                        if is_edge:
                            delta_ind5 += 1
                        else:
                            delta_ind5 -= 1
                            
    return delta_k5, delta_ind5

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # Start with Paley 37
    n0, edges0 = _paley_37_edges()
    best_n = n0
    best_edges = edges0
    
    # Try to expand to 38, 39, 40
    for target_n in [38, 39, 40]:
        if time.time() > deadline - 30:
            break
            
        found = False
        
        # Multiple restarts
        for restart in range(10):
            if time.time() > deadline - 10:
                break
                
            # Initialize random graph
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.5:
                        edges.append((i, j))
            
            masks = _build_bitmasks(target_n, edges)
            k5, ind5 = _count_k5_and_ind5(target_n, masks)
            penalty = k5 + ind5
            
            T = 1.0
            min_T = 0.01
            cool_rate = 0.99
            
            max_iters = 2000
            iters = 0
            stuck_count = 0
            
            while iters < max_iters and time.time() < deadline - 5:
                iters += 1
                
                if penalty == 0:
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
                    
                # Pick random edge
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                    
                # Calculate delta
                dk5, dind5 = _calculate_delta_penalty(target_n, masks, u, v)
                delta = dk5 + dind5
                
                # Accept?
                if delta <= 0:
                    accept = True
                else:
                    if T > 0:
                        accept = random.random() < math.exp(-delta / T)
                    else:
                        accept = False
                        
                if accept:
                    # Flip edge
                    if (masks[u] >> v) & 1:
                        # Remove edge
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                        edges.remove((u, v))
                    else:
                        # Add edge
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        edges.append((u, v))
                        
                    penalty += delta
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 500:
                        break
                        
                T *= cool_rate
                
            if found:
                break
                
    return best_n, best_edges
