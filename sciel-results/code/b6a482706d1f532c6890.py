import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    n = 40
    best_edges = []
    best_count = 0
    
    # Fast C6 check using bitmasks
    def check_c6_with_edge(mask, u, v):
        """
        Check if adding edge (u,v) to graph 'mask' creates a C6.
        A new C6 must use the new edge (u,v).
        So we need a path of length 5 from u to v in the current graph:
        u - a - b - c - d - v, with a,b,c,d distinct and not u,v.
        """
        nu = mask[u]
        nv = mask[v]
        
        a_candidates = nu & ~(1 << v)
        d_candidates = nv & ~(1 << u)
        
        while a_candidates:
            a_bit = a_candidates & (-a_candidates)
            a = a_bit.bit_length() - 1
            a_candidates ^= a_bit
            
            d_cand = d_candidates & ~(1 << a)
            while d_cand:
                d_bit = d_cand & (-d_cand)
                d = d_bit.bit_length() - 1
                d_cand ^= d_bit
                
                na = mask[a]
                b_cand = na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                while b_cand:
                    b_bit = b_cand & (-b_cand)
                    b = b_bit.bit_length() - 1
                    b_cand ^= b_bit
                    
                    c_cand = (mask[d] & mask[b]) & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                    if c_cand:
                        return True
        return False

    def is_edge_in_c6(mask, u, v):
        """Check if existing edge (u,v) is part of a C6 in mask."""
        orig_u = mask[u]
        orig_v = mask[v]
        mask[u] &= ~(1 << v)
        mask[v] &= ~(1 << u)
        res = check_c6_with_edge(mask, u, v)
        mask[u] = orig_u
        mask[v] = orig_v
        return res

    def random_graph(p, rng):
        mask = [0] * n
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    mask[i] |= (1 << j)
                    mask[j] |= (1 << i)
                    edges.append((i, j))
        return mask, edges

    def prune_c6(mask, edges, rng):
        """Remove edges that are part of C6s."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            rng.shuffle(edges)
            for i in range(len(edges)):
                u, v = edges[i]
                if is_edge_in_c6(mask, u, v):
                    mask[u] &= ~(1 << v)
                    mask[v] &= ~(1 << u)
                    edges.pop(i)
                    changed = True
                    break
        return mask, edges

    def greedy_add(mask, edges, rng):
        """Add all possible non-edges that don't create C6."""
        added = True
        while added and time.time() < deadline:
            added = False
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (mask[i] >> j) & 1:
                        non_edges.append((i, j))
            
            rng.shuffle(non_edges)
            
            for u, v in non_edges:
                if not check_c6_with_edge(mask, u, v):
                    mask[u] |= (1 << v)
                    mask[v] |= (1 << u)
                    edges.append((u, v))
                    added = True
        return mask, edges

    def local_search_swap(mask, edges, rng):
        """
        Try 1-for-1 and 1-for-2 swaps to escape local optima.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing an edge and adding two others
            # Sample random edges to try removing
            if not edges:
                break
            rng.shuffle(edges)
            
            for e_remove in edges:
                if time.time() > deadline:
                    return mask, edges, False
                u_r, v_r = e_remove
                # Save state
                old_mask = [m for m in mask]
                old_edges = list(edges)
                
                # Remove edge
                mask[u_r] &= ~(1 << v_r)
                mask[v_r] &= ~(1 << u_r)
                edges.remove(e_remove)
                
                # Try to add two edges
                addable = []
                for i in range(n):
                    for j in range(i+1, n):
                        if i == u_r and j == v_r:
                            continue
                        if not (mask[i] >> j) & 1:
                            if not check_c6_with_edge(mask, i, j):
                                addable.append((i, j))
                
                if len(addable) >= 2:
                    rng.shuffle(addable)
                    e1 = addable[0]
                    e2 = addable[1]
                    mask[e1[0]] |= (1 << e1[1])
                    mask[e1[1]] |= (1 << e1[0])
                    edges.append(e1)
                    mask[e2[0]] |= (1 << e2[1])
                    mask[e2[1]] |= (1 << e2[0])
                    edges.append(e2)
                    improved = True
                    break
                else:
                    # Revert
                    mask[:] = old_mask
                    edges[:] = old_edges
                    
            # Also try 1-for-1 swaps
            if not improved and time.time() < deadline:
                rng.shuffle(edges)
                for e_remove in edges[:20]:  # Limit to keep it fast
                    if time.time() > deadline:
                        return mask, edges, False
                    u_r, v_r = e_remove
                    old_mask = [m for m in mask]
                    old_edges = list(edges)
                    
                    mask[u_r] &= ~(1 << v_r)
                    mask[v_r] &= ~(1 << u_r)
                    edges.remove(e_remove)
                    
                    addable = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if i == u_r and j == v_r:
                                continue
                            if not (mask[i] >> j) & 1:
                                if not check_c6_with_edge(mask, i, j):
                                    addable.append((i, j))
                    
                    if addable:
                        # Pick the best one (arbitrary for now)
                        e_add = addable[0]
                        mask[e_add[0]] |= (1 << e_add[1])
                        mask[e_add[1]] |= (1 << e_add[0])
                        edges.append(e_add)
                        improved = True
                        break
                    else:
                        mask[:] = old_mask
                        edges[:] = old_edges
                        
        return mask, edges, improved

    # Main Loop: Multiple restarts with different densities
    rng = random.Random()
    densities = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        p = rng.choice(densities)
        mask, edges = random_graph(p, rng)
        
        mask, edges = prune_c6(mask, edges, rng)
        mask, edges = greedy_add(mask, edges, rng)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [[u, v] for u, v in edges]
            
        # Local search to escape local optima
        if time.time() < deadline - 10:
            mask, edges, improved = local_search_swap(mask, edges, rng)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [[u, v] for u, v in edges]
        
    return best_edges
