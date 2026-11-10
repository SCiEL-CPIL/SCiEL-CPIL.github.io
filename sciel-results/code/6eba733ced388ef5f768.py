import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140
    
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute triples for faster checking?
    # We only need triples involving u or v.
    # We can generate them on the fly.
    
    def is_valid_add(masks, u, v):
        # Tentatively add
        mu_orig = masks[u]
        mv_orig = masks[v]
        masks[u] = mu_orig | (1 << v)
        masks[v] = mv_orig | (1 << u)
        
        # Check triples containing u
        for a in range(n):
            if a == u: continue
            for b in range(a + 1, n):
                if b == u: continue
                # Triple (u, a, b)
                inter = masks[u] & masks[a] & masks[b]
                if inter.bit_count() > 2:
                    masks[u] = mu_orig
                    masks[v] = mv_orig
                    return False
        # Check triples containing v (but not u, to avoid double checking)
        for a in range(n):
            if a == v or a == u: continue
            for b in range(a + 1, n):
                if b == v or b == u: continue
                # Triple (v, a, b)
                inter = masks[v] & masks[a] & masks[b]
                if inter.bit_count() > 2:
                    masks[u] = mu_orig
                    masks[v] = mv_orig
                    return False
                    # Note: triples containing both u and v were checked in the first loop?
                    # First loop: a != u, b != u. Triple (u, a, b).
                    # If a=v, b=w. Triple (u, v, w). Checked.
                    # Second loop: a != v, a != u. b != v, b != u.
                    # So triples with both u and v are only in first loop.
                    # This is correct.
        
        return True

    best_edges_count = 0
    best_masks = [0] * n

    while time.time() < deadline:
        masks = [0] * n
        random.shuffle(edges)
        current_count = 0
        
        for u, v in edges:
            if is_valid_add(masks, u, v):
                current_count += 1
        
        if current_count > best_edges_count:
            best_edges_count = current_count
            best_masks = masks[:]

    result = []
    for u in range(n):
        m = best_masks[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                result.append([u, v])
            m ^= lsb
    return result
