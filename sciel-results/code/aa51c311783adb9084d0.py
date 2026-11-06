import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 256
    squares = set()
    k = 2
    while k*k <= 512:
        squares.add(k*k)
        k += 1

    # Exclude self-conflicting vertices (2v is a square)
    valid = []
    for v in range(1, N+1):
        if 2*v not in squares:
            valid.append(v)
    
    M = len(valid)
    
    # Build adjacency
    adj = [0] * M
    for i in range(M):
        vi = valid[i]
        mask = 0
        for j in range(M):
            if i != j:
                vj = valid[j]
                if (vi + vj) in squares:
                    mask |= (1 << j)
        adj[i] = mask

    # Sort by degree descending
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda x: -degrees[x])
    
    # Rebuild in sorted order
    adj_s = [0] * M
    for ni, oi in enumerate(order):
        m = 0
        for nj in range(M):
            if nj != ni:
                oj = order[nj]
                if adj[oi] & (1 << oj):
                    m |= (1 << nj)
        adj_s[ni] = m

    best_size = [0]
    best_mask = [0]

    def clique_cover_bound(cand_mask):
        """Upper bound on MIS of subgraph induced by cand_mask via greedy clique cover."""
        remaining = cand_mask
        count = 0
        while remaining:
            count += 1
            # Greedy: pick a vertex, build maximal clique from remaining
            # Pick vertex with max degree in remaining for better bounds
            best_v = -1
            best_deg = -1
            temp = remaining
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                deg = bin(adj_s[v] & remaining).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp &= ~(1 << v)
            
            if best_v < 0:
                break
            
            # Build clique starting from best_v
            clique = 1 << best_v
            # Candidates for adding to clique: vertices in remaining adjacent to best_v
            cand_clique = remaining & adj_s[best_v]
            while cand_clique:
                w = (cand_clique & (-cand_clique)).bit_length() - 1
                # Check w adjacent to all in clique
                if (adj_s[w] & clique) == clique:
                    clique |= (1 << w)
                    cand_clique &= ~(1 << w)
                    cand_clique &= adj_s[w]
                else:
                    cand_clique &= ~(1 << w)
            
            remaining &= ~clique
        
        return count

    def branch(cand_mask, cur_size, cur_mask):
        nonlocal best_size, best_mask
        if time.time() > deadline:
            return
        
        if cand_mask == 0:
            if cur_size > best_size[0]:
                best_size[0] = cur_size
                best_mask[0] = cur_mask
            return
        
        ub = cur_size + clique_cover_bound(cand_mask)
        if ub <= best_size[0]:
            return
        
        # Pick branch vertex: highest degree in cand subgraph
        best_v = -1
        best_deg = -1
        temp = cand_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            deg = bin(adj_s[v] & cand_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp &= ~(1 << v)
        
        if best_v < 0:
            if cur_size > best_size[0]:
                best_size[0] = cur_size
                best_mask[0] = cur_mask
            return
        
        # Include best_v
        new_cand = cand_mask & ~(1 << best_v) & ~adj_s[best_v]
        branch(new_cand, cur_size + 1, cur_mask | (1 << best_v))
        
        # Exclude best_v
        new_cand = cand_mask & ~(1 << best_v)
        branch(new_cand, cur_size, cur_mask)

    all_mask = (1 << M) - 1
    branch(all_mask, 0, 0)

    # Extract result
    result = []
    m = best_mask[0]
    while m:
        v = (m & (-m)).bit_length() - 1
        result.append(valid[order[v]])
        m &= ~(1 << v)
    
    if not result:
        result = [1]
    
    # Verify
    final = []
    for a in result:
        ok = True
        if 2*a in squares:
            ok = False
        else:
            for b in result:
                if a != b and (a+b) in squares:
                    ok = False
                    break
        if ok:
            final.append(a)
    
    if not final:
        final = [1]
    
    return final
