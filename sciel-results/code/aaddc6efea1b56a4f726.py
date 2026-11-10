import time
import numpy as np
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin
    
    # Precompute all triples
    triples = list(combinations(range(n), 3))
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_neighbors(adj_masks, v):
        """Get list of neighbors of vertex v."""
        mask = adj_masks[v]
        nbrs = []
        for x in range(n):
            if x != v and (mask >> x) & 1:
                nbrs.append(x)
        return nbrs

    def init_pair_common(adj_masks):
        """Initialize pair_common[a][b] = N(a) & N(b)."""
        pair_common = [[0]*n for _ in range(n)]
        for a in range(n):
            for b in range(a+1, n):
                common = adj_masks[a] & adj_masks[b]
                pair_common[a][b] = common
                pair_common[b][a] = common
        return pair_common

    def update_pair_common_add(pair_common, adj_masks, u, v):
        """Update pair_common after adding edge (u,v)."""
        # pair_common[u][v] and pair_common[v][u] remain unchanged (as derived)
        # But we must ensure they are correct if they were 0 before?
        # N(u) & N(v) doesn't change because u not in N(u) and v not in N(v).
        
        for w in range(n):
            if w == u or w == v:
                continue
            # If v in N(w), then v is in N(u) & N(w) (since v in N(u) now)
            if (adj_masks[w] >> v) & 1:
                pair_common[u][w] |= (1 << v)
                pair_common[w][u] |= (1 << v)
            # If u in N(w), then u is in N(v) & N(w) (since u in N(v) now)
            if (adj_masks[w] >> u) & 1:
                pair_common[v][w] |= (1 << u)
                pair_common[w][v] |= (1 << u)

    def update_pair_common_remove(pair_common, adj_masks, u, v):
        """Update pair_common after removing edge (u,v)."""
        # adj_masks already updated (u,v removed)
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj_masks[w] >> v) & 1:
                pair_common[u][w] &= ~(1 << v)
                pair_common[w][u] &= ~(1 << v)
            if (adj_masks[w] >> u) & 1:
                pair_common[v][w] &= ~(1 << u)
                pair_common[w][v] &= ~(1 << u)

    def can_add(adj_masks, pair_common, u, v):
        """
        Check if adding edge (u,v) creates a K3,3.
        """
        # Case A: u in A, v in B. Pairs (a,b) in N(v) \ {u}.
        # Need |N(u) & N(a) & N(b)| >= 2.
        # N(a) & N(b) is pair_common[a][b].
        mask_u = adj_masks[u]
        mask_v = adj_masks[v]
        
        # Iterate over pairs in N(v) \ {u}
        m = mask_v & ~(1 << u)
        nbrs_v = []
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nbrs_v.append(idx)
            m ^= lsb
        
        for i in range(len(nbrs_v)):
            a = nbrs_v[i]
            for j in range(i + 1, len(nbrs_v)):
                b = nbrs_v[j]
                common = mask_u & pair_common[a][b]
                if common.bit_count() >= 2:
                    return False
        
        # Case B: v in A, u in B. Pairs (a,b) in N(u) \ {v}.
        m = mask_u & ~(1 << v)
        nbrs_u = []
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nbrs_u.append(idx)
            m ^= lsb
            
        for i in range(len(nbrs_u)):
            a = nbrs_u[i]
            for j in range(i + 1, len(nbrs_u)):
                b = nbrs_u[j]
                common = mask_v & pair_common[a][b]
                if common.bit_count() >= 2:
                    return False
        
        return True

    def is_valid(adj_masks):
        """Check if graph is K3,3-free."""
        for i, j, k in triples:
            common = adj_masks[i] & adj_masks[j] & adj_masks[k]
            if common.bit_count() >= 3:
                return False
        return True

    def clean_graph(adj_masks, pair_common, rng):
        """Remove edges until graph is K3,3-free."""
        removed = 0
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            found = False
            for i, j, k in triples:
                common = adj_masks[i] & adj_masks[j] & adj_masks[k]
                if common.bit_count() >= 3:
                    nbrs = []
                    m = common
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        nbrs.append(idx)
                        m ^= lsb
                    
                    part_a = [i, j, k]
                    for _try in range(30):
                        a = part_a[rng.integers(0, 3)]
                        b = nbrs[rng.integers(0, len(nbrs))]
                        if (adj_masks[a] >> b) & 1:
                            adj_masks[a] &= ~(1 << b)
                            adj_masks[b] &= ~(1 << a)
                            update_pair_common_remove(pair_common, adj_masks, a, b)
                            removed += 1
                            break
                    else:
                        a = i
                        b = nbrs[0]
                        adj_masks[a] &= ~(1 << b)
                        adj_masks[b] &= ~(1 << a)
                        update_pair_common_remove(pair_common, adj_masks, a, b)
                        removed += 1
                    found = True
                    break
            if not found:
                break
        return removed

    def greedy_add(adj_masks, pair_common, rng, deadline_check=True):
        """Greedily add edges in random order."""
        added = 0
        missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                   if not ((adj_masks[i] >> j) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if deadline_check and time.time() > deadline - 3:
                break
            if can_add(adj_masks, pair_common, u, v):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                update_pair_common_add(pair_common, adj_masks, u, v)
                added += 1
        return added

    def local_search(adj_masks, pair_common, rng, max_rounds=30):
        """
        Local search: try 1-for-1 and 1-for-2 edge swaps.
        """
        improvements = 0
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            round_improved = False
            edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                    if (adj_masks[i] >> j) & 1]
            rng.shuffle(edges)
            
            for u, v in edges:
                if time.time() > deadline - 5:
                    break
                
                # Remove edge (u,v)
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                update_pair_common_remove(pair_common, adj_masks, u, v)
                
                # Try to add up to 2 edges from missing
                missing = [(i, j) for i in range(n) for j in range(i+1, n)
                          if not ((adj_masks[i] >> j) & 1)]
                rng.shuffle(missing)
                
                added_edges = []
                for a, b in missing:
                    if len(added_edges) >= 2:
                        break
                    if can_add(adj_masks, pair_common, a, b):
                        adj_masks[a] |= (1 << b)
                        adj_masks[b] |= (1 << a)
                        update_pair_common_add(pair_common, adj_masks, a, b)
                        added_edges.append((a, b))
                
                if len(added_edges) >= 1:
                    improvements += len(added_edges) - 1
                    round_improved = True
                else:
                    # Restore original edge
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    update_pair_common_add(pair_common, adj_masks, u, v)
            
            if not round_improved:
                break
        
        return improvements

    def edge_count(adj_masks):
        return sum((adj_masks[i] >> j) & 1 for i in range(n) for j in range(i+1, n))
    
    best_adj = [0] * n
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    num_restarts = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        p = 0.3 + 0.4 * rng.random()
        adj_masks = [0] * n
        
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        
        pair_common = init_pair_common(adj_masks)
        
        clean_graph(adj_masks, pair_common, rng)
        greedy_add(adj_masks, pair_common, rng)
        local_search(adj_masks, pair_common, rng, max_rounds=25)
        greedy_add(adj_masks, pair_common, rng)
        local_search(adj_masks, pair_common, rng, max_rounds=15)
        
        cur_count = edge_count(adj_masks)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj_masks[:]
            
            if best_count >= 58:
                if time.time() > deadline - 30:
                    break
    
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                best_edges.append([i, j])
    
    return best_edges
