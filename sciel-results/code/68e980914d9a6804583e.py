import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # cn[u][v] = number of common neighbors of u and v
    # adj[u] = bitmask of neighbors
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    def can_add_fast(adj, cn, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        # Common neighbors of u and v must be <= 2
        if cn[u][v] > 2:
            return False
        # For each neighbor x of u: cn[v][x] must be <= 1
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            x = lsb.bit_length() - 1
            if cn[v][x] > 1:
                return False
            mask ^= lsb
        # For each neighbor y of v: cn[u][y] must be <= 1
        mask = adj[v]
        while mask:
            lsb = mask & -mask
            y = lsb.bit_length() - 1
            if cn[u][y] > 1:
                return False
            mask ^= lsb
        return True
    
    def add_edge_fast(adj, cn, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # Update common neighbor counts
        # For all w: cn[u][w] and cn[v][w] may change
        # Actually, adding edge (u,v) means:
        # - u and v now share each other as neighbor? No, cn counts common neighbors.
        # - For any w that is neighbor of both u and v, cn[u][v] increases? No.
        # Let me think again.
        # cn[a][b] = |N(a) ∩ N(b)|
        # Adding edge (u,v): N(u) gains v, N(v) gains u.
        # For pair (u, w) where w != v: N(u) gains v, so if v was in N(w), cn[u][w] increases by 1.
        #   i.e., if w is neighbor of v (other than u), then cn[u][w] += 1
        # For pair (v, w) where w != u: N(v) gains u, so if u was in N(w), cn[v][w] += 1
        #   i.e., if w is neighbor of u (other than v), then cn[v][w] += 1
        # For pair (u, v): N(u) gains v, N(v) gains u. 
        #   cn[u][v] = |N(u) ∩ N(v)|. Before: N(u) had some set, N(v) had some set.
        #   After: N(u) ∪ {v}, N(v) ∪ {u}. 
        #   New intersection = (old N(u) ∩ old N(v)) ∪ (old N(u) ∩ {u}) ∪ ({v} ∩ old N(v)) ∪ ({v} ∩ {u})
        #   = old intersection ∪ (if u in N(u)? no) ∪ (if v in N(v)? no) ∪ (if u==v? no)
        #   Wait, u is not in N(u) (no self-loops), v is not in N(v).
        #   So cn[u][v] unchanged? No wait:
        #   N(u) new = N(u) old ∪ {v}
        #   N(v) new = N(v) old ∪ {u}
        #   Intersection = (N(u)old ∪ {v}) ∩ (N(v)old ∪ {u})
        #   = (N(u)old ∩ N(v)old) ∪ (N(u)old ∩ {u}) ∪ ({v} ∩ N(v)old) ∪ ({v} ∩ {u})
        #   = (N(u)old ∩ N(v)old) ∪ ∅ ∪ (∅ if v not in N(v)old) ∪ ∅
        #   So cn[u][v] stays the same.
        # 
        # For pair (u, w), w != v:
        #   N(u) new = N(u)old ∪ {v}
        #   N(w) unchanged
        #   cn[u][w] new = |N(u)old ∪ {v} ∩ N(w)| = cn[u][w]old + (1 if v in N(w) else 0)
        #   v in N(w) means w is neighbor of v, i.e., w in N(v)old (since v's neighbors don't change except gaining u)
        #   Wait, N(v) new = N(v)old ∪ {u}, but for checking if v is in N(w), we look at N(w), not N(v).
        #   v is in N(w) iff w is in N(v). N(v) new includes u, but we're checking if v is neighbor of w.
        #   v is neighbor of w iff w is neighbor of v. After adding (u,v), N(v) = N(v)old ∪ {u}.
        #   So v is in N(w) iff w in N(v)old ∪ {u}. Since w != v and w != u (we're iterating w != v, and w is a neighbor of u so w != u... actually w could be u? No, w is neighbor of u, so w != u).
        #   So v in N(w) iff w in N(v)old.
        #   Therefore cn[u][w] += 1 if w in N(v)old, for w != v.
        #
        # Similarly for cn[v][w], w != u: cn[v][w] += 1 if w in N(u)old.
        
        # Update cn for pairs involving u
        mask_v_old = adj[v] & ~(1 << u)  # N(v) old (v doesn't have self, and we just added u to adj[v])
        # Actually adj[v] already has u now. Let me be careful.
        # Before update: adj[v] has u just added. So N(v) current = N(v)old ∪ {u}.
        # For w in N(v)old (i.e., w in adj[v] before we added u, but adj[v] now has u too):
        # We need w such that w is neighbor of v. Since we just added u to adj[v], adj[v] now includes u.
        # N(v)old = adj[v] & ~(1 << u) ... but wait, did adj[v] have u before? No, we just added it.
        # So N(v)old = adj[v] & ~(1 << u) is correct if we consider that u was just added.
        # Actually simpler: iterate over current adj[v] which includes u and all old neighbors.
        # For each w in adj[v]: if w != u, then w was in N(v)old, so cn[u][w] += 1 and cn[w][u] += 1
        mask = adj[v]
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            if w != u:
                cn[u][w] += 1
                cn[w][u] += 1
            mask ^= lsb
        
        # For each w in adj[u]: if w != v, then w was in N(u)old, so cn[v][w] += 1 and cn[w][v] += 1
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            if w != v:
                cn[v][w] += 1
                cn[w][v] += 1
            mask ^= lsb
    
    def remove_edge_fast(adj, cn, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Reverse of add_edge_fast
        # For each w in adj[v] (after removal, so N(v)old without u... wait)
        # After removal, adj[v] no longer has u. 
        # Before removal, N(v) = N(v)after ∪ {u}.
        # cn[u][w] was incremented by 1 if w in N(v)before. N(v)before = adj[v]after ∪ {u}.
        # But w != u in our iteration, and w in N(v)before means w in adj[v]after or w == u.
        # Since w != u, w in N(v)before iff w in adj[v]after.
        # So for each w in adj[v]after (which is current adj[v]): cn[u][w] -= 1
        mask = adj[v]
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            cn[u][w] -= 1
            cn[w][u] -= 1
            mask ^= lsb
        
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            cn[v][w] -= 1
            cn[w][v] -= 1
            mask ^= lsb
    
    def count_edges(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    def get_edges_list(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges
    
    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    missing.append((i, j))
        return missing
    
    def get_existing_edges(adj):
        existing = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    existing.append((i, j))
                mask ^= lsb
        return existing
    
    def greedy_fill(adj, cn, max_iter=50):
        """Repeatedly add any valid edge until none remain."""
        for _ in range(max_iter):
            added = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if can_add_fast(adj, cn, u, v):
                    add_edge_fast(adj, cn, u, v)
                    added = True
            if not added:
                break
    
    def local_search(adj, cn, time_limit):
        """Edge-swap local search to escape local optima."""
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Try greedy fill first
            greedy_fill(adj, cn, 5)
            
            existing = get_existing_edges(adj)
            missing = get_missing_edges(adj)
            if not existing or not missing:
                break
            
            # Try random edge swaps
            improved = False
            for _ in range(20):
                if time.time() >= end_time:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                
                # Try removing (u1,v1) and adding (u2,v2)
                remove_edge_fast(adj, cn, u1, v1)
                if can_add_fast(adj, cn, u2, v2):
                    add_edge_fast(adj, cn, u2, v2)
                    # Greedy fill to take advantage
                    greedy_fill(adj, cn, 3)
                    improved = True
                else:
                    # Revert
                    add_edge_fast(adj, cn, u1, v1)
            
            if not improved:
                # Try single edge removal to break symmetry
                u1, v1 = random.choice(existing)
                remove_edge_fast(adj, cn, u1, v1)
                greedy_fill(adj, cn, 5)
                
                # Try to re-add
                if can_add_fast(adj, cn, u1, v1):
                    add_edge_fast(adj, cn, u1, v1)
    
    best_count = 0
    best_adj = None
    best_cn = None
    
    # Multiple restarts with different initializations
    for restart in range(50):
        if time.time() >= deadline:
            break
        
        adj = [0] * n
        cn = [[0] * n for _ in range(n)]
        
        # Random initialization: try to build up with random edge ordering
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_fast(adj, cn, u, v):
                add_edge_fast(adj, cn, u, v)
        
        # Greedy fill
        greedy_fill(adj, cn, 20)
        
        # Local search with edge swaps
        remaining_time = deadline - time.time()
        ls_time = min(1.5, remaining_time * 0.5)
        if ls_time > 0.1:
            local_search(adj, cn, ls_time)
        
        # Final greedy polish
        greedy_fill(adj, cn, 10)
        
        cur_count = count_edges(adj)
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]
            best_cn = [row[:] for row in cn]
        
        # If we've found a good solution, reduce time per restart
        if best_count >= 150:
            # Tighten deadline for subsequent restarts
            pass
    
    # Use best solution
    if best_adj is None:
        # Fallback: empty graph
        return []
    
    return get_edges_list(best_adj)
