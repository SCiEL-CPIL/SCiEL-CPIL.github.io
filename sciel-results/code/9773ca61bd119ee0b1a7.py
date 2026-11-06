import time
import random
import sys

def construct():
    """
    Construct a pentagon-free (C5-free) graph with no independent 9-set.
    """
    
    # Helper: Check if graph has C5
    def has_c5(n, adj_mask):
        # Iterate over all pairs (u, v) with u < v
        # Check if there is a path v-w-x-u such that u-v, v-w, w-x, x-u are edges
        # and u,v,w,x distinct.
        # This counts C5s where u-v is the "first" edge.
        # To avoid duplicates, we can just check existence.
        
        for u in range(n):
            mask_u = adj_mask[u]
            # v must be a neighbor of u
            v_mask = mask_u
            while v_mask:
                lsb_v = v_mask & -v_mask
                v = lsb_v.bit_length() - 1
                v_mask -= lsb_v
                
                # w must be a neighbor of v, not u, not v
                w_mask = adj_mask[v] & ~lsb_v & ~(1 << u)
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    w = lsb_w.bit_length() - 1
                    w_mask -= lsb_w
                    
                    # x must be a neighbor of w, not u, not v, not w
                    # and x must be a neighbor of u
                    # x in N(w) \ {u, v, w} AND x in N(u)
                    # N(w) & N(u) gives candidates
                    # Exclude v, w, u
                    x_candidates = adj_mask[w] & adj_mask[u] & ~lsb_v & ~lsb_w
                    # x cannot be u? adj_mask[u] doesn't have u.
                    # x cannot be v? Excluded.
                    # x cannot be w? Excluded.
                    # x cannot be x? adj_mask[w] doesn't have w.
                    if x_candidates:
                        return True
        return False

    # Helper: Check if graph has independent set of size k
    def has_is_k(n, adj_mask, k):
        # Use a recursive search with bitmasks
        # State: cand (bitmask of available vertices), rem (vertices needed)
        
        def dfs(cand, rem):
            if rem == 0:
                return True
            if cand == 0:
                return False
            if cand.bit_count() < rem:
                return False
            
            # Heuristic: pick vertex with highest degree in cand
            # This tends to prune the 'include' branch more effectively?
            # Actually, for IS, including a vertex removes its neighbors.
            # High degree vertex removes many candidates.
            # So branching on high degree vertex first might find a solution faster?
            # Or it might fail faster?
            # Usually, for finding *any* IS, we want to include vertices that are "easy" to place?
            # No, we want to find a set of size k.
            # Let's try the one with max degree.
            
            v = -1
            max_deg = -1
            c = cand
            while c:
                lsb = c & -c
                vi = lsb.bit_length() - 1
                deg = (adj_mask[vi] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                c -= lsb
            
            if v == -1:
                return False
                
            lsb_v = 1 << v
            
            # Option 1: Include v
            # New candidates: cand minus v and its neighbors
            new_cand_incl = cand & ~adj_mask[v] & ~lsb_v
            if dfs(new_cand_incl, rem - 1):
                return True
            
            # Option 2: Exclude v
            new_cand_excl = cand & ~lsb_v
            if dfs(new_cand_excl, rem):
                return True
            
            return False

        all_mask = (1 << n) - 1
        return dfs(all_mask, k)

    def is_valid(n, adj_mask):
        if has_c5(n, adj_mask):
            return False
        if has_is_k(n, adj_mask, 9):
            return False
        return True

    def adj_to_edges(n, adj_mask):
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges

    # Base 32-vertex graph
    def get_base_graph():
        n = 32
        adj_mask = [0] * n
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    adj_mask[base + i] |= (1 << (base + j))
                    adj_mask[base + j] |= (1 << (base + i))
        return n, adj_mask

    # Search for n
    def search_for_n(n_target, time_limit):
        start_time = time.time()
        deadline = start_time + time_limit
        
        # Initialize
        if n_target == 33:
            n, adj_mask = get_base_graph()
            adj_mask.append(0)
            # Randomly connect 32 to some vertices
            for i in range(n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << 32)
                    adj_mask[32] |= (1 << i)
            n = 33
        else:
            adj_mask = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.5:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            n = n_target

        # Simulated Annealing
        T = 1.0
        cooling = 0.995
        best_adj = adj_mask[:]
        best_cost = float('inf')
        
        def cost(adj_mask):
            c5_pen = 100 if has_c5(n, adj_mask) else 0
            is_pen = 10 if has_is_k(n, adj_mask, 9) else 0
            return c5_pen + is_pen

        current_cost = cost(adj_mask)
        
        while time.time() < deadline:
            # Flip a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            mask_u = 1 << u
            mask_v = 1 << v
            
            if adj_mask[u] & mask_v:
                adj_mask[u] &= ~mask_v
                adj_mask[v] &= ~mask_u
            else:
                adj_mask[u] |= mask_v
                adj_mask[v] |= mask_u
                
            new_cost = cost(adj_mask)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
            else:
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 0.01)):
                    current_cost = new_cost
                else:
                    if adj_mask[u] & mask_v:
                        adj_mask[u] &= ~mask_v
                        adj_mask[v] &= ~mask_u
                    else:
                        adj_mask[u] |= mask_v
                        adj_mask[v] |= mask_u
            
            T *= cooling
            
            if best_cost == 0:
                return best_adj

        return None

    # Try n=35, 34, 33
    total_time = 110.0
    time_per_n = total_time / 3
    
    for n_target in [35, 34, 33]:
        start = time.time()
        adj_mask = search_for_n(n_target, time_per_n)
        if adj_mask is not None:
            if is_valid(n_target, adj_mask):
                edges = adj_to_edges(n_target, adj_mask)
                return {'n': n_target, 'edges': edges}
        
        if time.time() - start > total_time - 10:
            break

    # Fallback to 32
    n, adj_mask = get_base_graph()
    edges = adj_to_edges(n, adj_mask)
    return {'n': n, 'edges': edges}
