import time
import random
import sys

def construct():
    """
    Construct a pentagon-free (C5-free) graph with no independent 9-set.
    """
    
    # Helper: Check if graph has C5
    # adj_mask: list of int, adj_mask[i] is bitmask of neighbors of i
    def has_c5(n, adj_mask):
        # For each edge (u, v), check for path u-v-w-x-y-u
        # u, v, w, x, y distinct.
        # u-v is edge.
        # v-w is edge.
        # w-x is edge.
        # x-y is edge.
        # y-u is edge.
        
        # Iterate u
        for u in range(n):
            # Iterate v in N(u)
            v_mask = adj_mask[u]
            while v_mask:
                lsb_v = v_mask & -v_mask
                v = lsb_v.bit_length() - 1
                v_mask -= lsb_v
                
                # Iterate w in N(v) \ {u}
                w_mask = adj_mask[v] & ~lsb_v & ~(1 << u)
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    w = lsb_w.bit_length() - 1
                    w_mask -= lsb_w
                    
                    # Iterate x in N(w) \ {u, v}
                    x_mask = adj_mask[w] & ~lsb_w & ~lsb_v & ~(1 << u)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        x = lsb_x.bit_length() - 1
                        x_mask -= lsb_x
                        
                        # Check if there exists y in N(x) \ {u, v, w} such that y in N(u)
                        # y must be in N(x) and N(u)
                        # and y != u, v, w
                        # N(x) & N(u) gives candidates for y
                        # We must exclude v and w (and u, but u is not in N(u))
                        # Also y must not be x (simple graph)
                        y_candidates = adj_mask[x] & adj_mask[u] & ~lsb_v & ~lsb_w
                        # y cannot be u? adj_mask[u] doesn't contain u.
                        # y cannot be v? Excluded.
                        # y cannot be w? Excluded.
                        # y cannot be x? adj_mask[x] doesn't contain x.
                        # So if y_candidates is not 0, we found a C5.
                        if y_candidates:
                            return True
        return False

    # Helper: Check if graph has independent set of size k
    def has_is_k(n, adj_mask, k):
        # candidates: bitmask of vertices that can still be added
        # rem: number of vertices needed
        def dfs(cand, rem):
            if rem == 0:
                return True
            if cand == 0:
                return False
            if cand.bit_count() < rem:
                return False
            
            # Heuristic: pick vertex with max degree in cand?
            # Or just first bit?
            # Let's pick the one with max degree to prune 'include' branch more?
            # Actually, for IS, including a high-degree vertex removes many candidates.
            # Excluding it leaves many.
            # Branching on high degree might be better.
            
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
            # Remove v and its neighbors
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
        
        # Initialize with random graph
        # Or start with base graph if n_target == 33
        if n_target == 33:
            n, adj_mask = get_base_graph()
            # Add vertex 32
            adj_mask.append(0)
            # Randomly connect 32 to some vertices
            for i in range(n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << 32)
                    adj_mask[32] |= (1 << i)
            n = 33
        else:
            # Random graph
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
        
        # Cost function
        def cost(adj_mask):
            # Penalty for C5
            # We don't want to count all C5s, just check existence?
            # For SA, we need a smooth-ish cost.
            # Counting C5s is expensive.
            # Let's use: 100 if has_c5 else 0.
            # And 10 if has_is_9 else 0.
            # This is very rough.
            
            # Better: 
            # c5_count is hard to get fast.
            # Let's just use the boolean checks.
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
            
            # Toggle edge (u, v)
            mask_u = 1 << u
            mask_v = 1 << v
            
            if adj_mask[u] & mask_v:
                # Remove edge
                adj_mask[u] &= ~mask_v
                adj_mask[v] &= ~mask_u
            else:
                # Add edge
                adj_mask[u] |= mask_v
                adj_mask[v] |= mask_u
                
            new_cost = cost(adj_mask)
            
            # Accept?
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
            else:
                # Metropolis criterion
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 0.01)):
                    current_cost = new_cost
                else:
                    # Revert
                    if adj_mask[u] & mask_v:
                        adj_mask[u] &= ~mask_v
                        adj_mask[v] &= ~mask_u
                    else:
                        adj_mask[u] |= mask_v
                        adj_mask[v] |= mask_u
            
            T *= cooling
            
            # Check if we found a valid graph
            if best_cost == 0:
                return best_adj

        return None

    # Try n=35, 34, 33
    # Allocate time
    total_time = 110.0
    time_per_n = total_time / 3
    
    for n_target in [35, 34, 33]:
        start = time.time()
        adj_mask = search_for_n(n_target, time_per_n)
        if adj_mask is not None:
            # Verify
            if is_valid(n_target, adj_mask):
                edges = adj_to_edges(n_target, adj_mask)
                return {'n': n_target, 'edges': edges}
        
        # If time is running out, break
        if time.time() - start > total_time - 10:
            break

    # Fallback to 32
    n, adj_mask = get_base_graph()
    edges = adj_to_edges(n, adj_mask)
    return {'n': n, 'edges': edges}
