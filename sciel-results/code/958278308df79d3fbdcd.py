import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # First, try to extend the known n=32 solution (8x K4)
    # by adding 1-3 vertices
    for extra in range(3, 0, -1):
        if time.time() > deadline - 30:
            break
        n = 32 + extra
        result = try_extend_known(n, deadline)
        if result is not None:
            return result
    
    # Try random search for n=35, 34, 33
    for n in [35, 34, 33]:
        if time.time() > deadline - 20:
            break
        
        # Multiple restarts with different initial densities
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            adj = [0] * n
            p = 0.4 + 0.1 * restart  # Vary density
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search with combined objective
            adj = local_search(n, adj, deadline)
            
            # Check if valid
            if count_c5_fast(n, adj) == 0:
                alpha = calc_alpha_fast(n, adj)
                if alpha < 9:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    # Fallback to known good solution (n=32)
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}

def count_c5_fast(n, adj):
    """Fast C5 detection with early exit. Returns count capped at 10."""
    count = 0
    for u in range(n):
        nu = adj[u]
        # Iterate over neighbors v of u where v > u
        v_mask = nu & ((1 << n) - 1)
        v_mask &= ~((1 << (u + 1)) - 1)  # Only v > u
        while v_mask:
            lsb_v = v_mask & -v_mask
            v = lsb_v.bit_length() - 1
            v_mask ^= lsb_v
            
            # u-v is an edge
            # Find a in N(u) \ {v}, c in N(v) \ {u}
            nu_excl = nu & ~(1 << v)
            nv_excl = adj[v] & ~(1 << u)
            
            # For efficiency, iterate over the smaller set
            if popcount(nu_excl) < popcount(nv_excl):
                a_mask = nu_excl
                c_mask = nv_excl
            else:
                a_mask = nv_excl
                c_mask = nu_excl
            
            a_mask_iter = a_mask
            while a_mask_iter:
                lsb_a = a_mask_iter & -a_mask_iter
                a = lsb_a.bit_length() - 1
                a_mask_iter ^= lsb_a
                
                c_mask_iter = c_mask
                while c_mask_iter:
                    lsb_c = c_mask_iter & -c_mask_iter
                    c = lsb_c.bit_length() - 1
                    c_mask_iter ^= lsb_c
                    
                    if a == c:
                        continue
                    
                    # Check if a and c have a common neighbor b
                    # b must not be in {u, v, a, c}
                    common = adj[a] & adj[c]
                    exclude = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                    if common & ~exclude:
                        count += 1
                        if count >= 10:
                            return count
    return count

def popcount(x):
    """Count set bits in integer."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

def local_search(n, adj, deadline):
    """Local search to minimize C5 count."""
    best_adj = adj[:]
    best_cost = count_c5_fast(n, adj)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while T > T_min and time.time() < deadline:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        curr_cost = best_cost
        
        # Flip edge
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        new_cost = count_c5_fast(n, adj)
        
        # Accept if better or with probability
        if new_cost < curr_cost or (T > 0 and random.random() < np.exp(min(50, (curr_cost - new_cost) / T))):
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = adj[:]
            if best_cost == 0:
                break
        else:
            # Revert
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        T *= cooling
    
    return best_adj

def try_extend_known(n, deadline):
    """Try to extend the known 8xK4 solution by adding vertices."""
    # Start with 8xK4 on vertices 0-31
    adj = [0] * n
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i] |= (1 << (base + j))
                adj[base + j] |= (1 << (base + i))
    
    # New vertices are 32..n-1
    # Try to add edges from new vertices to old ones
    # Strategy: each new vertex should connect to vertices in a way that
    # doesn't create C5 and keeps alpha < 9
    
    # For each new vertex, try different connection patterns
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        # Randomly decide connections for new vertices
        adj_copy = adj[:]
        for new_v in range(32, n):
            # Connect to some subset of old vertices
            # To avoid C5, be conservative
            for old_v in range(32):
                if random.random() < 0.3:
                    adj_copy[new_v] |= (1 << old_v)
                    adj_copy[old_v] |= (1 << new_v)
        
        # Also try adding edges between new vertices
        for i in range(32, n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_copy[i] |= (1 << j)
                    adj_copy[j] |= (1 << i)
        
        # Check if valid
        if count_c5_fast(n, adj_copy) == 0:
            alpha = calc_alpha_fast(n, adj_copy)
            if alpha < 9:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj_copy[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    return None

def calc_alpha_fast(n, adj):
    """Calculate independence number using branch and bound with bitsets."""
    # Independence number = max clique in complement
    # Use bitset representation for speed
    
    best = [0]
    
    # Complement adjacency: comp_adj[i] has bit j set if i and j are NOT adjacent in original
    comp_adj = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = all_mask & ~(adj[i] | (1 << i))
    
    def dfs(cand_mask, size):
        if size + popcount(cand_mask) <= best[0]:
            return
        if cand_mask == 0:
            if size > best[0]:
                best[0] = size
            return
        
        # Pick vertex with highest degree in complement induced by cand
        # For speed, just pick the lowest set bit
        lsb = cand_mask & -cand_mask
        v = lsb.bit_length() - 1
        rest_mask = cand_mask & ~lsb
        
        # Option 1: Include v
        # New candidates: neighbors of v in complement, intersected with rest
        new_cand = comp_adj[v] & rest_mask
        dfs(new_cand, size + 1)
        
        # Option 2: Exclude v
        dfs(rest_mask, size)
    
    dfs((1 << n) - 1, 0)
    return best[0]

# EVOLVE-BLOCK-END
