"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_START_TIME = time.time()
_HARD_DEADLINE = 145.0  # 5s margin for 150s limit

def _time_left():
    return _HARD_DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _build_adj(colors, n):
    """Build adjacency bitsets for a given coloring."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    """Count total monochromatic triangles using bitsets."""
    count = 0
    for c in range(4):
        for i in range(n):
            # Only count triangles where i < j < k to avoid overcounting
            # adj[c][i] has bits for all neighbors.
            # We want neighbors j > i.
            mask = ~((1 << (i + 1)) - 1)
            neigh = adj[c][i] & mask
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                # Count common neighbors k > j
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _find_one_mono_triangle(adj, n):
    """Find one monochromatic triangle, return (i, j, k, c) or None."""
    for c in range(4):
        for i in range(n):
            mask = ~((1 << (i + 1)) - 1)
            neigh = adj[c][i] & mask
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                if common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    return (i, j, k, c)
    return None

def _greedy_init(n):
    """
    Greedy initialization: build vertex by vertex, choosing edge colors
    to minimize immediate monochromatic triangle creation.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                # Cost is number of triangles formed if we color (u,v) with c
                # Triangles are (u, v, w) where w < u.
                # We need common neighbors of u and v in color c among {0..u-1}
                # But adj[c][v] is empty for v > u initially? No, we build v sequentially.
                # adj[c][u] contains neighbors < u.
                # adj[c][v] is empty for now (v is new).
                # So we are forming triangles (w, u, v) where w is a common neighbor?
                # No, (w, u) and (w, v) must be color c.
                # (w, v) is being set now? No, we are setting (u, v).
                # The triangle is (w, u, v). Edges (w, u) and (w, v) and (u, v).
                # We are setting (u, v) to c.
                # We need (w, u) to be c AND (w, v) to be c.
                # But (w, v) is not set yet?
                # In greedy construction for v, we set edges (0, v), (1, v), ..., (v-1, v).
                # When setting (u, v), edges (w, v) for w < u are ALREADY SET.
                # So we check: for each w < u, is color(w, v) == c AND color(w, u) == c?
                
                # Optimization: 
                # Let S_c be the set of w < u such that color(w, u) == c.
                # We need to count how many w in S_c have color(w, v) == c.
                
                # This is complex to do online without storing full history efficiently.
                # Simpler greedy: Random with slight bias?
                # Or just count common neighbors in current partial graph.
                # For (u, v), triangles involving a new vertex w? No, w < u.
                # Triangle (w, u, v).
                # We need color(w, u) == c and color(w, v) == c.
                
                cost = 0
                # Iterate w < u
                # This is O(u) per edge, O(n^3) total. For n=50, 125k ops. Fast.
                for w in range(u):
                    # Get color of (w, u) and (w, v)
                    # Index for (w, u): w*(n-1) - w*(w+1)//2 + (u-w-1)
                    # Index for (w, v): w*(n-1) - w*(w+1)//2 + (v-w-1)
                    idx_wu = w * (n - 1) - w * (w + 1) // 2 + (u - w - 1)
                    idx_wv = w * (n - 1) - w * (w + 1) // 2 + (v - w - 1)
                    
                    if colors[idx_wu] == c and colors[idx_wv] == c:
                        cost += 1
                        
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            
            idx_uv = u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
            colors[idx_uv] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
    
    return colors

def _sa_search(n, time_budget):
    """
    Simulated annealing to find a valid 4-coloring of K_n.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Precompute index-to-edge mapping
    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1
            
    # Precompute edge index from (i, j)
    # We need fast access for delta calculation.
    # We store colors in a flat list.
    
    best_result = None
    best_tri = float('inf')
    
    # Try a few restarts
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize
        # Use greedy init for better starting point
        colors = _greedy_init(n)
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        # SA Parameters
        T = 10.0
        T_min = 0.001
        cooling = 0.999
        accepted = 0
        total_steps = 0
        
        while True:
            if time.time() - start_time > time_budget:
                break
            
            # Pick random edge
            ri = random.randint(0, m - 1)
            old_c = colors[ri]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            u, v = idx_to_edge[ri]
            
            # Delta calculation
            # Remove edge from old_c, add to new_c
            # Triangles removed: common neighbors of u, v in old_c
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            # Triangles added: common neighbors of u, v in new_c
            added = (adj[new_c][u] & adj[new_c][v]).bit_count()
            delta = added - removed
            
            total_steps += 1
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                if T > T_min:
                    # Use log to avoid underflow? delta is small.
                    # prob = exp(-delta/T)
                    if random.random() < math.exp(-delta / T):
                        accept = True
            
            if accept:
                # Update state
                colors[ri] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                tri_count += delta
                accepted += 1
                
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_result = colors[:]
                    if tri_count == 0:
                        return colors
            
            T *= cooling
            
            # Restart if stuck
            if total_steps > 2000 and accepted < total_steps * 0.01:
                break
                
    return best_result if best_tri == 0 else None

def construct():
    # Try to find a coloring for n=50, then 49, ..., down to 32
    # Time budget: 140s
    total_budget = 140.0
    start = time.time()
    
    # Start with the known good 32
    best_n = 32
    best_colors = _seed_32()
    
    # Try larger n
    for n in range(50, 32, -1):
        # Allocate time proportional to difficulty?
        # Just give a fixed chunk or remaining time?
        # Let's give 10 seconds per n attempt, or more if time permits.
        # Since 50 is the target, give it most time.
        time_left = _time_left()
        if time_left < 5:
            break
            
        # Allocate time: 50% of remaining for 50, less for others?
        # Simple: 15s for 50, 5s for others.
        budget = 15.0 if n == 50 else 5.0
        if budget > time_left:
            budget = time_left - 1.0
            
        if budget <= 0:
            break
            
        result = _sa_search(n, budget)
        
        if result is not None:
            # Found a valid coloring for n
            best_n = n
            best_colors = result
            # If we found 50, we can return immediately as it's the record
            if n == 50:
                break
        else:
            # Failed to find valid coloring for n
            # Continue to smaller n?
            # If we fail 50, 49 is likely hard too.
            # But maybe 48 is easier?
            # We will just continue the loop.
            pass
            
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
