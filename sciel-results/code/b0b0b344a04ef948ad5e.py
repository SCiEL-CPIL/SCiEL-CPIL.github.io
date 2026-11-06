import time
import random
import math
import sys

_START_TIME = time.time()
_DEADLINE = 135.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

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
    return {'n': 32, 'colors': colors}

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

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
            # Only consider neighbors j > i to avoid double counting and self-loops
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
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
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
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
                # Count triangles formed if we add edge (u,v) with color c
                # This is the number of common neighbors w < u of u and v in color c
                # Actually, the cost is the number of edges (w,u) and (w,v) that are color c
                # which form a triangle (w,u,v) if (u,v) is c.
                # w must be < u < v.
                cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
    
    return colors

def _sa_search(n, time_budget):
    """
    Simulated annealing to find a valid 4-coloring of K_n.
    Uses the structural idea from the Inspiration: robust SA with acceptance rate restarts.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Precompute index-to-edge mapping for speed
    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1
    
    best_result = None
    best_tri = float('inf')
    
    # Multiple restarts with different initializations
    # The Inspiration used greedy init primarily. We will mix greedy and random.
    for restart in range(1000): # High limit, controlled by time
        if time.time() - start_time > time_budget:
            break
        
        # Alternate initialization strategies
        if restart % 2 == 0:
            colors = _greedy_init(n)
        else:
            colors = [random.randint(0, 3) for _ in range(m)]
        
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        # Simulated Annealing Parameters (from Inspiration)
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        accepted = 0
        total_steps = 0
        steps_since_improve = 0
        
        while True:
            if time.time() - start_time > time_budget:
                break
            
            # Pick a random edge
            ri = random.randint(0, m - 1)
            old_c = colors[ri]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            u, v = idx_to_edge[ri]
            
            # Compute delta in triangle count
            # Removed triangles: common neighbors of u,v in color old_c
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            # Added triangles: common neighbors of u,v in color new_c
            added = (adj[new_c][u] & adj[new_c][v]).bit_count()
            delta = added - removed
            
            total_steps += 1
            
            if delta <= 0:
                # Accept if improvement or equal
                colors[ri] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                tri_count += delta
                accepted += 1
                steps_since_improve = 0
                
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_result = colors[:]
                    if tri_count == 0:
                        return colors
            else:
                # Accept with probability exp(-delta/T)
                if T > T_min:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        colors[ri] = new_c
                        adj[old_c][u] &= ~(1 << v)
                        adj[old_c][v] &= ~(1 << u)
                        adj[new_c][u] |= (1 << v)
                        adj[new_c][v] |= (1 << u)
                        tri_count += delta
                        accepted += 1
                        steps_since_improve = 0
                else:
                    steps_since_improve += 1
            
            T *= cooling
            
            # Restart condition: if we've taken many steps but very few were accepted
            # This indicates we are stuck in a local minimum with high energy barriers
            if total_steps > 1000 and accepted < total_steps * 0.01:
                break
            
            # Also restart if stuck for too long without improvement
            if steps_since_improve > 5000:
                break
                
    return best_result if best_tri == 0 else None

def _local_search(n, time_budget):
    """
    Local search: iteratively recolor edges in monochromatic triangles
    to reduce triangle count.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    # Precompute index-to-edge
    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1
            
    best_result = None
    best_tri = float('inf')
    
    for restart in range(100):
        if time.time() - start_time > time_budget:
            break
        
        # Start from greedy
        colors = _greedy_init(n)
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        no_improve = 0
        max_no_improve = 100
        
        while no_improve < max_no_improve:
            if time.time() - start_time > time_budget:
                break
            
            tri = _find_one_mono_triangle(adj, n)
            if tri is None:
                return colors
            
            i, j, k, c = tri
            
            # Try all 3 edges and 3 new colors each
            edges = [(i, j), (i, k), (j, k)]
            best_delta = float('inf')
            best_edge = None
            best_new_c = -1
            
            for u, v in edges:
                for new_c in range(4):
                    if new_c == c:
                        continue
                    removed = (adj[c][u] & adj[c][v]).bit_count()
                    added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                    delta = added - removed
                    if delta < best_delta:
                        best_delta = delta
                        best_edge = (u, v)
                        best_new_c = new_c
            
            if best_edge is None or best_delta >= 0:
                no_improve += 1
                # If stuck, try a random move to escape
                if no_improve > 10:
                    ri = random.randint(0, m - 1)
                    u, v = idx_to_edge[ri]
                    old_c = colors[ri]
                    new_c = random.randint(0, 3)
                    if new_c != old_c:
                        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
                        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                        tri_count += added - removed
                        colors[ri] = new_c
                        adj[old_c][u] &= ~(1 << v)
                        adj[old_c][v] &= ~(1 << u)
                        adj[new_c][u] |= (1 << v)
                        adj[new_c][v] |= (1 << u)
                continue
            
            u, v = best_edge
            idx = _compute_index(n, u, v)
            
            removed = (adj[c][u] & adj[c][v]).bit_count()
            added = (adj[best_new_c][u] & adj[best_new_c][v]).bit_count()
            tri_count += added - removed
            
            colors[idx] = best_new_c
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            adj[best_new_c][u] |= (1 << v)
            adj[best_new_c][v] |= (1 << u)
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_result = colors[:]
                no_improve = 0
            else:
                no_improve += 1
    
    return best_result if best_tri == 0 else None

def _tabu_search(n, time_budget):
    """
    Tabu search: uses a tabu list to prevent cycling and escape local minima.
    This is a structural change from pure SA/LS by adding memory of recent moves.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1
    
    best_result = None
    best_tri = float('inf')
    
    tabu_len = 50
    tabu_list = {}  # edge_index -> iteration when it became untabu
    
    for restart in range(50):
        if time.time() - start_time > time_budget:
            break
        
        colors = _greedy_init(n)
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)
        
        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        
        if tri_count == 0:
            return colors
        
        current_iter = 0
        no_improve = 0
        max_no_improve = 200
        
        while no_improve < max_no_improve:
            if time.time() - start_time > time_budget:
                break
            
            current_iter += 1
            
            # Find the best non-tabu move
            best_delta = float('inf')
            best_edge_idx = -1
            best_new_c = -1
            
            # Sample a subset of edges to keep it fast, or iterate all if n is small
            # For n=50, m=1225. Iterating all is feasible if we're careful.
            # Let's iterate all edges for better quality moves.
            
            for ri in range(m):
                if ri in tabu_list and current_iter < tabu_list[ri]:
                    continue
                    
                u, v = idx_to_edge[ri]
                old_c = colors[ri]
                
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    
                    removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
                    added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                    delta = added - removed
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_edge_idx = ri
                        best_new_c = new_c
            
            if best_edge_idx == -1:
                no_improve += 1
                # Escape: clear tabu list or pick random
                if no_improve > 20:
                    ri = random.randint(0, m - 1)
                    u, v = idx_to_edge[ri]
                    old_c = colors[ri]
                    new_c = random.randint(0, 3)
                    if new_c != old_c:
                        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
                        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                        tri_count += added - removed
                        colors[ri] = new_c
                        adj[old_c][u] &= ~(1 << v)
                        adj[old_c][v] &= ~(1 << u)
                        adj[new_c][u] |= (1 << v)
                        adj[new_c][v] |= (1 << u)
                        tabu_list[ri] = current_iter + tabu_len
                continue
            
            u, v = idx_to_edge[best_edge_idx]
            old_c = colors[best_edge_idx]
            
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            added = (adj[best_new_c][u] & adj[best_new_c][v]).bit_count()
            tri_count += added - removed
            
            colors[best_edge_idx] = best_new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[best_new_c][u] |= (1 << v)
            adj[best_new_c][v] |= (1 << u)
            
            tabu_list[best_edge_idx] = current_iter + tabu_len
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_result = colors[:]
                no_improve = 0
                if tri_count == 0:
                    return colors
            else:
                no_improve += 1
    
    return best_result if best_tri == 0 else None

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - elapsed
        if remaining < 5:
            return best
        
        # Targets: Try 50 first (record), then fall back
        # We allocate time based on the target. 50 is the hardest, so it gets the most time.
        targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 40, 38, 36, 34]
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 3:
                break
            
            # Allocate budget
            # If we have a lot of time, give it to SA.
            # SA is the primary solver that found 50 in the inspiration.
            # We now also use Tabu Search as a structural alternative.
            if target_n == 50:
                sa_budget = min(remaining - 20, 60)
                tabu_budget = min(remaining - 20, 40)
                ls_budget = min(remaining - 20, 10)
            elif target_n >= 45:
                sa_budget = min(remaining - 10, 30)
                tabu_budget = min(remaining - 10, 15)
                ls_budget = min(remaining - 10, 5)
            else:
                sa_budget = min(remaining - 5, 10)
                tabu_budget = min(remaining - 5, 5)
                ls_budget = min(remaining - 5, 2)
            
            if sa_budget < 2 and tabu_budget < 2:
                continue
                
            # 1. Try Simulated Annealing
            result = _sa_search(target_n, sa_budget)
            if result is not None:
                return {'n': target_n, 'colors': result}
            
            # 2. Try Tabu Search (Structural change: memory-based search)
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining > 5 and tabu_budget > 1:
                result = _tabu_search(target_n, tabu_budget)
                if result is not None:
                    return {'n': target_n, 'colors': result}
            
            # 3. Try Local Search if we have time
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining > 5 and ls_budget > 1:
                result = _local_search(target_n, ls_budget)
                if result is not None:
                    return {'n': target_n, 'colors': result}
        
        return best
    except Exception as e:
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
