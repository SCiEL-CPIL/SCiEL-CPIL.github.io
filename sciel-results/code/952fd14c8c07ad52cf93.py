import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    
    # 1. Precompute Conflict Graph (Adjacency Bitmasks)
    # adj[i] is a bitmask where bit j is 1 if lcm(i+1, j+1) > N
    adj = [0] * N
    
    # Helper to compute LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Build adjacency
    # We can optimize by noting that if i > N, it's not in range.
    # For i from 1 to N, j from i+1 to N
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) > N:
                adj[i-1] |= (1 << (j-1))
                adj[j-1] |= (1 << (i-1))
    
    # 2. Initial Lower Bound (Greedy Heuristic)
    best_size = 0
    best_set = set()
    
    def greedy_is():
        avail = (1 << N) - 1
        result = set()
        while avail:
            # Pick node with minimum degree in available subgraph
            best_v = -1
            best_deg = N + 1
            # Optimization: iterate only set bits
            a = avail
            while a:
                # Get index of lowest set bit
                v = (a & -a).bit_length() - 1
                a &= a - 1  # clear lowest set bit
                d = (adj[v] & avail).bit_count()
                if d < best_deg:
                    best_deg = d
                    best_v = v
            if best_v == -1:
                break
            result.add(best_v + 1)
            avail &= ~(1 << best_v)
            avail &= ~adj[best_v]
        return result

    # Try a few different greedy strategies or just one good one
    g_set = greedy_is()
    best_size = len(g_set)
    best_set = g_set
    
    # Try starting with divisors of 360 as a seed?
    # Divisors of 360: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 48, 60, 72, 80, 90, 120, 180, 360
    seed = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 48, 60, 72, 80, 90, 120, 180, 360}
    if len(seed) > best_size:
        best_size = len(seed)
        best_set = seed

    # 3. Branch and Bound with Greedy Coloring
    
    # To speed up coloring, we precompute the list of vertices for a given mask?
    # No, we compute it on the fly.
    
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        return verts

    def greedy_color_bound(available):
        """
        Returns a list of (vertex, color) sorted by color descending.
        Also returns the max color count (upper bound).
        """
        verts = get_vertices(available)
        if not verts:
            return [], 0
        
        # Order by degree in available subgraph (descending)
        # This heuristic helps produce a tighter coloring
        verts.sort(key=lambda v: (adj[v] & available).bit_count(), reverse=True)
        
        color = {}
        next_color = 0
        
        for v in verts:
            used_colors = set()
            # Check neighbors in verts that are already colored
            # Since we process in order, we can just check all previously processed
            # But we only care about neighbors.
            # adj[v] & available gives neighbors in available.
            # We need to intersect with the set of already colored vertices.
            # Since we are iterating through verts, we can keep a set of colored vertices?
            # Or just check if neighbor is in color dict.
            
            # Optimization:
            # We can iterate through neighbors of v in available.
            # If a neighbor u is in color, add color[u] to used.
            nb_mask = adj[v] & available
            # Iterate set bits of nb_mask
            m = nb_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                if u in color:
                    used_colors.add(color[u])
            
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            if c > next_color:
                next_color = c
        
        # Create list of (v, color)
        res = [(v, color[v]) for v in verts]
        # Sort by color descending
        res.sort(key=lambda x: x[1], reverse=True)
        return res, next_color + 1

    def solve(available, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline - 1:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                # Reconstruct set? 
                # This is a problem. We don't track the set in this recursive signature.
                # We need to track the set or reconstruct it.
                # To save time, we can just track size. But we need to return the set.
                # Let's modify solve to accept the current set as a frozenset or list?
                # That might be slow.
                # Alternative: Store the path.
                # Or, since we only need to return ONE valid set, we can store the best set found.
                # We need to pass the set down.
                pass
        
        # Simple bound
        if current_size + available.bit_count() <= best_size:
            return
        
        # Coloring bound
        colored_verts, ub = greedy_color_bound(available)
        if current_size + ub <= best_size:
            return
        
        # Branch
        # colored_verts is sorted by color descending
        # We iterate through them
        # We need to manage the 'available' mask carefully
        # For each vertex v in colored_verts:
        #   1. Check bound: current_size + color[v] <= best_size -> break
        #   2. Include v: new_avail = available & ~adj[v] & ~(1<<v)
        #      solve(new_avail, current_size + 1, set + {v})
        #   3. Exclude v: available = available & ~(1<<v)
        
        # To handle the set, we can pass a list `current_set_list` and append/pop.
        # But `solve` is recursive.
        
        # Let's refactor solve to accept `current_set` as a set (immutable) or use a global stack.
        # Given N=512, recursion depth is at most 512.
        # Passing a set copy at each step is $O(N)$. Total cost $O(N^2)$.
        # With 512, $512^2 \approx 260k$. If we have many nodes, this adds up.
        # But B&B should prune heavily.
        
        # Let's try passing the set.
        
        for v, c in colored_verts:
            if current_size + c <= best_size:
                break
            
            # Include v
            bit_v = 1 << v
            new_avail = available & ~adj[v] & ~bit_v
            solve(new_avail, current_size + 1, current_set | {v + 1})
            
            # Exclude v
            available &= ~bit_v
            
            # If available becomes empty, we can break?
            # No, we just continue to next vertex (which will be handled by the loop structure)
            # But wait, if we exclude v, we remove it from available.
            # The next iteration of the loop will pick the next vertex from `colored_verts`.
            # But `colored_verts` was computed for the *original* available set.
            # If we exclude v, the subgraph changes.
            # The standard Tomita algorithm handles this by re-coloring?
            # No, the standard algorithm processes vertices in the order determined by the *initial* coloring of the current node's subgraph.
            # When we exclude a vertex, we just remove it from the candidate list.
            # The bound `current_size + c` for subsequent vertices is still valid because removing a vertex cannot increase the chromatic number (or rather, the bound on IS size) of the remaining subgraph?
            # Actually, if we remove a vertex, the IS size of the remainder can only stay same or decrease.
            # So the bound holds.
            
            # However, we must ensure that `available` is updated correctly for the next branch.
            # The loop iterates over `colored_verts`.
            # If we exclude v, we remove it from `available`.
            # The next vertex v_next in `colored_verts` is still in `available` (unless it was a neighbor of v? No, we didn't remove neighbors in the exclude branch, only in the include branch).
            # Wait.
            # In the INCLUDE branch, we remove v and its neighbors.
            # In the EXCLUDE branch, we remove only v.
            # So `available` is modified in the EXCLUDE branch.
            # This modification persists for the next iteration of the loop.
            # This is correct. We are branching:
            #   Branch 1: v is in IS. (Recurse with v and neighbors removed)
            #   Branch 2: v is NOT in IS. (Remove v, continue with remaining candidates)
            
            # One detail: `colored_verts` contains vertices from the *original* available set.
            # If we excluded some previous vertices, they are no longer in `available`.
            # But we are iterating over the static list `colored_verts`.
            # We should check if v is still in `available`?
            # Actually, we removed v in the exclude step.
            # So for the next vertex, it is still in `available`.
            # But what if a previous exclude step removed a neighbor? No, exclude only removes the vertex itself.
            # So the logic holds.

    # We need to modify solve to accept current_set
    def solve(available, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline - 1:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Simple bound
        if current_size + available.bit_count() <= best_size:
            return
        
        # Coloring bound
        colored_verts, ub = greedy_color_bound(available)
        if current_size + ub <= best_size:
            return
        
        for v, c in colored_verts:
            if current_size + c <= best_size:
                break
            
            bit_v = 1 << v
            # Check if v is still in available? 
            # It should be, unless we removed it? 
            # We remove v at the end of the loop iteration (exclude step).
            # So at the start of the iteration, v is in available.
            
            # Include v
            new_avail = available & ~adj[v] & ~bit_v
            solve(new_avail, current_size + 1, current_set | {v + 1})
            
            # Exclude v
            available &= ~bit_v

    # Start
    all_nodes = (1 << N) - 1
    solve(all_nodes, 0, set())
    
    return list(best_set)

# EVOLVE-BLOCK-END
