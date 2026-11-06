import time
import random
import numpy as np

def _build_adj(colors, n):
    """Build adjacency matrix from colors list. Returns (n,n) bool array where True=color 1."""
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx]:
                adj[i, j] = True
                adj[j, i] = True
            idx += 1
    return adj

def _count_k222_np(adj, n):
    """Count monochromatic K2,2,2 subgraphs using vectorized NumPy operations.
    
    K2,2,2 is the complete tripartite graph with parts of size 2.
    A monochromatic K2,2,2 of color c exists iff there are 6 vertices partitioned
    into 3 pairs such that all 12 cross-pair edges have color c.
    
    We count for color 1 (adj=True) and color 0 (adj=False) separately.
    """
    total = 0
    # For each color, count K2,2,2
    for color_mask in [adj, ~adj]:
        # color_mask[i,j] = True means edge (i,j) has the target color
        # K2,2,2 with parts {a,b},{c,d},{e,f}: all cross edges present
        # Strategy: for each pair (a,b) as part 1, find common neighbors (vertices connected to both a and b)
        # Then among common neighbors, count K2,2 (pairs (c,d) and (e,f) with all 4 cross edges)
        # But we also need c-d and e-f NOT to matter (they're within parts, so not required)
        # Actually in K2,2,2, edges WITHIN parts don't exist, so c-d is not an edge of K2,2,2.
        # The 12 edges are: a-c,a-d,a-e,a-f, b-c,b-d,b-e,b-f, c-e,c-f,d-e,d-f
        # So we need: c,d,e,f all common neighbors of a,b, AND c-e,c-f,d-e,d-f all present,
        # AND c-d NOT required, e-f NOT required.
        
        # For pair (a,b): common = neighbors of both a and b
        # Then need to pick 4 from common forming K2,2 (bipartite complete between 2 and 2)
        # K2,2 on {c,d,e,f}: c-e, c-f, d-e, d-f all present
        # This means: in the subgraph on common, the induced bipartite graph between {c,d} and {e,f} is complete.
        # Equivalently: for some partition of 4 vertices into 2+2, all 4 cross edges present.
        
        count = 0
        for a in range(n - 1):
            for b in range(a + 1, n):
                if not color_mask[a, b]:
                    continue  # a-b must be in the same part, so NOT an edge in K2,2,2
                    # Wait, in K2,2,2, a and b are in the SAME part, so edge a-b is NOT part of K2,2,2.
                    # So a-b can be either color; it's just not one of the 12 required edges.
                    # Let me reconsider.
                # Actually, a and b are in the same part of the tripartite graph, so edge a-b is NOT 
                # required to be any particular color. The 12 edges are all cross-part edges.
                # So we don't need a-b to be any specific color.
                pass
        
        # Let me redo this more carefully.
        # K2,2,2 has 6 vertices in 3 parts of 2. All 12 edges between different parts must have the target color.
        # Edges within the same part are NOT part of the graph (they don't need to exist or have any color).
        # 
        # So for a set of 6 vertices {v1,v2,v3,v4,v5,v6} partitioned into {v1,v2},{v3,v4},{v5,v6}:
        # Required edges (all must be target color):
        # v1-v3, v1-v4, v1-v5, v1-v6, v2-v3, v2-v4, v2-v5, v2-v6, v3-v5, v3-v6, v4-v5, v4-v6
        # Not required: v1-v2, v3-v4, v5-v6
        
        # Approach: enumerate all 6-subsets? Too slow for n=29.
        # Better: for each pair (a,b) as a potential part:
        #   common = {v : color_mask[a,v] AND color_mask[b,v]}
        #   Then from common, we need to find two disjoint pairs (c,d) and (e,f) such that:
        #     c-e, c-f, d-e, d-f all have target color
        #   (c-d and e-f don't matter)
        #   
        #   For a fixed pair (c,d) in common:
        #     neighbors_of_cd = {v in common : color_mask[c,v] AND color_mask[d,v]}
        #     We need at least 2 more vertices e,f in neighbors_of_cd (distinct from c,d)
        #     But we also need c and d to be in common (already true)
        #     And e,f must be distinct from c,d
        #     Count: for each pair (c,d) in common, let S = neighbors_of_cd \ {c,d}
        #            then C(|S|, 2) gives the number of ways to choose (e,f)
        #   But this overcounts because each K2,2,2 is counted multiple times (once for each choice of which pair is "part 1")
        #   Each K2,2,2 has 3 parts, so it's counted 3 times.
        
        # Let's just count directly and divide by 3 at the end.
        count = 0
        for a in range(n - 1):
            for b in range(a + 1, n):
                # a,b form one part. They must both connect to all 4 other vertices with target color.
                common = np.where(color_mask[a] & color_mask[b])[0]
                common = common[common > b]  # ensure c,d,e,f > b to avoid overcounting? No, let's not restrict.
                # Actually, let's not restrict and just count all, then divide by 3.
                m = len(common)
                if m < 4:
                    continue
                # For each pair (c,d) in common:
                for ii in range(m - 1):
                    c = common[ii]
                    for jj in range(ii + 1, m):
                        d = common[jj]
                        # Find vertices in common that connect to both c and d
                        S = np.where(color_mask[c] & color_mask[d] & color_mask[a] & color_mask[b])[0]
                        # Remove c and d themselves
                        S = S[(S != c) & (S != d)]
                        # Also remove a and b (they shouldn't be in S since S must be common neighbors of c,d AND a,b)
                        # Actually S already requires connection to a and b, and c,d are in common (connected to a,b)
                        # But we need S to be additional vertices beyond a,b,c,d
                        S = S[(S != a) & (S != b)]
                        cnt = len(S)
                        if cnt >= 2:
                            count += cnt * (cnt - 1) // 2
        total += count
    
    # Each K2,2,2 is counted 3 times (once per part chosen as "a,b")
    total //= 3
    return total

def _count_k222_fast(adj, n):
    """Faster counting using bit operations on numpy arrays."""
    total = 0
    for color in [0, 1]:
        if color == 1:
            mask = adj
        else:
            mask = ~adj
        
        count = 0
        # Precompute bit representations for speed
        # Use Python ints for bit operations
        bits = [0] * n
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    bits[i] |= (1 << j)
        
        for a in range(n - 1):
            ba = bits[a]
            for b in range(a + 1, n):
                common = ba & bits[b]
                # Remove a and b
                common &= ~(1 << a)
                common &= ~(1 << b)
                # Get list of common neighbors
                verts = []
                bits_tmp = common
                while bits_tmp:
                    lsb = bits_tmp & (-bits_tmp)
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    bits_tmp ^= lsb
                m = len(verts)
                if m < 4:
                    continue
                for ii in range(m - 1):
                    c = verts[ii]
                    bc = bits[c]
                    for jj in range(ii + 1, m):
                        d = verts[jj]
                        S = common & bc & bits[d]
                        S &= ~(1 << c)
                        S &= ~(1 << d)
                        S &= ~(1 << a)
                        S &= ~(1 << b)
                        cnt = S.bit_count()
                        if cnt >= 2:
                            count += cnt * (cnt - 1) // 2
        total += count
    total //= 3
    return total

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _local_search(colors, n, max_iters=5000, time_limit=30):
    """Local search: flip edges to reduce K2,2,2 count."""
    start_time = time.time()
    adj = _build_adj(colors, n)
    current_cost = _count_k222_fast(adj, n)
    
    if current_cost == 0:
        return colors, True
    
    num_edges = n * (n - 1) // 2
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try random subset of edges
        if num_edges <= 200:
            candidates = range(num_edges)
        else:
            candidates = random.sample(range(num_edges), min(200, num_edges))
        
        best_idx = -1
        best_cost = current_cost
        
        for idx in candidates:
            i, j = edge_list[idx]
            # Flip edge
            old_val = adj[i, j]
            adj[i, j] = not old_val
            adj[j, i] = not old_val
            
            new_cost = _count_k222_fast(adj, n)
            
            # Unflip
            adj[i, j] = old_val
            adj[j, i] = old_val
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_idx = idx
        
        if best_idx >= 0:
            i, j = edge_list[best_idx]
            colors[best_idx] = 1 - colors[best_idx]
            adj[i, j] = not adj[i, j]
            adj[j, i] = not adj[j, i]
            current_cost = best_cost
            if current_cost == 0:
                return colors, True
        else:
            break
    
    return colors, current_cost == 0

def _simulated_annealing(colors, n, max_iters=20000, time_limit=60, T0=1.0, T1=0.01):
    """Simulated annealing for K2,2,2 minimization."""
    start_time = time.time()
    adj = _build_adj(colors, n)
    current_cost = _count_k222_fast(adj, n)
    
    if current_cost == 0:
        return colors, True
    
    best_colors = colors[:]
    best_cost = current_cost
    num_edges = n * (n - 1) // 2
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        T = T0 * (T1 / T0) ** (it / max_iters)
        
        # Pick random edge to flip
        idx = random.randint(0, num_edges - 1)
        i, j = edge_list[idx]
        
        old_val = adj[i, j]
        adj[i, j] = not old_val
        adj[j, i] = not old_val
        
        new_cost = _count_k222_fast(adj, n)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            colors[idx] = 1 - colors[idx]
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors, True
        else:
            # Revert
            adj[i, j] = old_val
            adj[j, i] = old_val
    
    return best_colors, best_cost == 0

def construct():
    deadline = time.time() + 140
    target_n = 29
    
    # 1. Try Paley(29)
    p = 29
    colors = _paley(target_n, p)
    if colors:
        adj = _build_adj(colors, target_n)
        if _count_k222_fast(adj, target_n) == 0:
            return {'n': target_n, 'colors': colors}
    
    # 2. Local search from Paley
    colors_ls = colors[:]
    _, success = _local_search(colors_ls, target_n, max_iters=10000, time_limit=40)
    if success:
        return {'n': target_n, 'colors': colors_ls}
    
    # 3. Simulated annealing from Paley
    if time.time() < deadline - 50:
        colors_sa = colors[:]
        _, success = _simulated_annealing(colors_sa, target_n, max_iters=50000, time_limit=80)
        if success:
            return {'n': target_n, 'colors': colors_sa}
    
    # 4. Multiple random restarts with local search
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        colors_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        _, success = _local_search(colors_rand, target_n, max_iters=3000, time_limit=8)
        if success:
            return {'n': target_n, 'colors': colors_rand}
    
    # 5. Try different Paley primes for smaller n
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                colors = _paley(n, pp)
                if colors:
                    adj = _build_adj(colors, n)
                    if _count_k222_fast(adj, n) == 0:
                        return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
