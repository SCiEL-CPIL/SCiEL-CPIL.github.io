import time
import random
import numpy as np

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

def _build_adj_bitsets(colors, n):
    """Build adjacency bitsets from color list."""
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    return adj0, adj1

def _count_k222_bitsets(adj, n):
    """
    Count K2,2,2 subgraphs in graph defined by bitset adjacency.
    For each pair (c,d) that is a non-edge, find common neighbors,
    then check for K2,2 structure among them.
    """
    count = 0
    for c in range(n):
        for d in range(c + 1, n):
            # c and d must be non-adjacent (same part in K2,2,2)
            if adj[c] & (1 << d):
                continue
            # Common neighbors of c and d
            common = adj[c] & adj[d]
            # Get list of vertices in common
            verts = []
            mask = common
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                verts.append(v)
                mask &= mask - 1
            m = len(verts)
            if m < 4:
                continue
            # For each pair (a,b) in verts, check if they have >= 2 common neighbors in verts
            # that are also non-adjacent to each other... actually for K2,2,2:
            # parts: {c,d}, {a,b}, {e,f}
            # Need: a~e, a~f, b~e, b~f (all cross edges present)
            # a,b can be adjacent or not; e,f can be adjacent or not
            # But a,b must both connect to e,f
            for i in range(m - 1):
                a = verts[i]
                for j in range(i + 1, m):
                    b = verts[j]
                    # Common neighbors of a and b within verts
                    # These will be candidates for {e,f}
                    ab_common = adj[a] & adj[b] & common
                    # Need at least 2 vertices in ab_common that are in verts
                    # Count bits in ab_common & common (which is already intersected)
                    # Actually ab_common already has & common
                    # Check if ab_common has at least 2 bits
                    if ab_common & (ab_common - 1):
                        count += 1
    return count

def _is_valid_bitsets(adj0, adj1, n):
    return _count_k222_bitsets(adj0, n) == 0 and _count_k222_bitsets(adj1, n) == 0

def _simulated_annealing_bitsets(colors, n, max_iters=100000, time_limit=60):
    """
    Simulated annealing using bitset adjacency for fast K2,2,2 counting.
    """
    start_time = time.time()
    adj0, adj1 = _build_adj_bitsets(colors, n)
    current_cost = _count_k222_bitsets(adj0, n) + _count_k222_bitsets(adj1, n)
    
    if current_cost == 0:
        return colors, True
    
    # Precompute edge index to (i,j) mapping
    edge_map = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map.append((i, j))
            idx += 1
    num_edges = len(edge_map)
    
    T0 = 5.0
    T_min = 0.001
    T = T0
    best_cost = current_cost
    best_colors = colors[:]
    no_improve = 0
    stuck_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive temperature: reheat if stuck
        if no_improve > 2000:
            T = max(T0 * 0.3, T_min)
            no_improve = 0
            stuck_count += 1
        
        # Pick random edge to flip
        idx = random.randint(0, num_edges - 1)
        i, j = edge_map[idx]
        
        # Determine which color class this edge belongs to
        in_color0 = bool(adj0[i] & (1 << j))
        
        # Flip: remove from one, add to other
        if in_color0:
            adj0[i] &= ~(1 << j)
            adj0[j] &= ~(1 << i)
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
            colors[idx] = 1
        else:
            adj0[i] |= (1 << j)
            adj0[j] |= (1 << i)
            adj1[i] &= ~(1 << j)
            adj1[j] &= ~(1 << i)
            colors[idx] = 0
        
        new_cost = _count_k222_bitsets(adj0, n) + _count_k222_bitsets(adj1, n)
        delta = new_cost - current_cost
        
        if delta <= 0:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            if T > T_min:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    # Revert
                    if in_color0:
                        adj0[i] |= (1 << j)
                        adj0[j] |= (1 << i)
                        adj1[i] &= ~(1 << j)
                        adj1[j] &= ~(1 << i)
                        colors[idx] = 0
                    else:
                        adj0[i] &= ~(1 << j)
                        adj0[j] &= ~(1 << i)
                        adj1[i] |= (1 << j)
                        adj1[j] |= (1 << i)
                        colors[idx] = 1
            else:
                # Revert (temperature too low)
                if in_color0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                    adj1[i] &= ~(1 << j)
                    adj1[j] &= ~(1 << i)
                    colors[idx] = 0
                else:
                    adj0[i] &= ~(1 << j)
                    adj0[j] &= ~(1 << i)
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                    colors[idx] = 1
        
        T *= 0.9995
        
        if current_cost == 0:
            return colors, True
    
    return best_colors, best_cost == 0

def construct():
    deadline = time.time() + 145
    target_n = 29
    p = 29  # 29 is prime
    
    # 1. Try Paley(29) directly
    colors = _paley(target_n, p)
    if colors:
        adj0, adj1 = _build_adj_bitsets(colors, target_n)
        if _is_valid_bitsets(adj0, adj1, target_n):
            return {'n': target_n, 'colors': colors}
    
    # 2. Simulated Annealing from Paley (long run)
    if time.time() < deadline - 40:
        colors_sa = colors[:]
        _, success = _simulated_annealing_bitsets(colors_sa, target_n, max_iters=200000, time_limit=50)
        if success:
            return {'n': target_n, 'colors': colors_sa}
    
    # 3. Multi-restart with SA from random starts
    for attempt in range(30):
        if time.time() > deadline - 15:
            break
        colors_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        # Perturb Paley as alternative seed
        if attempt % 2 == 0 and colors:
            # Start from Paley with random flips
            colors_rand = colors[:]
            for _ in range(random.randint(5, 20)):
                idx = random.randint(0, len(colors_rand) - 1)
                colors_rand[idx] = 1 - colors_rand[idx]
        
        _, success = _simulated_annealing_bitsets(colors_rand, target_n, max_iters=80000, time_limit=20)
        if success:
            return {'n': target_n, 'colors': colors_rand}
    
    # 4. Fallback: try smaller n with Paley
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                colors = _paley(n, pp)
                if colors:
                    adj0, adj1 = _build_adj_bitsets(colors, n)
                    if _is_valid_bitsets(adj0, adj1, n):
                        return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
