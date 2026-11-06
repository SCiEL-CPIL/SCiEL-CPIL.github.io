import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try to find a valid graph on n vertices
    # The best known is 39. We aim for 39.
    for n in [39, 38, 37]:
        result = try_construct(n, deadline)
        if result:
            return result
            
    # Fallback to Polarity Graph O_5 (n=31)
    result = construct_polarity_graph(5)
    if result:
        return result

    # Fallback to a known valid smaller graph if all else fails
    # Ten disjoint triangles: n=30, alpha=10
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def construct_polarity_graph(q):
    # q must be a prime power
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    points.append((0, 1, (z * inv_y) % q))
                else:
                    points.append((0, 0, 1))
    
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            dot = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % q
            if dot == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}

def try_construct(n, deadline):
    neighbors = [0] * n
    
    def can_add_edge(u, v):
        if (neighbors[u] >> v) & 1:
            return False
        if neighbors[u] & neighbors[v]:
            return False
        return True

    def add_edge(u, v):
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)

    def max_independent_set():
        best_size = 0
        
        def greedy_color_upper_bound(mask):
            # Greedy coloring of the induced subgraph on 'mask'
            # Returns number of colors used (upper bound on alpha)
            remaining = mask
            colors = 0
            color_classes = []
            while remaining:
                colors += 1
                # Pick a vertex
                # Heuristic: pick vertex with highest degree in remaining?
                # Or just lowest index.
                # Let's pick lowest index for simplicity
                v = (remaining & -remaining).bit_length() - 1
                # Add v to current color class
                # Remove v and its neighbors from remaining
                # Wait, greedy coloring:
                # Assign color 1 to as many vertices as possible (independent set).
                # Then color 2, etc.
                # So we need to find a maximal independent set in 'remaining' for color 1.
                # Then remove them and repeat.
                
                # This is effectively finding a greedy independent set.
                # So the number of iterations is the number of colors.
                # And each iteration removes an independent set.
                # So this is correct.
                
                # However, to get a *tight* upper bound, we should use a heuristic
                # that produces *small* independent sets?
                # No, we want to minimize the number of colors.
                # So we want *large* independent sets.
                # Greedy with min-degree usually gives large independent sets.
                
                # Let's use a simple greedy: pick v, remove N(v) and v.
                # This gives a maximal independent set.
                # The size of this set is the "color class size".
                # The number of such sets is the bound.
                
                # Actually, the standard bound is:
                # alpha(G) <= chi(G).
                # Greedy coloring gives an upper bound on chi(G).
                # So we just need to run a greedy coloring.
                
                # Implementation of greedy coloring:
                # 1. Order vertices (e.g. by degree).
                # 2. Assign smallest possible color.
                
                # For speed in B&B, let's use a simpler bound:
                # alpha(G) <= |V| / delta_min + 1 ? No.
                # Let's stick to the recursive B&B with a simple bound:
                # upper_bound = current_size + mask.bit_count()
                # This is too weak.
                
                # Let's use: upper_bound = current_size + greedy_independent_set_size(mask)
                # No, that's a lower bound on alpha.
                
                # I will use a precomputed greedy coloring count.
                
                # For now, let's use a simple heuristic:
                # Pick a vertex v.
                # Branch 1: Include v. New mask = mask & ~N(v) & ~(1<<v).
                # Branch 2: Exclude v. New mask = mask & ~(1<<v).
                # Prune if current_size + mask.bit_count() <= best_size.
                
                # This is what the original code did.
                # To improve, I will add a greedy independent set calculation to update best_size early.
                
                pass # Placeholder
        
        def bnb(mask, current_size):
            nonlocal best_size
            if time.time() > deadline:
                return -1
            
            if not mask:
                if current_size > best_size:
                    best_size = current_size
                return
            
            if current_size + mask.bit_count() <= best_size:
                return
            
            # Greedy lower bound update
            # Pick a vertex, remove neighbors, repeat.
            m = mask
            greedy_count = 0
            while m:
                v = (m & -m).bit_length() - 1
                greedy_count += 1
                m &= ~(neighbors[v] | (1 << v))
            
            if current_size + greedy_count > best_size:
                best_size = current_size + greedy_count
            
            # Branch
            # Pick vertex with highest degree in mask to prune better?
            # Let's pick the first one for simplicity, but maybe sort?
            # Sorting is expensive.
            # Let's pick the vertex with max degree in mask.
            max_deg = -1
            pick = -1
            temp = mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (neighbors[v] & mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pick = v
            
            # Branch 1: Include pick
            new_mask = mask & ~(neighbors[pick] | (1 << pick))
            ret = bnb(new_mask, current_size + 1)
            if ret == -1: return -1
            
            # Branch 2: Exclude pick
            new_mask = mask & ~(1 << pick)
            ret = bnb(new_mask, current_size)
            if ret == -1: return -1
            
        all_vertices = (1 << n) - 1
        ret = bnb(all_vertices, 0)
        if ret == -1:
            return None
        return best_size

    random.seed(int(time.time() * 1000) % 10000)
    
    for attempt in range(200): # Reduced attempts to allow more time for B&B?
        if time.time() > deadline:
            break
            
        neighbors = [0] * n
        
        potential_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(potential_edges)
        
        for u, v in potential_edges:
            if can_add_edge(u, v):
                add_edge(u, v)
            if time.time() > deadline:
                break
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            alpha = max_independent_set()
            if alpha is None:
                break
            if alpha <= 10:
                edges = []
                for i in range(n):
                    mask = neighbors[i]
                    while mask:
                        j = (mask & -mask).bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask &= mask - 1
                return {'n': n, 'edges': edges}
                
            # Find greedy independent set
            def find_greedy_independent_set(vertices):
                remaining = vertices
                s = set()
                while remaining:
                    min_deg = float('inf')
                    pick = -1
                    temp = remaining
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        temp &= temp - 1
                        deg = (neighbors[v] & remaining).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            pick = v
                    s.add(pick)
                    remaining &= ~(neighbors[pick] | (1 << pick))
                return s
            
            S = find_greedy_independent_set((1 << n) - 1)
            
            added = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u = S_list[i]
                    v = S_list[j]
                    if can_add_edge(u, v):
                        add_edge(u, v)
                        added = True
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
                    
            if not added:
                improved = False
            else:
                improved = True
        
    return None
