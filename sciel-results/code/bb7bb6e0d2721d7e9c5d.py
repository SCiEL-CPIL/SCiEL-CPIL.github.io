import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    def greedy_init():
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_improve(mask):
        improved = True
        while improved:
            improved = False
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    # Precompute Complement Adjacency for Tomita-style Max Clique (which is MIS in original)
    comp_adj = [0] * N
    for v in range(N):
        if forbidden[v]:
            comp_adj[v] = 0
        else:
            # Neighbors in complement are those in cand_mask, not v, and not in adj[v]
            comp_adj[v] = cand_mask & ~adj[v] & ~(1 << v)
    
    # Tomita-style Maximum Clique Search on Complement Graph
    # This finds Maximum Independent Set in the original graph
    def max_clique_tomita(P, current_size, current_set, best_size, best_set, deadline):
        # P: bitmask of candidate vertices for the clique (in comp graph)
        # Returns (best_size, best_set)
        
        if time.time() > deadline:
            return best_size, best_set
        
        # If no candidates, update best if needed
        if P == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return best_size, best_set
        
        # Bound: Greedy coloring of P in comp graph
        # Order vertices by degree in comp graph (descending)
        # Assign colors greedily
        # The max color is an upper bound on clique size
        
        # Extract vertices from P
        verts = []
        m = P
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Sort by degree in comp graph within P (descending)
        # Degree in comp graph within P = number of neighbors in P
        verts.sort(key=lambda v: (comp_adj[v] & P).bit_count(), reverse=True)
        
        # Greedy coloring
        # colors[v] = color assigned to v
        # color_bound[v] = max color used so far when processing v
        # We assign colors such that adjacent vertices in comp graph get different colors
        # Wait, for Max Clique, we need a coloring where non-adjacent in comp graph can share color?
        # No. Tomita's algorithm for Max Clique uses a coloring of the graph such that
        # vertices with the same color are non-adjacent (independent set).
        # Then, a clique can contain at most one vertex from each color class.
        # So the number of colors is an upper bound on the clique size.
        
        # So we need to color the comp graph such that adjacent vertices have different colors.
        # This is standard graph coloring.
        
        # Greedy coloring:
        # For each vertex in order, assign the smallest color not used by its neighbors in the processed set.
        
        color = [0] * N
        max_color = 0
        # color_bound for each vertex: the max color assigned to any vertex up to and including this one
        color_bound = [0] * N
        # We need to track which colors are used by neighbors for each vertex
        
        # More efficient:
        # For each vertex v in order:
        #   used_colors = set of colors of neighbors of v that have been colored
        #   color[v] = smallest positive integer not in used_colors
        #   max_color = max(max_color, color[v])
        #   color_bound[v] = max_color
        
        # To make it faster, we can use bitmasks for colors if N is small, but N=256 is too large for bitmask of colors.
        # However, the number of colors is usually small (maybe < 20 for this graph?).
        # Let's use a list of booleans for used colors per vertex.
        
        # Actually, a simpler approach for the bound:
        # Just compute the max color used in a greedy coloring.
        
        # Let's implement a fast greedy coloring.
        # We process vertices in the sorted order.
        # For each vertex, we check its neighbors that have been processed.
        # We need to know the colors of those neighbors.
        
        # To speed this up, we can maintain an array of used colors.
        # But checking neighbors is O(degree).
        
        # Given time constraints, let's use a simpler bound if coloring is too slow.
        # But let's try the coloring first.
        
        # Optimization: Only color vertices that are still in P.
        # The order is already sorted.
        
        # We need to map vertex index to position in the sorted list?
        # No, we just iterate through the sorted list.
        
        # Let's store the color for each vertex in an array.
        # We need to quickly find the colors of neighbors.
        
        # This might be too slow in Python for deep search.
        # Let's use a simpler bound: |P|
        # Or a better heuristic: 
        # The size of the maximum clique is at most the size of the maximum independent set in the complement of the subgraph? No.
        
        # Let's try a different bound:
        # Greedy coloring is standard. Let's implement it efficiently.
        
        # We'll use an array `used_colors` of size N+1 (colors 1..N)
        # For each vertex v, we clear used_colors, then for each neighbor u in P that is processed,
        # we mark used_colors[color[u]] = True.
        # Then find the smallest c such that used_colors[c] is False.
        
        # To know if a neighbor is processed, we can check if it's in the "processed" set.
        # Since we process in order, we can just keep a set of processed vertices.
        # But checking membership in a set is O(1).
        
        # However, iterating through neighbors and checking if they are in P and processed is O(degree).
        
        # Let's try it.
        
        processed = set()
        used_colors = [False] * (N + 1)  # colors 1..N
        
        # We need to store the color assigned to each vertex
        vertex_color = {}
        
        # The bound for the current node is the max color used.
        # But Tomita's algorithm uses a "color sort" where the bound is the color of the last vertex?
        # No, the bound is the max color.
        # But for pruning, we can use the color of each vertex as a bound for the subproblem starting from that vertex?
        # In Tomita's algorithm, the vertices are ordered by color, and the color of vertex i is an upper bound
        # on the clique size in the subgraph induced by vertices i, i+1, ..., n.
        
        # So we need to compute the color for each vertex in the sorted order, and store it.
        # Then, when branching, we use the color of the current vertex as the bound.
        
        # Let's compute the colors.
        
        # To speed up, we can skip the full coloring if |P| <= best_size - current_size
        if P.bit_count() <= best_size - current_size:
            return best_size, best_set
        
        # Greedy coloring
        # We'll process vertices in the sorted order.
        # For each vertex, we assign the smallest available color.
        
        # To make it faster, we can use a bitset for used colors if the number of colors is small.
        # But let's use a simple array for now.
        
        # We need to know the colors of neighbors that have been processed.
        # We can store the color of each vertex in a dictionary or array.
        
        # Let's use an array `v_color` of size N, initialized to 0.
        v_color = [0] * N
        
        # Process vertices in order
        max_c = 0
        for v in verts:
            # Clear used_colors
            # Instead of clearing the whole array, we can use a timestamp or just reset the relevant parts.
            # For simplicity, let's use a set of used colors.
            used_c = set()
            
            # For each neighbor u of v in P that has been processed
            # How to check if u is processed? We can check if u is in the `processed` set.
            # But `processed` is a set of vertices.
            
            # Iterate through neighbors of v in comp graph
            # m = comp_adj[v] & P
            # But we only want those in `processed`.
            
            # This is O(degree).
            
            # Let's try a different approach:
            # Instead of checking neighbors, we can maintain an array of color counts?
            # No, we need to know if a specific color is used by a neighbor.
            
            # Given the time limit, let's use a simpler bound: |P|
            # And rely on the search order and pruning by current_size.
            
            # Actually, let's use a very simple bound: |P|
            # And hope that the search order (by degree) helps.
            
            # For now, let's just use |P| as the bound.
            # This is weak, but fast.
            
            break  # Exit the coloring loop, use |P| as bound
        
        # Use |P| as the bound
        bound = P.bit_count()
        
        if current_size + bound <= best_size:
            return best_size, best_set
        
        # Branch: pick a vertex from P
        # To improve pruning, pick a vertex with high degree in comp graph?
        # Or low degree?
        # In Max Clique, picking a vertex with high degree might be better?
        # Or we can iterate through all vertices in P.
        
        # Tomita's algorithm iterates through vertices in the order they were colored/sorted.
        # Let's iterate through verts in order.
        
        # For each vertex v in verts:
        #   If current_size + color_bound[v] <= best_size: break (since subsequent vertices have lower color bounds)
        #   Add v to current_set
        #   Recurse with P = P & comp_adj[v] (neighbors of v in comp graph)
        #   Remove v from P
        
        # But we need the color_bound for each vertex.
        # Since we didn't compute the colors properly, let's just iterate and use |P| as bound for each branch?
        # No, that's too weak.
        
        # Let's try a different strategy:
        # Use a simple greedy heuristic to find a large clique, then use B&B to improve.
        # Or, use a limited B&B with a simple bound.
        
        # Given the complexity, let's implement a simple recursive search with a simple bound.
        
        # Pick a vertex v from P (e.g., the one with highest degree in comp graph within P)
        # Try including v
        # Try excluding v
        
        # This is exponential, but with pruning it might work for small N.
        
        # Let's pick v = verts[0] (highest degree)
        v = verts[0]
        
        # Branch 1: Include v
        new_P = P & comp_adj[v]
        new_size = current_size + 1
        new_set = current_set | (1 << v)
        best_size, best_set = max_clique_tomita(new_P, new_size, new_set, best_size, best_set, deadline)
        
        # Branch 2: Exclude v
        P = P & ~(1 << v)
        if P.bit_count() > best_size - current_size:  # Prune if bound is too low
            best_size, best_set = max_clique_tomita(P, current_size, current_set, best_size, best_set, deadline)
        
        return best_size, best_set

    # Let's focus on improving the Local Search.
    # The current local search only adds vertices.
    # We need to allow swaps.
    
    def local_search_with_swaps(mask, max_iters=1000):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            iters += 1
            
            # 1. Try to add
            mask = local_improve(mask)
            
            # 2. Try 1-1 swaps
            m = mask
            swapped = False
            while m:
                lsb_u = m & -m
                u = lsb_u.bit_length() - 1
                m ^= lsb_u
                
                # Find v in comp_adj[u] (i.e., not neighbor in original)
                # that is not in mask, and replacing u with v allows adding more?
                # Simple 1-1 swap: remove u, add v. Size same.
                # Only useful if it leads to a better local optimum.
                
                # Find v in candidates, v not in mask, v not neighbor of (mask \ {u})
                # i.e., (adj[v] & (mask ^ lsb_u)) == 0
                
                # Candidates for v:
                # Must be in cand_mask
                # Must not be in mask
                # Must not be adjacent to any node in mask except possibly u
                # So (adj[v] & mask) is a subset of {u}
                # i.e., (adj[v] & mask) == 0 or (adj[v] & mask) == lsb_u
                
                # Let's iterate through all candidates not in mask.
                # This is O(N) per u, so O(N^2) per iteration.
                # N=256, N^2=65536. 1000 iters -> 65M ops. Might be tight in 58s.
                
                # Optimization:
                # Only consider v that are "close" to u?
                # Or just iterate through all v not in mask.
                
                not_in_mask = cand_mask & ~mask
                mv = not_in_mask
                while mv:
                    lsb_v = mv & -mv
                    v = lsb_v.bit_length() - 1
                    mv ^= lsb_v
                    
                    # Check if v can replace u
                    # New set would be (mask ^ lsb_u) | lsb_v
                    # We need (adj[v] & (mask ^ lsb_u)) == 0
                    if (adj[v] & (mask ^ lsb_u)) == 0:
                        # Swap is valid.
                        # Does it help?
                        # Try it and see if we can add more.
                        new_mask = (mask ^ lsb_u) | lsb_v
                        new_mask = local_improve(new_mask)
                        if new_mask.bit_count() > mask.bit_count():
                            mask = new_mask
                            improved = True
                            swapped = True
                            break
                        # Even if size doesn't increase immediately, it might escape a local optimum.
                        # But for now, let's only accept if size increases.
                
                if swapped:
                    break
                    
        return mask

    # Main Loop
    best_mask = 0
    best_size = 0
    
    # Initial greedy
    mask = greedy_init()
    mask = local_improve(mask)
    best_mask = mask
    best_size = mask.bit_count()
    
    # ILS
    while time.time() - start_time < DEADLINE:
        current = best_mask
        
        # Perturb: remove random vertices
        # Number of removals proportional to size?
        num_removals = random.randint(2, 5)
        for _ in range(num_removals):
            if not current:
                break
            m_iter = current
            # Pick random set bit
            count = current.bit_count()
            r = random.randint(1, count)
            # Find r-th set bit
            # Simple way: iterate
            cnt = 0
            m_temp = current
            while m_temp:
                lsb = m_temp & -m_temp
                cnt += 1
                if cnt == r:
                    current ^= lsb
                    break
                m_temp ^= lsb
        
        # Improve
        current = local_improve(current)
        current = local_search_with_swaps(current, max_iters=50)
        
        if current.bit_count() > best_size:
            best_size = current.bit_count()
            best_mask = current
        
        # Also try fresh greedy with randomization
        if random.random() < 0.3:
            def greedy_random():
                available_mask = cand_mask
                selected_mask = 0
                while available_mask:
                    best_v = -1
                    min_deg = N + 1
                    m = available_mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        m ^= lsb
                        deg = (adj[v] & available_mask).bit_count()
                        deg += random.uniform(0, 1.0)
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                        if deg == 0:
                            break
                    if best_v == -1:
                        break
                    selected_mask |= (1 << best_v)
                    available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
                return selected_mask

            mask_r = greedy_random()
            mask_r = local_improve(mask_r)
            mask_r = local_search_with_swaps(mask_r, max_iters=50)
            if mask_r.bit_count() > best_size:
                best_size = mask_r.bit_count()
                best_mask = mask_r

    # Convert best_mask to list of values (1-based)
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
