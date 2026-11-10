import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    n = 256

    # Precompute conflicts as bitmasks
    conflicts = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)

    # Build non-conflict adjacency (clique graph): bit j set in nonconf[a] iff a,b don't conflict
    # We want maximum clique in this graph = maximum independent set in conflict graph
    nonconf = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if a != b and not (conflicts[a] & (1 << b)):
                nonconf[a] |= (1 << b)

    def popcount(x):
        return bin(x).count('1')

    # Tomita's maximum clique algorithm with pivot-based coloring
    # Works on the non-conflict graph (we want max clique)
    def max_clique_tomita(vertices, current_clique, best_clique, best_size, deadline):
        """
        vertices: bitmask of candidate vertices
        current_clique: bitmask of vertices in current clique
        best_clique: [bitmask] mutable, holds best found
        best_size: [int] mutable, holds best size found
        """
        if time.time() > deadline:
            return
        
        if not vertices:
            sz = popcount(current_clique)
            if sz > best_size[0]:
                best_size[0] = sz
                best_clique[0] = current_clique
            return
        
        # Color the vertices in order for upper bound
        # Greedy coloring: assign colors to vertices
        color_class = [0] * (n + 1)  # color_class[v] = color assigned to v
        color_count = [0] * (n + 1)  # number of vertices with color i
        max_color = 0
        remaining = vertices
        
        while remaining:
            max_color += 1
            color_count[max_color] = 0
            uncolored = remaining
            while uncolored:
                # Pick a vertex from uncolored
                v = (uncolored & (-uncolored)).bit_length() - 1
                uncolored &= uncolored - 1
                # Check if v can take this color (no conflict with same-color vertices in remaining)
                # For simplicity in clique: color v with max_color if it doesn't conflict with any vertex already colored with max_color
                # Actually for clique: we need v to NOT be adjacent to vertices of same color
                # Wait - in Tomita's algorithm for maximum clique:
                # We color such that each color class is an independent set in the graph
                # i.e., no two vertices of same color are adjacent
                # So v can have color c if v is not adjacent to any vertex already assigned color c
                same_color_mask = 0
                for u in range(1, n + 1):
                    if color_class[u] == max_color:
                        same_color_mask |= (1 << u)
                # v can have this color if no edge between v and any same-color vertex
                if not (nonconf[v] & same_color_mask):
                    color_class[v] = max_color
                    color_count[max_color] += 1
                    remaining &= ~(1 << v)
                    break
            if not uncolored and not remaining:
                break
        
        # Now search: for each color level from high to low
        # If current_clique size + max possible from this color level <= best, prune
        search_order = []
        for c in range(max_color, 0, -1):
            for v in range(1, n + 1):
                if color_class[v] == c:
                    search_order.append(v)
        
        for v in search_order:
            if time.time() > deadline:
                return
            
            # Upper bound: current size + number of vertices with color >= color_class[v]
            ub = popcount(current_clique) + sum(color_count[c] for c in range(color_class[v], max_color + 1))
            if ub <= best_size[0]:
                break
            
            # Add v to clique
            new_clique = current_clique | (1 << v)
            # New candidates: intersection of remaining vertices that are adjacent to v
            new_candidates = vertices & nonconf[v]
            # Remove v from candidates
            new_candidates &= ~(1 << v)
            
            max_clique_tomita(new_candidates, new_clique, best_clique, best_size, deadline)
            
            # Remove v from remaining for subsequent iterations
            vertices &= ~(1 << v)
    
    # Greedy heuristic to get initial good solution
    def greedy_clique():
        """Greedy: repeatedly pick vertex with max degree in remaining graph"""
        remaining = (1 << (n + 1)) - 2  # bits 1..n set
        clique = 0
        while remaining:
            # Pick vertex with max nonconf degree within remaining
            best_v = -1
            best_deg = -1
            temp = remaining
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                temp &= temp - 1
                deg = popcount(nonconf[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v < 0:
                break
            clique |= (1 << best_v)
            remaining &= nonconf[best_v]
        return clique

    # Local search on clique
    def local_search_clique(clique, max_iters=200):
        best_clique = clique
        best_size = popcount(clique)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try to add a vertex
            for b in range(1, n + 1):
                bit_b = 1 << b
                if clique & bit_b:
                    continue
                # b must be non-conflicting with all in clique
                if not (conflicts[b] & clique):
                    clique |= bit_b
                    sz = popcount(clique)
                    if sz > best_size:
                        best_size = sz
                        best_clique = clique
                    improved = True
                    break
            if improved:
                continue
            
            # Try 1-swap: remove one, add one
            for b in range(1, n + 1):
                bit_b = 1 << b
                if clique & bit_b:
                    continue
                cwb = conflicts[b] & clique
                if cwb == 0:
                    continue
                if cwb & (cwb - 1) == 0:  # exactly one conflict
                    a_bit = cwb
                    new_clique = (clique & ~a_bit) | bit_b
                    if not (conflicts[b] & (new_clique & ~bit_b)):
                        sz = popcount(new_clique)
                        if sz > best_size:
                            clique = new_clique
                            best_size = sz
                            best_clique = new_clique
                            improved = True
                            break
                        elif sz == best_size and random.random() < 0.1:
                            clique = new_clique
                            improved = True
                            break
            if improved:
                continue
            
            # Try 2-swap: remove two, add one
            for b in range(1, n + 1):
                bit_b = 1 << b
                if clique & bit_b:
                    continue
                cwb = conflicts[b] & clique
                if cwb == 0:
                    continue
                if cwb & (cwb - 1) == 0:
                    continue
                if (cwb & (cwb - 1)) & ((cwb & (cwb - 1)) - 1) == 0:
                    a_bit = cwb & (-cwb)
                    c_bit = cwb ^ a_bit
                    new_clique = (clique & ~a_bit & ~c_bit) | bit_b
                    if not (conflicts[b] & (new_clique & ~bit_b)):
                        sz = popcount(new_clique)
                        if sz > best_size:
                            clique = new_clique
                            best_size = sz
                            best_clique = new_clique
                            improved = True
                            break
            if improved:
                continue
            
            # Random kick
            if random.random() < 0.3:
                elems = [i for i in range(1, n + 1) if clique & (1 << i)]
                if elems:
                    k = random.randint(1, 3)
                    to_remove = random.sample(elems, min(k, len(elems)))
                    for r in to_remove:
                        clique &= ~(1 << r)
                    # Greedy refill
                    remaining = [e for e in range(1, n + 1) if not (clique & (1 << e))]
                    random.shuffle(remaining)
                    for e in remaining:
                        if not (conflicts[e] & clique):
                            clique |= (1 << e)
                    sz = popcount(clique)
                    if sz > best_size:
                        best_size = sz
                        best_clique = clique
                    improved = True
            
            if not improved:
                break
        
        return best_clique, best_size

    # Phase 1: Greedy initial solution
    random.seed(42)
    init_clique = greedy_clique()
    init_clique, init_sz = local_search_clique(init_clique, max_iters=200)
    
    best_mask = init_clique
    best_size = init_sz

    # Phase 2: Run Tomita max clique on full graph
    if time.time() < deadline - 10:
        best_clique = [0]
        best_sz = [0]
        # Start with empty clique, all vertices as candidates
        all_vertices = (1 << (n + 1)) - 2  # bits 1..n
        max_clique_tomita(all_vertices, 0, best_clique, best_sz, deadline - 5)
        if best_sz[0] > best_size:
            best_size = best_sz[0]
            best_mask = best_clique[0]

    # Phase 3: Local search from Tomita result
    if time.time() < deadline - 5:
        s_mask, sz = local_search_clique(best_mask, max_iters=300)
        if sz > best_size:
            best_size = sz
            best_mask = s_mask

    # Phase 4: Randomized restarts with kicks
    for trial in range(40):
        if time.time() > deadline - 3:
            break
        s_mask = best_mask
        elems = [i for i in range(1, n + 1) if s_mask & (1 << i)]
        K = random.randint(3, 10)
        to_remove = random.sample(elems, min(K, len(elems)))
        for r in to_remove:
            s_mask &= ~(1 << r)
        remaining = [e for e in range(1, n + 1) if not (s_mask & (1 << e))]
        random.shuffle(remaining)
        for e in remaining:
            if not (conflicts[e] & s_mask):
                s_mask |= (1 << e)
        s_mask, sz = local_search_clique(s_mask, max_iters=150)
        if sz > best_size:
            best_size = sz
            best_mask = s_mask

    # Convert to list
    result = [i for i in range(1, n + 1) if best_mask & (1 << i)]
    if not result:
        result = [1]
    return result
