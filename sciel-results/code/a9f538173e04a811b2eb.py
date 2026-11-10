# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    # Build conflict graph using bitsets (0-indexed: vertex i = number i+1)
    conflicts = [0] * n
    for a in range(n):
        for b in range(a + 1, n):
            x, y = a + 1, b + 1
            if (x * y) % (x + y) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)
    
    # Vertices with no conflicts can always be included
    isolated = [i for i in range(n) if conflicts[i] == 0]
    non_isolated = [i for i in range(n) if conflicts[i] != 0]
    
    m = len(non_isolated)
    if m == 0:
        return sorted([v + 1 for v in isolated])

    # Build compact conflict graph
    cconf = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            vi, vj = non_isolated[i], non_isolated[j]
            if (conflicts[vi] >> vj) & 1:
                cconf[i] |= (1 << j)
                cconf[j] |= (1 << i)
    
    # Greedy initial solution on compact graph
    def greedy_iset(candidates):
        mask = 0
        remaining = candidates
        while remaining:
            r = remaining
            best_v = -1
            best_d = float('inf')
            while r:
                lb = r & (-r)
                v = lb.bit_length() - 1
                d = bin(cconf[v] & remaining).count('1')
                if d < best_d:
                    best_d = d
                    best_v = v
                r ^= lb
            mask |= (1 << best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~cconf[best_v]
        return mask

    initial_mask = greedy_iset((1 << m) - 1)
    best_size = bin(initial_mask).count('1')
    best_mask_global = initial_mask
    start_time = time.time()
    TIMEOUT = 55

    def coloring_bound(candidates):
        # Greedy coloring: assign smallest possible color
        # Order vertices by degree descending in the subgraph
        verts = []
        r = candidates
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            verts.append(v)
            r ^= lb
            
        # Sort by degree descending
        verts.sort(key=lambda i: bin(cconf[i] & candidates).count('1'), reverse=True)
        
        colors = {}
        max_color = 0
        for v in verts:
            used = set()
            # Check neighbors in verts that are already colored
            # Since we process in order, we only need to check those already in 'colors'
            # But 'colors' only contains processed vertices.
            # We need to check if any neighbor u of v has been assigned a color.
            for u in verts:
                if u not in colors:
                    continue
                if (cconf[v] >> u) & 1:
                    used.add(colors[u])
            
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
        return max_color

    def bnb2(candidates, count, current_mask):
        nonlocal best_size, best_mask_global, start_time
        if time.time() - start_time > TIMEOUT:
            return
        if candidates == 0:
            if count > best_size:
                best_size = count
                best_mask_global = current_mask
            return
        
        # Upper bound using coloring
        ub = count + coloring_bound(candidates)
        if ub <= best_size:
            return

        # Pick vertex with max degree in candidates
        r = candidates
        best_v = -1
        best_d = -1
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = bin(cconf[v] & candidates).count('1')
            if d > best_d:
                best_d = d
                best_v = v
            r ^= lb
        
        v_bit = 1 << best_v
        
        # Include
        bnb2(candidates & ~v_bit & ~cconf[best_v], count + 1, current_mask | v_bit)
        
        # Exclude
        bnb2(candidates & ~v_bit, count, current_mask)

    bnb2((1 << m) - 1, 0, 0)
    
    # Reconstruct result from best_mask_global
    result_set = set(isolated)
    for i in range(m):
        if (best_mask_global >> i) & 1:
            result_set.add(non_isolated[i])
            
    result = sorted([v + 1 for v in result_set])
    return result
# EVOLVE-BLOCK-END
