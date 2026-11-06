import time
import random
import sys

def construct():
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 145.0  # Safety margin before 150s kill
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed
        
        if remaining_time < 1.0:
            cliques = trivial_partition(n, p, seed)
        else:
            # Allocate time
            if idx == 0:
                instance_time_limit = min(remaining_time * 0.4, 50.0)
            else:
                instance_time_limit = min(remaining_time * 0.6, 90.0)
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
        
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def solve_instance(n, p, seed, time_limit):
    # Generate edges and build bitset adjacency
    rng = random.Random(seed)
    adj = [0] * n
    edge_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                
    if edge_count == 0:
        return []
        
    cliques = []
    start_time = time.time()
    
    # Mask for all vertices
    all_mask = (1 << n) - 1
    
    while edge_count > 0:
        if time.time() - start_time > time_limit:
            # Fallback to edges for remaining
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        edge_count -= 1
                    mask ^= lsb
            break
            
        # Find max clique in current adj
        clique = find_max_clique(adj, n)
        
        if not clique:
            # Safety: if no clique found but edges exist (shouldn't happen)
            # Pick any edge
            found = False
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        clique = [u, v]
                        found = True
                        break
                    mask ^= lsb
                if found: break
            if not found: break
            
        # Remove edges of clique
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edge_count -= 1
                    
        cliques.append(clique)
        
    return cliques

def find_max_clique(adj, n):
    best = []
    best_len = 0
    
    # Use a non-recursive or optimized recursive approach?
    # Python recursion limit is 1000 by default. Depth is at most n=64.
    # So recursion is fine.
    
    def expand(C, P):
        nonlocal best, best_len
        
        if not P:
            if len(C) > best_len:
                best_len = len(C)
                best = C[:]
            return
            
        # Bound: len(C) + popcount(P)
        if len(C) + P.bit_count() <= best_len:
            return
            
        # Greedy Coloring of P
        # Get vertices in P
        verts = []
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            verts.append(v)
            temp_P ^= lsb
            
        # Sort by degree in P
        verts.sort(key=lambda v: (P & adj[v]).bit_count(), reverse=True)
        
        colors = {}
        color_classes = []
        
        for v in verts:
            c = 0
            # Find smallest color c such that v is not adjacent to any vertex with color c
            while c < len(color_classes) and (adj[v] & color_classes[c]):
                c += 1
            if c == len(color_classes):
                color_classes.append(0)
            color_classes[c] |= (1 << v)
            colors[v] = c
            
        max_color = len(color_classes) - 1
        
        # Search in reverse order of colors
        for c in range(max_color, -1, -1):
            if len(C) + c + 1 <= best_len: # c+1 because color indices are 0..max_color
                break
                
            class_verts = []
            temp_C = color_classes[c]
            while temp_C:
                lsb = temp_C & -temp_C
                v = lsb.bit_length() - 1
                class_verts.append(v)
                temp_C ^= lsb
                
            for v in class_verts:
                # Only consider if v is still in P?
                # Yes, we iterate over color classes of P.
                # But P might have changed? No, P is fixed for this expand call.
                # However, we should check if v is actually in P.
                # Since we built color_classes from P, v is in P.
                
                new_C = C + [v]
                new_P = P & adj[v]
                
                expand(new_C, new_P)
                
                # Remove v from P for subsequent iterations
                P &= ~(1 << v)
                
                # Re-evaluate bound for remaining vertices in this color class?
                # The bound len(C) + c + 1 is still valid for the current c.
                # But P has shrunk, so popcount(P) is smaller.
                # The tight bound is len(C) + popcount(P).
                if len(C) + P.bit_count() <= best_len:
                    return

    expand([], (1 << n) - 1)
    return best
