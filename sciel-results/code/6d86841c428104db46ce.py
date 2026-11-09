import random
import time

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples and their bitmasks
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                mask = (1 << a) | (1 << b) | (1 << c)
                p_ab = (a, b)
                p_ac = (a, c)
                p_bc = (b, c)
                all_triples.append((mask, p_ab, p_ac, p_bc, (a,b,c)))
                
    # Total triples: 560
    # Max possible linear packing size: 37
    
    best_alpha = n + 1
    best_triples = []
    
    # Exact independence number calculation using branch-and-bound
    # on the shadow graph (vertices are points, edges connect pairs in same triple)
    def calc_alpha(masks):
        # Build shadow graph: adjacency bitmask for each point
        # adj[i] is a bitmask of points j such that i and j appear in some triple
        adj = [0] * n
        for m in masks:
            # Extract the 3 points from mask
            pts = []
            for i in range(n):
                if m & (1 << i):
                    pts.append(i)
            # Add edges between all pairs in this triple
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    adj[pts[i]] |= (1 << pts[j])
                    adj[pts[j]] |= (1 << pts[i])
        
        # Now find maximum independent set in this graph
        # A set S is independent if no two vertices in S are adjacent
        # i.e., for all u,v in S, adj[u] & (1<<v) == 0
        
        best_size = 0
        
        # Branch and bound with bitset
        # candidates is a bitmask of available vertices
        # We use a recursive search with pruning
        
        def search(candidates, current_size):
            nonlocal best_size
            
            # Upper bound: current_size + popcount(candidates)
            if current_size + bin(candidates).count('1') <= best_size:
                return
            
            if candidates == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pick a vertex with maximum degree in the candidate subgraph
            # to maximize pruning
            max_deg = -1
            v = -1
            c = candidates
            while c:
                lb = c & (-c)
                idx = (lb.bit_length() - 1)
                # Count neighbors in candidates
                deg = bin(adj[idx] & candidates).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                c -= lb  # clear lowest set bit
            
            if v == -1:
                return
            
            v_bit = 1 << v
            
            # Branch 1: Exclude v
            search(candidates & ~v_bit, current_size)
            
            # Branch 2: Include v
            # Neighbors of v cannot be included
            new_cands = (candidates & ~v_bit) & ~adj[v]
            search(new_cands, current_size + 1)
        
        search((1 << n) - 1, 0)
        return best_size
    
    def generate_packing():
        random.shuffle(all_triples)
        used_pairs = set()
        triples = []
        masks = []
        
        for mask, p1, p2, p3, pts in all_triples:
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
            triples.append(pts)
            masks.append(mask)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples, masks, len(triples)
    
    while time.time() - start_time < timeout:
        triples, masks, size = generate_packing()
        
        # We prioritize maximal packings (size 37)
        if size < 37:
            continue
            
        alpha = calc_alpha(masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            
            if best_alpha <= 4:
                break
    
    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples
