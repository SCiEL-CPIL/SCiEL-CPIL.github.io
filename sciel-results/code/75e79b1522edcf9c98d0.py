import time
import random
import itertools
import sys

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute Q3 edge masks for a fixed set of 8 vertices (0..7)
    # Standard Q3 edges:
    # 0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
    base_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # Function to convert edge list to 28-bit mask
    # Pairs: (0,1), (0,2), ..., (0,7), (1,2), ...
    # Index calculation:
    # i < j
    # idx = i * 7 - (i * (i + 1) // 2) + j - i - 1 ?
    # Let's just use a dictionary or a formula.
    # Total pairs = 28.
    # Map (u, v) to bit index.
    pair_to_bit = {}
    bit = 0
    for u in range(8):
        for v in range(u+1, 8):
            pair_to_bit[(u, v)] = bit
            pair_to_bit[(v, u)] = bit
            bit += 1
            
    def edges_to_mask(edges):
        mask = 0
        for u, v in edges:
            mask |= (1 << pair_to_bit[(u, v)])
        return mask
            
    # Generate all isomorphic Q3 masks
    q3_masks = set()
    # Permutations of 0..7
    for perm in itertools.permutations(range(8)):
        # perm[i] is the new label for old label i?
        # Or old label i maps to perm[i]?
        # Let's say new_label = perm[old_label]
        new_edges = []
        for u, v in base_edges:
            nu = perm[u]
            nv = perm[v]
            if nu > nv:
                nu, nv = nv, nu
            new_edges.append((nu, nv))
        q3_masks.add(edges_to_mask(new_edges))
        
    # Precompute all 8-vertex subsets and their edge bit mappings
    subsets_8 = []
    for combo in itertools.combinations(range(N), 8):
        # combo is a tuple of 8 vertices
        # We need to map these 8 vertices to 0..7 to use the q3_masks
        # And we need to know which bit in the 28-bit mask corresponds to which edge in the global graph
        # Actually, we can just compute the 28-bit mask for the induced subgraph on the fly.
        # For a subset S, we iterate over all pairs (u, v) in S.
        # There are 28 pairs.
        # We check if edge (u, v) is in the graph.
        # We set the bit corresponding to the local index of (u, v).
        
        # To speed up, precompute the list of pairs and their local bit indices for each subset?
        # Or just compute on the fly.
        subsets_8.append(combo)
        
    best_edges = 0
    best_adj = None
    
    def edge_count(adj):
        count = 0
        for i in range(N):
            count += adj[i].bit_count()
        return count // 2
        
    def has_q3(adj):
        """
        Check if graph adj contains Q3.
        Returns True if yes, False if no.
        Also returns one Q3 mask if found?
        For the removal strategy, we need to find all or many.
        """
        found = []
        for combo in subsets_8:
            # Compute 28-bit mask of induced subgraph
            local_mask = 0
            # Map combo vertices to 0..7
            # We need to iterate pairs in combo
            # And check adj
            # And set bit in local_mask
            # The bit index in local_mask must match the index in pair_to_bit
            # pair_to_bit is defined for (0..7)
            # So we need to map combo[i] to i?
            # No, the isomorphism check requires that the *structure* matches.
            # The labels in combo are arbitrary.
            # So we should map combo[0]->0, combo[1]->1, etc.
            # Then the local_mask is computed based on this mapping.
            # Then we check if local_mask is in q3_masks.
            
            # Mapping: combo[i] -> i
            # Pairs (i, j) in 0..7
            # Edge (combo[i], combo[j]) in global graph?
            # If yes, set bit pair_to_bit[(i, j)]
            
            m = 0
            for i in range(8):
                for j in range(i+1, 8):
                    u = combo[i]
                    v = combo[j]
                    if (adj[u] >> v) & 1:
                        m |= (1 << pair_to_bit[(i, j)])
                        
            if m in q3_masks:
                found.append(combo)
                
        return found

    # Restart loop
    while time.time() < deadline:
        # Init K16
        adj = [0] * N
        for i in range(N):
            mask = 0
            for j in range(N):
                if i != j:
                    mask |= (1 << j)
            adj[i] = mask
        
        current_adj = adj[:]
        current_q3s = has_q3(current_adj)
        
        while time.time() < deadline:
            if not current_q3s:
                break
            
            # Count edge frequencies
            edge_freq = {}
            for combo in current_q3s:
                for i in range(8):
                    for j in range(i+1, 8):
                        u = combo[i]
                        v = combo[j]
                        if (current_adj[u] >> v) & 1:
                            key = (min(u, v), max(u, v))
                            edge_freq[key] = edge_freq.get(key, 0) + 1
                            
            if not edge_freq:
                break
                
            sorted_edges = sorted(edge_freq.items(), key=lambda x: x[1], reverse=True)
            
            if random.random() < 0.2:
                combo = random.choice(current_q3s)
                edges_in_q3 = []
                for i in range(8):
                    for j in range(i+1, 8):
                        u = combo[i]
                        v = combo[j]
                        if (current_adj[u] >> v) & 1:
                            edges_in_q3.append((min(u, v), max(u, v)))
                if edges_in_q3:
                    u, v = random.choice(edges_in_q3)
                else:
                    u, v = sorted_edges[0][0]
            else:
                u, v = sorted_edges[0][0]
                
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
            current_q3s = has_q3(current_adj)
            
        ec = edge_count(current_adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = current_adj[:]
            
        if time.time() > deadline:
            break
            
    edges = []
    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    for i in range(N):
        m = best_adj[i]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if i < v:
                edges.append([i, v])
            m ^= lsb
            
    return edges

