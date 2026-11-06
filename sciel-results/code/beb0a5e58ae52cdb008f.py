import time
import random
from functools import lru_cache

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        
        # Find vertices that are part of any edge
        active_mask = 0
        for v in range(n):
            if adj[v]:
                active_mask |= (1 << v)
        
        if active_mask == 0:
            return []

        # Branch and Bound for Minimum Vertex Cover
        best_size = n # Upper bound: all vertices
        best_cover = [0] * n # 1 if in cover
        
        # Greedy matching lower bound
        def greedy_matching_size(mask):
            # Returns size of a maximal matching in the subgraph induced by mask
            # This is a lower bound on the vertex cover size
            remaining = mask
            match_count = 0
            while remaining:
                # Pick a vertex with highest degree in remaining
                best_v = -1
                best_deg = -1
                temp = remaining
                while temp:
                    v_bit = temp & -temp
                    v = v_bit.bit_length() - 1
                    deg = (adj[v] & remaining).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    temp ^= v_bit
                
                if best_v == -1:
                    break
                
                # Find a neighbor
                neighbors = adj[best_v] & remaining
                if not neighbors:
                    # Isolated vertex in remaining, can't match
                    remaining ^= (1 << best_v)
                    continue
                
                # Pick first neighbor
                n_bit = neighbors & -neighbors
                n_v = n_bit.bit_length() - 1
                
                match_count += 1
                # Remove both vertices from remaining
                remaining ^= (1 << best_v)
                remaining ^= (1 << n_v)
            
            return match_count

        def bb(mask, current_size, current_cover):
            nonlocal best_size, best_cover
            
            # Lower bound
            lb = greedy_matching_size(mask)
            if current_size + lb >= best_size:
                return
            
            # If no edges left
            # Check if mask has any edges
            has_edge = False
            temp = mask
            while temp:
                v_bit = temp & -temp
                v = v_bit.bit_length() - 1
                if adj[v] & mask:
                    has_edge = True
                    break
                temp ^= v_bit
            
            if not has_edge:
                if current_size < best_size:
                    best_size = current_size
                    best_cover = current_cover[:]
                return
            
            # Pick vertex with max degree
            best_v = -1
            best_deg = -1
            temp = mask
            while temp:
                v_bit = temp & -temp
                v = v_bit.bit_length() - 1
                deg = (adj[v] & mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= v_bit
            
            if best_v == -1:
                return
                
            v_bit = 1 << best_v
            
            # Branch 1: Include best_v in cover
            new_mask_1 = mask & ~v_bit
            current_cover[best_v] = 1
            bb(new_mask_1, current_size + 1, current_cover)
            current_cover[best_v] = 0
            
            # Branch 2: Exclude best_v, include all neighbors
            neighbors = adj[best_v] & mask
            # The neighbors to add are those in mask
            # We add them to the cover
            # New mask: remove best_v and all its neighbors
            new_mask_2 = mask & ~v_bit & ~neighbors
            
            # Add neighbors to cover
            added = 0
            temp = neighbors
            while temp:
                n_bit = temp & -temp
                n_v = n_bit.bit_length() - 1
                current_cover[n_v] = 1
                added += 1
                temp ^= n_bit
                
            bb(new_mask_2, current_size + added, current_cover)
            
            # Undo adding neighbors
            temp = neighbors
            while temp:
                n_bit = temp & -temp
                n_v = n_bit.bit_length() - 1
                current_cover[n_v] = 0
                temp ^= n_bit

        # Initial call
        bb(active_mask, 0, [0] * n)
        
        return [i for i in range(n) if best_cover[i] == 1]

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        results.append(solve_instance(n, p, seed))
    return results
