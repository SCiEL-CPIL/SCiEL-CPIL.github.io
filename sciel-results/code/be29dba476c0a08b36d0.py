import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 256
    vertices = list(range(2, n + 1))
    N = len(vertices)
    
    # Map value to index for easier handling? 
    # Or just use values as indices in a dict?
    # Let's use 0..N-1 indices.
    # val[i] is the integer value.
    vals = vertices
    
    # Build complement graph (coprime graph) adjacency masks
    # adj_mask[i] is a bitmask of vertices j such that gcd(vals[i], vals[j]) == 1
    adj_mask = [0] * N
    
    for i in range(N):
        vi = vals[i]
        mask = 0
        for j in range(N):
            if i != j:
                if math.gcd(vi, vals[j]) == 1:
                    mask |= (1 << j)
        adj_mask[i] = mask
        
    # Order vertices by descending weight (value)
    # This helps the bound and the search
    order = sorted(range(N), key=lambda i: vals[i], reverse=True)
    
    # Re-index or just use the order in the B&B
    # We will maintain a list of candidate indices.
    
    best_sum = 1 # Include 1
    best_set = {1}
    
    # Initial greedy solution
    def greedy_solve(candidates_mask):
        s = 0
        c_mask = candidates_mask
        while c_mask:
            # Pick vertex with max weight
            best_v = -1
            best_w = -1
            temp = c_mask
            while temp:
                # Get lowest set bit
                lb = temp & -temp
                idx = lb.bit_length() - 1
                if vals[idx] > best_w:
                    best_w = vals[idx]
                    best_v = idx
                temp ^= lb
            
            if best_v == -1:
                break
                
            s += best_w
            # Remove best_v and its neighbors in G (non-coprimes)
            # Keep only coprime neighbors
            # New candidates = c_mask & adj_mask[best_v]
            c_mask = c_mask & adj_mask[best_v]
            c_mask &= ~(1 << best_v) # best_v is not coprime to itself, so it should be removed, but adj_mask doesn't contain self.
            
        return s

    # Initial bound
    all_mask = (1 << N) - 1
    init_sum = greedy_solve(all_mask)
    if init_sum > best_sum:
        best_sum = init_sum
        # Reconstruct set for best_sum if needed, but we only need the sum for the score.
        # However, we need to return the list.
        # Let's store the best set found.
        
    # We need to track the actual set to return it.
    # B&B with backtracking to keep track of the set.
    
    def bnb(candidates_mask, current_sum, current_set):
        nonlocal best_sum
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
            
        # Upper bound calculation
        # Greedy coloring of the subgraph induced by candidates in the coprime graph
        # We want to color such that adjacent nodes (coprime) have different colors.
        # Then a clique (independent set in conflict graph) can pick at most 1 from each color class.
        
        # To make coloring fast, we process vertices in order of decreasing weight?
        # Or just iterate through bits.
        # Let's collect vertices
        cand_list = []
        temp = candidates_mask
        while temp:
            lb = temp & -temp
            idx = lb.bit_length() - 1
            cand_list.append(idx)
            temp ^= lb
            
        # Sort by weight descending for better coloring bound?
        # Actually, the bound is sum of max weight in each color.
        # Coloring is greedy.
        
        colors = {}
        # We need to assign colors to cand_list
        # For each v in cand_list:
        #   Find the smallest color c such that no neighbor of v in cand_list has color c.
        
        # Optimization: If the number of candidates is small, this is fast.
        
        max_weights_per_color = {}
        
        for v in cand_list:
            # Find used colors by neighbors of v in cand_list
            # Neighbors of v in the subgraph are neighbors in adj_mask[v] that are also in candidates_mask
            neigh_mask = adj_mask[v] & candidates_mask
            used_colors = set()
            temp_neigh = neigh_mask
            while temp_neigh:
                lb = temp_neigh & -temp_neigh
                n_idx = lb.bit_length() - 1
                if n_idx in colors:
                    used_colors.add(colors[n_idx])
                temp_neigh ^= lb
                
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
                
            colors[v] = c
            if c not in max_weights_per_color:
                max_weights_per_color[c] = vals[v]
            else:
                if vals[v] > max_weights_per_color[c]:
                    max_weights_per_color[c] = vals[v]
                    
        bound = sum(max_weights_per_color.values())
        
        if current_sum + bound <= best_sum:
            return
            
        # Branch
        # Pick a vertex to branch on.
        # Heuristic: Pick the vertex with the highest weight? Or the one that appears first in the sorted order?
        # Let's pick the one with the highest weight in candidates.
        
        best_v = -1
        best_w = -1
        temp = candidates_mask
        while temp:
            lb = temp & -temp
            idx = lb.bit_length() - 1
            if vals[idx] > best_w:
                best_w = vals[idx]
                best_v = idx
            temp ^= lb
            
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # New candidates: neighbors of best_v in coprime graph (adj_mask[best_v]) intersected with current candidates
        # Note: best_v itself is removed.
        new_cand_include = candidates_mask & adj_mask[best_v]
        
        current_set.add(vals[best_v])
        bnb(new_cand_include, current_sum + vals[best_v], current_set)
        current_set.remove(vals[best_v])
        
        # Branch 2: Exclude best_v
        new_cand_exclude = candidates_mask & ~(1 << best_v)
        bnb(new_cand_exclude, current_sum, current_set)
        
    bnb(all_mask, 0, set())
    
    # best_set contains the optimal set of numbers from 2..256
    # Add 1
    result = list(best_set)
    result.append(1)
    
    # Ensure types are correct
    return [int(x) for x in result]
