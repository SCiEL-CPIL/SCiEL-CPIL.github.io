import math

def construct():
    # Generate candidate vertices: numbers in [1, 255] with gcd(x, 255) > 1
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute adjacency masks
    # adj_mask[i] is an integer where bit j is 1 if candidates[i] and candidates[j] are connected
    adj_mask = [0] * n
    
    for i in range(n):
        vi = candidates[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                mask |= (1 << j)
        adj_mask[i] = mask
        
    # Calculate degrees (popcount of mask)
    # Using bit_count if available, else fallback
    try:
        degrees = [m.bit_count() for m in adj_mask]
    except AttributeError:
        degrees = [bin(m).count('1') for m in adj_mask]
        
    # Sort indices by degree descending
    sorted_indices = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    best_clique_size = 0
    best_clique_set = set()
    
    # Try starting from each candidate, prioritizing high degree
    for start_idx in sorted_indices:
        clique = {candidates[start_idx]}
        current_mask = adj_mask[start_idx]
        
        while current_mask:
            # Find vertex in current_mask with highest degree within current_mask
            best_v_idx = -1
            max_deg = -1
            
            # Iterate through set bits in current_mask
            m = current_mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                
                # Calculate degree of idx within current_mask
                try:
                    deg = (adj_mask[idx] & current_mask).bit_count()
                except AttributeError:
                    deg = bin(adj_mask[idx] & current_mask).count('1')
                
                if deg > max_deg:
                    max_deg = deg
                    best_v_idx = idx
                    
                m ^= lsb # Clear the bit
            
            if best_v_idx == -1:
                break
                
            clique.add(candidates[best_v_idx])
            current_mask = current_mask & adj_mask[best_v_idx]
            
        if len(clique) > best_clique_size:
            best_clique_size = len(clique)
            best_clique_set = clique
            
    # Ensure 255 is in the result
    if 255 not in best_clique_set:
        best_clique_set.add(255)
        
    return list(best_clique_set)
