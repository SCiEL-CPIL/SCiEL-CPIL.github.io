def branch_and_bound(candidates_mask, current_size, current_set):
    nonlocal best_size, best_set
    
    if candidates_mask == 0:
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
        return
    
    cb = color_bound(candidates_mask)
    if current_size + cb <= best_size:
        return
    
    vertices = []
    temp = candidates_mask
    while temp:
        v = (temp & -temp).bit_length() - 1
        vertices.append(v)
        temp &= temp - 1
        
    vertices.sort(key=lambda v: bin(adj[v] & candidates_mask).count('1'), reverse=True)
    
    p = candidates_mask
    
    for v in vertices:
        if not (p & (1 << v)):
            continue
        
        current_set.append(v)
        
        mask_greater = full_mask & ~((1 << (v + 1)) - 1)
        new_mask = p & adj[v] & mask_greater
        
        branch_and_bound(new_mask, current_size + 1, current_set)
        current_set.pop()
        
        p &= ~(1 << v)
