import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    squares = {i*i for i in range(2, int((2*N)**0.5) + 2)}
    
    # 1. Filter valid vertices (2v not square)
    valid_verts = []
    for v in range(1, N+1):
        if 2*v not in squares:
            valid_verts.append(v)
            
    M = len(valid_verts)
    if M == 0:
        return [1]
        
    # Map to indices
    idx_map = {v: i for i, v in enumerate(valid_verts)}
    
    # 2. Build adjacency bitsets
    adj = [0] * M
    for i in range(M):
        vi = valid_verts[i]
        mask = 0
        for j in range(M):
            if i == j: continue
            vj = valid_verts[j]
            if vi + vj in squares:
                mask |= (1 << j)
        adj[i] = mask
        
    # 3. Greedy Initialization
    # Sort by degree ascending (pick least constrained first)
    degs = [adj[i].bit_count() for i in range(M)]
    order_greedy = sorted(range(M), key=lambda x: degs[x])
    
    greedy_mask = 0
    greedy_size = 0
    available = (1 << M) - 1
    
    for v in order_greedy:
        if available & (1 << v):
            greedy_mask |= (1 << v)
            greedy_size += 1
            # Remove v and its neighbors from available
            available &= ~adj[v]
            available &= ~(1 << v)
            
    best_size = greedy_size
    best_set_mask = greedy_mask
    
    # 4. Branch and Bound
    # Order for branching: High degree first to prune faster
    order_branch = sorted(range(M), key=lambda x: degs[x], reverse=True)
    
    def search(current_mask, current_size, candidates_mask):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
            
        # Prune
        if current_size + candidates_mask.bit_count() <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Pick vertex
        v_idx = -1
        for v in order_branch:
            if candidates_mask & (1 << v):
                v_idx = v
                break
                
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Include
        new_cand_in = candidates_mask & ~adj[v_idx] & ~v_bit
        search(current_mask | v_bit, current_size + 1, new_cand_in)
        
        # Exclude
        new_cand_out = candidates_mask & ~v_bit
        search(current_mask, current_size, new_cand_out)

    search(0, 0, (1 << M) - 1)
    
    result = []
    for i in range(M):
        if best_set_mask & (1 << i):
            result.append(valid_verts[i])
            
    if not result:
        result = [valid_verts[0]]
        
    return result
