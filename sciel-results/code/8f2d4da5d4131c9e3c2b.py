import time
import random
import math

def construct():
    N = 40
    M = 401
    deadline = time.time() + 140
    
    # Initialize with random graph
    masks = [0] * N
    
    # Generate random edges
    all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_possible)
    for u, v in all_possible[:M]:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        
    def get_edges(masks):
        edges = []
        for i in range(N):
            m = masks[i]
            for j in range(i+1, N):
                if (m >> j) & 1:
                    edges.append([i, j])
        return edges
        
    def get_cost(masks):
        max_b = 0
        sum_b = 0
        for i in range(N):
            m_i = masks[i]
            for j in range(i+1, N):
                if (m_i >> j) & 1:
                    b = (m_i & masks[j]).bit_count()
                    if b > max_b:
                        max_b = b
                    sum_b += b
        return max_b * 10000 + sum_b
        
    current_cost = get_cost(masks)
    best_cost = current_cost
    best_masks = masks[:]
    
    T = 10.0
    min_T = 0.01
    
    while time.time() < deadline:
        # Perform a swap move
        # 1. Pick random u, v
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u == v:
            continue
            
        exists_uv = (masks[u] >> v) & 1
        
        if exists_uv:
            # Remove (u, v), Add (x, y)
            # Remove
            masks[u] ^= (1 << v)
            masks[v] ^= (1 << u)
            
            # Find (x, y) to add
            while True:
                x = random.randint(0, N-1)
                y = random.randint(0, N-1)
                if x == y:
                    continue
                if not ((masks[x] >> y) & 1):
                    break
            # Add
            masks[x] |= (1 << y)
            masks[y] |= (1 << x)
        else:
            # Add (u, v), Remove (x, y)
            # Add
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
            # Find (x, y) to remove
            while True:
                x = random.randint(0, N-1)
                y = random.randint(0, N-1)
                if x == y:
                    continue
                if ((masks[x] >> y) & 1):
                    break
            # Remove
            masks[x] ^= (1 << y)
            masks[y] ^= (1 << x)
            
        new_cost = get_cost(masks)
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_masks = masks[:]
        else:
            # Revert the swap
            # This is tricky to revert exactly without storing state.
            # Easier: Just revert the specific changes.
            # If we did Remove(u,v) Add(x,y):
            # Revert: Add(u,v) Remove(x,y)
            # If we did Add(u,v) Remove(x,y):
            # Revert: Remove(u,v) Add(x,y)
            
            # We need to know what we did.
            # The logic above is deterministic based on `exists_uv`.
            # So we can just repeat the inverse operation.
            
            if exists_uv:
                # We removed u,v and added x,y.
                # Revert: remove x,y and add u,v.
                masks[x] ^= (1 << y)
                masks[y] ^= (1 << x)
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                # We added u,v and removed x,y.
                # Revert: remove u,v and add x,y.
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                masks[x] |= (1 << y)
                masks[y] |= (1 << x)
                
        T *= 0.999
        if T < min_T:
            T = 10.0
            
    return get_edges(best_masks)
