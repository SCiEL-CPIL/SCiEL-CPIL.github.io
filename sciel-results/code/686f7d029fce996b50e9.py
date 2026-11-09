import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Seed: 32 vertices
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    n = 32
    best_n = 32
    best_colors = colors[:]
    
    # Bitmask neighbors
    # neighbors_mask[k][i] is an integer bitmask of vertices j such that color(i,j) == k
    neighbors_mask = [[0] * n for _ in range(4)]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            neighbors_mask[c][i] |= (1 << j)
            neighbors_mask[c][j] |= (1 << i)
            idx += 1
            
    # Backtracking function to add one vertex
    def add_vertex(n, neighbors_mask, colors):
        # We want to assign colors to (n, 0), ..., (n, n-1)
        # assigned[i] is the color of edge (n, i)
        assigned = [-1] * n
        
        # Precompute for each vertex i, the bitmask of neighbors for each color
        # This is already in neighbors_mask.
        
        # Heuristic: MRV
        # We will use a recursive search with time limit.
        
        def backtrack(k):
            if time.time() > deadline:
                return False, None
            
            if k == n:
                return True, assigned[:]
            
            # Determine available colors for vertex k
            # A color c is forbidden if there exists j < k such that assigned[j] == c and (1 << k) & neighbors_mask[c][j]
            available = [0, 1, 2, 3]
            for c in available[:]:
                forbidden = False
                for j in range(k):
                    if assigned[j] == c:
                        if (1 << k) & neighbors_mask[c][j]:
                            forbidden = True
                            break
                if forbidden:
                    available.remove(c)
            
            if not available:
                return False, None
            
            # Randomize order?
            random.shuffle(available)
            
            for c in available:
                assigned[k] = c
                success, res = backtrack(k + 1)
                if success:
                    return True, res
                assigned[k] = -1
            
            return False, None

        return backtrack(0)

    # Try to extend
    while time.time() < deadline:
        # Try to add vertex n
        # We might need to restart if we fail
        # Let's try a few times with different random shuffles
        
        success, assigned = add_vertex(n, neighbors_mask, colors)
        
        if success:
            # Add to colors
            colors.extend(assigned)
            n += 1
            best_n = n
            best_colors = colors[:]
            
            # Update neighbors_mask
            new_idx = n - 1
            for c in range(4):
                neighbors_mask[c].append(0)
            
            for j in range(new_idx):
                c = assigned[j]
                neighbors_mask[c][j] |= (1 << new_idx)
                neighbors_mask[c][new_idx] |= (1 << j)
            
            # Continue
            continue
        else:
            # Failed
            # We can try to restart with a slightly different seed?
            # Or just break.
            # Given the hardness, we might not go much further.
            break

    return {'n': best_n, 'colors': best_colors}
