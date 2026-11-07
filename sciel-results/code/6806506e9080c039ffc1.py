import time
import random
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Precompute adjacency bitmasks for fast degree calculation
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_score = 0
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # Helper to calculate distinct degrees for a mask
    def calc_score(mask):
        if mask.bit_count() < min_v:
            return 0
        degrees = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg_mask = adj[v] & mask
            deg = deg_mask.bit_count()
            degrees.add(deg)
            m ^= lsb
        return len(degrees)
    
    # Simulated Annealing
    # We will run multiple restarts
    num_restarts = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with a random subset of size between min_v and n
        size = random.randint(min_v, n)
        vertices = list(range(n))
        random.shuffle(vertices)
        mask = 0
        for i in range(size):
            mask |= (1 << vertices[i])
            
        current_score = calc_score(mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
            
        # If we hit the theoretical max for this size, or global max, we can stop early
        # Max distinct degrees is size. If size == n, max is n.
        if best_score == n:
            break
            
        temp = 10.0
        min_temp = 0.01
        cooling = 0.95
        
        while temp > min_temp:
            if time.time() > deadline:
                break
                
            # Generate a neighbor
            # Strategy: Swap a vertex in the set with one outside, or add/remove if size allows
            # To keep size >= min_v, we can:
            # 1. If size > min_v, try removing a vertex.
            # 2. If size < n, try adding a vertex.
            # 3. Try swapping.
            
            in_mask = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                in_mask.append(v)
                m ^= lsb
                
            out_mask = ((1 << n) - 1) ^ mask
            out_vertices = []
            m = out_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                out_vertices.append(v)
                m ^= lsb
                
            if not in_mask or not out_vertices:
                # Cannot move if set is full or empty (though min_v ensures not empty)
                # If full, only remove. If empty (impossible), only add.
                if in_mask and out_vertices:
                    break # Should not happen for valid min_v < n
            
            move_type = random.choice(['swap', 'add', 'remove'])
            
            if move_type == 'swap' and in_mask and out_vertices:
                v_in = random.choice(in_mask)
                v_out = random.choice(out_vertices)
                new_mask = mask ^ (1 << v_in) ^ (1 << v_out)
            elif move_type == 'add' and out_vertices and mask.bit_count() < n:
                v_out = random.choice(out_vertices)
                new_mask = mask | (1 << v_out)
            elif move_type == 'remove' and in_mask and mask.bit_count() > min_v:
                v_in = random.choice(in_mask)
                new_mask = mask & ~(1 << v_in)
            else:
                # Fallback to swap if possible, else skip
                if in_mask and out_vertices:
                    v_in = random.choice(in_mask)
                    v_out = random.choice(out_vertices)
                    new_mask = mask ^ (1 << v_in) ^ (1 << v_out)
                else:
                    continue

            new_score = calc_score(new_mask)
            
            if new_score >= current_score:
                mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                    if best_score == n:
                        break
            else:
                # Accept with probability exp(-delta/T)
                delta = current_score - new_score
                if delta > 0:
                    prob = pow(2.718281828, -delta / temp)
                    if random.random() < prob:
                        mask = new_mask
                        current_score = new_score
            
            temp *= cooling
            
        if best_score == n:
            break

    # Convert best_mask back to list of vertices
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
        
    # Ensure we return a valid list of at least min_v vertices
    # If best_mask is 0 (shouldn't happen), fallback to first min_v vertices
    if len(result) < min_v:
        result = list(range(min_v))
        
    return result
