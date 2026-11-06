import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_c4_free(adj, n):
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if popcount(ai & adj[j]) > 1:
                    return False
        return True

    def max_independent_set(adj, n):
        best = 0
        # Use a list for adj for faster access in recursion? 
        # Bitmask is fine.
        
        def backtrack(remaining_mask, current_size):
            nonlocal best
            if current_size + popcount(remaining_mask) <= best:
                return
            if remaining_mask == 0:
                best = max(best, current_size)
                return
            
            # Heuristic: Pick vertex with max degree in remaining
            max_deg = -1
            v = -1
            temp = remaining_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                deg = popcount(adj[idx] & remaining_mask)
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp ^= lsb
            
            if v == -1:
                return
                
            # Branch 1: Include v
            new_remaining = remaining_mask & ~(1 << v) & ~adj[v]
            backtrack(new_remaining, current_size + 1)
            
            # Branch 2: Exclude v
            new_remaining2 = remaining_mask & ~(1 << v)
            backtrack(new_remaining2, current_size)
            
        backtrack((1 << n) - 1, 0)
        return best

    # Initial construction: Polarity graph of PG(2,5)
    def gf5_inv(a):
        if a == 0: return 0
        for x in range(1, 5):
            if (a * x) % 5 == 1:
                return x
        return 0

    def canonicalize(v):
        for i in range(3):
            if v[i] != 0:
                inv = gf5_inv(v[i])
                return tuple((v[j] * inv) % 5 for j in range(3))
        return (0, 0, 0)

    points = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0: continue
                pt = canonicalize((a, b, c))
                if pt not in points:
                    points.append(pt)
    
    n_pts = len(points)
    adj = [0] * n_pts
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            u = points[i]
            v = points[j]
            b_val = (u[0]*v[1] + u[1]*v[2] + u[2]*v[0]) % 5
            if b_val == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    current_adj = adj
    current_n = n_pts
    best_adj = current_adj[:]
    best_n = current_n
    
    def try_add_vertex(adj, n):
        # Build conflict graph
        conflict = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if popcount(adj[i] & adj[j]) > 0:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        # Find independent sets in conflict graph
        # We want a large one.
        # Greedy with randomization
        best_set = 0
        for _ in range(5): # Try 5 times
            remaining = (1 << n) - 1
            chosen = 0
            while remaining:
                # Pick random vertex from remaining, or min degree?
                # Min degree is better for size
                min_deg = n
                v = -1
                temp = remaining
                # Add some randomness
                if random.random() < 0.3:
                    # Pick random
                    temp_list = []
                    t = remaining
                    while t:
                        lsb = t & -t
                        idx = lsb.bit_length() - 1
                        temp_list.append(idx)
                        t ^= lsb
                    if not temp_list: break
                    v = random.choice(temp_list)
                else:
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        deg = popcount(conflict[idx] & remaining)
                        if deg < min_deg:
                            min_deg = deg
                            v = idx
                        temp ^= lsb
                if v == -1: break
                chosen |= (1 << v)
                remaining &= ~(1 << v)
                remaining &= ~conflict[v]
            if popcount(chosen) > popcount(best_set):
                best_set = chosen
                
        if best_set == 0:
            return None, None
            
        new_adj = adj[:] + [0]
        new_adj[n] = best_set
        for i in range(n):
            if best_set & (1 << i):
                new_adj[i] |= (1 << n)
                
        if not is_c4_free(new_adj, n + 1):
            return None, None
            
        return new_adj, n + 1

    while time.time() < deadline:
        new_adj, new_n = try_add_vertex(current_adj, current_n)
        if new_n is not None:
            if max_independent_set(new_adj, new_n) < 11:
                current_adj = new_adj
                current_n = new_n
                if current_n > best_n:
                    best_n = current_n
                    best_adj = current_adj[:]
                continue
        
        # If stuck, try to remove an edge to potentially allow future growth
        # or just restart from best
        if random.random() < 0.1:
            # Restart from best
            current_adj = best_adj[:]
            current_n = best_n
            
        # Try removing a random edge
        if current_n > 10:
            edges = []
            for i in range(current_n):
                mask = current_adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append((i, j))
                    mask ^= lsb
            
            if edges:
                i, j = random.choice(edges)
                new_adj = current_adj[:]
                new_adj[i] &= ~(1 << j)
                new_adj[j] &= ~(1 << i)
                
                if max_independent_set(new_adj, current_n) < 11:
                    current_adj = new_adj
                    # Don't update best_n as n is same
                    # But this might help in adding vertices later

    # Final verification
    if max_independent_set(best_adj, best_n) < 11 and is_c4_free(best_adj, best_n):
        edges = []
        for i in range(best_n):
            mask = best_adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return {'n': best_n, 'edges': edges}
    else:
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
