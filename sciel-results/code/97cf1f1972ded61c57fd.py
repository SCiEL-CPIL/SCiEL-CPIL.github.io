import numpy as np
import random
import time

def construct():
    N = 50
    E = N * (N - 1) // 2
    
    # Precompute edge indices and endpoints
    edge_idx = np.full((N, N), -1, dtype=int)
    us = np.zeros(E, dtype=int)
    vs = np.zeros(E, dtype=int)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[i, j] = idx
            edge_idx[j, i] = idx
            us[idx] = i
            vs[idx] = j
            idx += 1
            
    # Precompute for each edge e, the list of other edges in triangles with it
    # For edge e=(u,v), neighbors are (u,w) and (v,w) for w!=u,v
    # We need to calculate delta E quickly.
    # Delta E for changing color of e from c_old to c_new:
    # For each w:
    #   e1 = (u,w), e2 = (v,w)
    #   If c(e1)==c_old and c(e2)==c_old: was_mono (contribution 1)
    #   If c(e1)==c_new and c(e2)==c_new: is_mono (contribution 1)
    #   Delta += is_mono - was_mono
    
    # To speed up, we can precompute for each edge e, the list of (e1, e2) pairs.
    # There are N-2 such pairs for each e.
    # Total pairs = E * (N-2) = 1225 * 48 = 58800.
    # Storing this is fine.
    
    tri_edges = []
    for e in range(E):
        u = us[e]
        v = vs[e]
        pairs = []
        for w in range(N):
            if w == u or w == v:
                continue
            e1 = edge_idx[u, w]
            e2 = edge_idx[v, w]
            pairs.append((e1, e2))
        tri_edges.append(pairs)

    def calc_energy(colors):
        # Sum of mono triangles
        # Each mono triangle is counted 3 times (once for each edge).
        # So Energy = sum(conflict_counts) / 3
        # But we can just count directly.
        energy = 0
        for e in range(E):
            c = colors[e]
            for e1, e2 in tri_edges[e]:
                if colors[e1] == c and colors[e2] == c:
                    energy += 1
        return energy // 3 # Each triangle counted 3 times

    def calc_delta(colors, e, c_old, c_new):
        delta = 0
        for e1, e2 in tri_edges[e]:
            c1 = colors[e1]
            c2 = colors[e2]
            was_mono = (c1 == c_old and c2 == c_old)
            is_mono = (c1 == c_new and c2 == c_new)
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    start_time = time.time()
    time_limit = 115 # Leave 5s for safety
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.9999 # Adjust based on iterations
    
    best_solution = None
    best_energy = float('inf')
    
    while time.time() - start_time < time_limit:
        # Initialize
        colors = np.random.randint(0, 4, size=E)
        energy = calc_energy(colors)
        
        if energy == 0:
            return {'n': N, 'colors': [int(c) for c in colors]}
            
        if energy < best_energy:
            best_energy = energy
            best_solution = colors.copy()
            
        T = T_start
        no_improve_count = 0
        
        # Inner SA Loop
        while T > T_end:
            if time.time() - start_time > time_limit:
                break
                
            e = np.random.randint(E)
            c_old = colors[e]
            
            # Find best color
            best_c = c_old
            best_delta = 0
            for c_new in range(4):
                if c_new == c_old:
                    continue
                d = calc_delta(colors, e, c_old, c_new)
                if d < best_delta:
                    best_delta = d
                    best_c = c_new
                    
            if best_c == c_old:
                # No improvement, maybe random kick?
                # In SA, we can still accept worse moves.
                # If best_delta is 0, we might still want to move to escape plateau.
                # Let's pick a random other color if best_delta >= 0 and random says so?
                # Standard SA: Pick a random neighbor? Or best neighbor?
                # Usually best neighbor is faster.
                # If best_delta > 0, we accept with prob exp(-d/T).
                # If best_delta == 0, we accept with prob 1 (or 0.5 to encourage movement).
                
                # Let's try a random color if stuck at 0 delta to ensure mixing
                if best_delta >= 0 and random.random() < 0.1:
                    best_c = (c_old + 1) % 4
                    best_delta = calc_delta(colors, e, c_old, best_c)
                
            # Acceptance
            if best_delta < 0:
                accept = True
            else:
                # Accept with probability exp(-delta / T)
                if best_delta == 0:
                    accept = True # Encourage movement
                else:
                    accept = random.random() < np.exp(-best_delta / T)
            
            if accept:
                colors[e] = best_c
                energy += best_delta
                
                if energy == 0:
                    return {'n': N, 'colors': [int(c) for c in colors]}
                
                if energy < best_energy:
                    best_energy = energy
                    best_solution = colors.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # If stuck for too long, restart
                if no_improve_count > 5000:
                    break
            else:
                no_improve_count += 1
                if no_improve_count > 5000:
                    break
                    
            T *= cooling_rate
            
    # If best_solution is valid, return it
    if best_solution is not None and calc_energy(best_solution) == 0:
        return {'n': N, 'colors': [int(c) for c in best_solution]}
        
    # Fallback to 32
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
    return {'n': 32, 'colors': colors}
