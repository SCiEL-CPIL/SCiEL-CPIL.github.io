import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(9), 3)]
    
    for n in range(10, 41):
        if time.time() > deadline - 15.0:
            break
            
        # Precompute indices for this n
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        tri_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_indices = []
        for quad in itertools.combinations(range(n), 4):
            inds = []
            for i in range(4):
                t = tuple(sorted(quad[:i] + quad[i+1:]))
                inds.append(tri_to_idx[t])
            k4_indices.append(inds)
        k4_indices = np.array(k4_indices, dtype=int) if k4_indices else np.array([], dtype=int).reshape(0, 4)
        
        k5_indices = []
        for quint in itertools.combinations(range(n), 5):
            inds = []
            for tri in itertools.combinations(quint, 3):
                inds.append(tri_to_idx[tri])
            k5_indices.append(inds)
        k5_indices = np.array(k5_indices, dtype=int) if k5_indices else np.array([], dtype=int).reshape(0, 10)
        
        if time.time() > deadline - 10.0:
            break

        def get_cost(colors):
            cost = 0
            if len(k4_indices) > 0:
                c4 = colors[k4_indices]
                cost += np.sum(c4.sum(axis=1) == 0)
            if len(k5_indices) > 0:
                c5 = colors[k5_indices]
                cost += np.sum(c5.sum(axis=1) == 10)
            return cost

        found = False
        for restart in range(100):
            if time.time() > deadline - 5.0:
                break
                
            colors = np.random.randint(0, 2, size=num_triples)
            current_cost = get_cost(colors)
            
            if current_cost == 0:
                found = True
                best_n = n
                best_colors = colors.tolist()
                break
            
            for step in range(1000):
                if time.time() > deadline - 5.0:
                    break
                    
                if current_cost == 0:
                    found = True
                    best_n = n
                    best_colors = colors.tolist()
                    break
                
                idx = random.randint(0, num_triples - 1)
                old_val = colors[idx]
                colors[idx] = 1 - old_val
                
                new_cost = get_cost(colors)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                else:
                    colors[idx] = old_val
                    break
            
            if found:
                break

    return {'n': best_n, 'colors': best_colors}
