import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Precompute nCr for index calculation
    # We need comb(n, k) for k=2,3 up to n=40
    # But we will generate indices for each n
    
    def get_triple_index(a, b, c):
        return comb(c, 3) + comb(b, 2) + a

    def search_for_n(n, time_limit):
        M = comb(n, 3)
        # Generate all triples
        triples = list(combinations(range(n), 3))
        
        # Map triple to index
        # We can just use the index in the list
        # But for cost calculation, we need indices of triples in 4-sets and 5-sets
        
        # Precompute indices for 4-sets
        # For each 4-set, get the 4 triples
        triples_in_4sets = []
        for quad in combinations(range(n), 4):
            t_indices = []
            for i in range(4):
                # Remove quad[i] to get the triple
                tri = tuple(x for j, x in enumerate(quad) if j != i)
                # Find index of tri in triples
                # This is slow. Better to map triple to index.
                pass
            
        # Better: Create a dictionary mapping triple tuple to index
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}
        
        triples_in_4sets = []
        for quad in combinations(range(n), 4):
            t_indices = []
            for i in range(4):
                tri = tuple(x for j, x in enumerate(quad) if j != i)
                t_indices.append(tri_to_idx[tri])
            triples_in_4sets.append(t_indices)
            
        triples_in_5sets = []
        for quint in combinations(range(n), 5):
            t_indices = []
            for tri in combinations(quint, 3):
                t_indices.append(tri_to_idx[tri])
            triples_in_5sets.append(t_indices)
            
        triples_in_4sets = np.array(triples_in_4sets, dtype=int)
        triples_in_5sets = np.array(triples_in_5sets, dtype=int)
        
        # Initial random coloring
        colors = np.random.randint(0, 2, size=M)
        
        # Calculate initial cost
        def calc_cost(colors):
            c4 = colors[triples_in_4sets]
            c5 = colors[triples_in_5sets]
            red_k4 = np.all(c4 == 0, axis=1)
            blue_k5 = np.all(c5 == 1, axis=1)
            return np.sum(red_k4) + np.sum(blue_k5)
        
        current_cost = calc_cost(colors)
        
        if current_cost == 0:
            return colors.tolist()
            
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() < start_time + time_limit:
            # Pick random triple
            t_idx = np.random.randint(0, M)
            
            # Flip
            old_val = colors[t_idx]
            colors[t_idx] = 1 - old_val
            
            new_cost = calc_cost(colors)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    return colors.tolist()
            else:
                # Simulated annealing accept
                if T > min_T:
                    delta = new_cost - current_cost
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_cost = new_cost
                else:
                    # Revert
                    colors[t_idx] = old_val
            
            T *= cooling
            
        return None

    # Try increasing n
    for n in range(10, 35):
        # Allocate time based on remaining time
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Smaller n gets less time? Or fixed time?
        # Let's give 2 seconds for n=10, 3 for n=11, etc?
        # Or just use remaining time / (34 - n + 1)
        # Actually, if we fail for n, we shouldn't waste too much time.
        # Let's try 5 seconds per n, but stop if we find a solution.
        # If we find a solution for n, we move to n+1.
        
        t_limit = min(5.0, remaining - 1.0)
        if t_limit <= 0:
            break
            
        res = search_for_n(n, t_limit)
        if res is not None:
            best_n = n
            best_colors = res
            # If we found a solution, we can try to find a better one for same n?
            # No, we want larger n.
            
    return {'n': best_n, 'colors': best_colors}
