import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to find a valid coloring for n=34 down to n=1
    # The known lower bound is 34.
    
    # First, try to find a valid coloring for n=34
    success, colors = find_valid_coloring(34, deadline)
    if success:
        return {'n': 34, 'colors': colors}
    
    # If we couldn't find n=34, try smaller values
    for n in range(33, 0, -1):
        if time.time() > deadline - 10:
            break
        success, colors = find_valid_coloring(n, deadline)
        if success:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a valid small coloring if we ran out of time
    # n=9 from seed is valid.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def find_valid_coloring(n, deadline):
    # Precompute structures for fast energy calculation
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s as lists of triple indices
    k4s = []
    for quad in combinations(range(n), 4):
        idxs = [triple_index[t] for t in combinations(quad, 3)]
        k4s.append(idxs)
        
    k5s = []
    for quint in combinations(range(n), 5):
        idxs = [triple_index[t] for t in combinations(quint, 3)]
        k5s.append(idxs)
    
    num_k4s = len(k4s)
    num_k5s = len(k5s)
    
    # For each triple, precompute which K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for ki, idxs in enumerate(k4s):
        for idx in idxs:
            triple_to_k4s[idx].append(ki)
            
    for ki, idxs in enumerate(k5s):
        for idx in idxs:
            triple_to_k5s[idx].append(ki)
    
    # Use local search with repair strategy
    # Start with a random coloring and iteratively flip triples to reduce violations
    
    best_energy = float('inf')
    best_colors = None
    
    max_iterations = 1000000
    
    # Use numpy for faster operations
    # We'll use a simpler approach: just track the counts
    
    while time.time() < deadline - 2:
        # Random initialization
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Compute initial energy and red/blue counts per K4 and K5
        k4_red_count = np.zeros(num_k4s, dtype=np.int32)
        k5_blue_count = np.zeros(num_k5s, dtype=np.int32)
        
        for ki, idxs in enumerate(k4s):
            for idx in idxs:
                if colors[idx] == 0:
                    k4_red_count[ki] += 1
                    
        for ki, idxs in enumerate(k5s):
            for idx in idxs:
                if colors[idx] == 1:
                    k5_blue_count[ki] += 1
        
        # Initial energy: number of K4s with all 4 red + number of K5s with all 10 blue
        energy = np.sum(k4_red_count == 4) + np.sum(k5_blue_count == 10)
        
        if energy == 0:
            return True, colors.tolist()
            
        if energy < best_energy:
            best_energy = energy
            best_colors = colors.copy()
        
        # Local search: iteratively flip triples to reduce violations
        # We'll use a greedy repair strategy
        no_improvement = 0
        max_no_improvement = 5000
        temperature = 1.0
        cooling = 0.999
        
        while no_improvement < max_no_improvement and time.time() < deadline - 2:
            # Pick a random triple to flip
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Calculate delta energy
            delta = 0
            
            # Update K4s containing triple i
            for ki in triple_to_k4s[i]:
                old_red = k4_red_count[ki]
                if old_color == 0:
                    # Was red, now blue: red count decreases
                    new_red = old_red - 1
                    if old_red == 4:
                        delta += 1  # Was violating, now not
                    if new_red == 4:
                        delta -= 1  # Was not violating, now is
                    k4_red_count[ki] = new_red
                else:
                    # Was blue, now red: red count increases
                    new_red = old_red + 1
                    if old_red == 4:
                        delta -= 1  # Was violating, now not
                    if new_red == 4:
                        delta += 1  # Was not violating, now is
                    k4_red_count[ki] = new_red
                    
            # Update K5s containing triple i
            for ki in triple_to_k5s[i]:
                old_blue = k5_blue_count[ki]
                if old_color == 1:
                    # Was blue, now red: blue count decreases
                    new_blue = old_blue - 1
                    if old_blue == 10:
                        delta += 1  # Was violating, now not
                    if new_blue == 10:
                        delta -= 1  # Was not violating, now is
                    k5_blue_count[ki] = new_blue
                else:
                    # Was red, now blue: blue count increases
                    new_blue = old_blue + 1
                    if old_blue == 10:
                        delta -= 1  # Was violating, now not
                    if new_blue == 10:
                        delta += 1  # Was not violating, now is
                    k5_blue_count[ki] = new_blue
            
            energy += delta
            
            if energy < 0:
                energy = 0  # Shouldn't happen but safety
                
            # Accept or reject using simulated annealing
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 0.001)):
                colors[i] = new_color
                if delta < 0:
                    no_improvement = 0
                    if energy < best_energy:
                        best_energy = energy
                        best_colors = colors.copy()
                else:
                    no_improvement += 1
            else:
                # Reject: revert the counts
                for ki in triple_to_k4s[i]:
                    if old_color == 0:
                        k4_red_count[ki] += 1
                    else:
                        k4_red_count[ki] -= 1
                for ki in triple_to_k5s[i]:
                    if old_color == 1:
                        k5_blue_count[ki] += 1
                    else:
                        k5_blue_count[ki] -= 1
                energy -= delta
                no_improvement += 1
            
            if energy == 0:
                return True, colors.tolist()
                
            temperature *= cooling
        
        # After local search, check if we found a solution
        if energy == 0:
            return True, colors.tolist()
            
    # If we didn't find 0, return False
    return False, None
