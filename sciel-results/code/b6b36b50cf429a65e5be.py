import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # The known lower bound is 34. Try to find valid colorings from 34 down.
    for n in range(34, 0, -1):
        if time.time() > deadline - 20:
            break
        success, colors = find_valid_coloring(n, deadline)
        if success:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a valid small coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

def find_valid_coloring(n, deadline):
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    if num_triples == 0:
        return True, []
    
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
    
    def compute_violations(colors):
        """Compute number of K4 red violations and K5 blue violations"""
        k4_red_count = [0] * num_k4s
        k5_blue_count = [0] * num_k5s
        
        for ki, idxs in enumerate(k4s):
            for idx in idxs:
                if colors[idx] == 0:
                    k4_red_count[ki] += 1
                    
        for ki, idxs in enumerate(k5s):
            for idx in idxs:
                if colors[idx] == 1:
                    k5_blue_count[ki] += 1
        
        k4_viol = sum(1 for c in k4_red_count if c == 4)
        k5_viol = sum(1 for c in k5_blue_count if c == 10)
        return k4_viol + k5_viol
    
    def compute_soft_energy(k4_red_count, k5_blue_count):
        """Compute smooth surrogate energy"""
        soft = 0.0
        for c in k4_red_count:
            if c >= 2:
                soft += (4 - c) ** 2
        for c in k5_blue_count:
            if c >= 7:
                soft += (10 - c) ** 2
        return soft
    
    # Multiple restarts with different strategies
    max_restarts = 30
    restart_count = 0
    
    while time.time() < deadline - 5 and restart_count < max_restarts:
        restart_count += 1
        
        # Different initialization strategies
        strategy = restart_count % 4
        
        if strategy == 0:
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(num_triples)]
        elif strategy == 1:
            # Slight red bias (to avoid all-red K4s)
            colors = [0 if random.random() < 0.55 else 1 for _ in range(num_triples)]
        elif strategy == 2:
            # Slight blue bias (to avoid all-blue K5s)
            colors = [0 if random.random() < 0.45 else 1 for _ in range(num_triples)]
        else:
            # Structured initialization based on vertex sum mod 3
            colors = []
            for a, b, c in triples:
                if (a + b + c) % 3 == 0:
                    colors.append(0)
                elif (a + b + c) % 3 == 1:
                    colors.append(1)
                else:
                    colors.append(random.randint(0, 1))
        
        # Compute initial counts
        k4_red_count = [0] * num_k4s
        k5_blue_count = [0] * num_k5s
        
        for ki, idxs in enumerate(k4s):
            for idx in idxs:
                if colors[idx] == 0:
                    k4_red_count[ki] += 1
                    
        for ki, idxs in enumerate(k5s):
            for idx in idxs:
                if colors[idx] == 1:
                    k5_blue_count[ki] += 1
        
        energy = sum(1 for c in k4_red_count if c == 4) + sum(1 for c in k5_blue_count if c == 10)
        soft = compute_soft_energy(k4_red_count, k5_blue_count)
        
        if energy == 0:
            return True, colors
            
        # Simulated annealing
        temperature = 2.0
        cooling = 0.9995
        no_improvement = 0
        max_no_improvement = 5000
        best_local_energy = energy
        best_local_soft = soft
        best_local_colors = colors[:]
        
        while no_improvement < max_no_improvement and time.time() < deadline - 5:
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta_hard = 0
            delta_soft = 0.0
            
            # Update K4s containing triple i
            for ki in triple_to_k4s[i]:
                old_red = k4_red_count[ki]
                if old_color == 0:
                    new_red = old_red - 1
                    if old_red == 4:
                        delta_hard += 1
                    if new_red == 4:
                        delta_hard -= 1
                    if old_red >= 2:
                        delta_soft += (4 - new_red) ** 2 - (4 - old_red) ** 2
                    elif new_red >= 2:
                        delta_soft += (4 - new_red) ** 2
                    k4_red_count[ki] = new_red
                else:
                    new_red = old_red + 1
                    if old_red == 4:
                        delta_hard -= 1
                    if new_red == 4:
                        delta_hard += 1
                    if old_red >= 2:
                        delta_soft += (4 - new_red) ** 2 - (4 - old_red) ** 2
                    elif new_red >= 2:
                        delta_soft += (4 - new_red) ** 2
                    k4_red_count[ki] = new_red
                    
            # Update K5s containing triple i
            for ki in triple_to_k5s[i]:
                old_blue = k5_blue_count[ki]
                if old_color == 1:
                    new_blue = old_blue - 1
                    if old_blue == 10:
                        delta_hard += 1
                    if new_blue == 10:
                        delta_hard -= 1
                    if old_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2 - (10 - old_blue) ** 2
                    elif new_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2
                    k5_blue_count[ki] = new_blue
                else:
                    new_blue = old_blue + 1
                    if old_blue == 10:
                        delta_hard -= 1
                    if new_blue == 10:
                        delta_hard += 1
                    if old_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2 - (10 - old_blue) ** 2
                    elif new_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2
                    k5_blue_count[ki] = new_blue
            
            energy += delta_hard
            soft += delta_soft
            
            if energy < 0:
                energy = 0
            if soft < 0:
                soft = 0
            
            # Accept if soft energy improves, or with probability based on temperature
            if delta_soft <= 0 or (temperature > 0.01 and random.random() < math.exp(-delta_soft / temperature)):
                colors[i] = new_color
                if delta_hard < 0 or (delta_hard == 0 and delta_soft < 0):
                    no_improvement = 0
                    if energy < best_local_energy or (energy == best_local_energy and soft < best_local_soft):
                        best_local_energy = energy
                        best_local_soft = soft
                        best_local_colors = colors[:]
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
                energy -= delta_hard
                soft -= delta_soft
                no_improvement += 1
            
            if energy == 0:
                return True, colors
                
            temperature *= cooling
        
        # After local search, check if we found a solution
        if energy == 0:
            return True, colors
            
        # If local best is better, try to polish it
        if best_local_energy > 0 and time.time() < deadline - 3:
            colors = best_local_colors[:]
            k4_red_count = [0] * num_k4s
            k5_blue_count = [0] * num_k5s
            
            for ki, idxs in enumerate(k4s):
                for idx in idxs:
                    if colors[idx] == 0:
                        k4_red_count[ki] += 1
                        
            for ki, idxs in enumerate(k5s):
                for idx in idxs:
                    if colors[idx] == 1:
                        k5_blue_count[ki] += 1
                        
            energy = sum(1 for c in k4_red_count if c == 4) + sum(1 for c in k5_blue_count if c == 10)
            soft = compute_soft_energy(k4_red_count, k5_blue_count)
            
            # Targeted local search: focus on triples in near-violations
            temperature = 0.3
            cooling = 0.998
            no_improvement = 0
            max_no_improvement = 8000
            
            while no_improvement < max_no_improvement and time.time() < deadline - 3:
                # Find candidate triples in near-violations
                candidate_triples = set()
                
                for ki, idxs in enumerate(k4s):
                    if k4_red_count[ki] >= 3:
                        for idx in idxs:
                            if colors[idx] == 1:  # Blue triple in mostly-red K4
                                candidate_triples.add(idx)
                                
                for ki, idxs in enumerate(k5s):
                    if k5_blue_count[ki] >= 8:
                        for idx in idxs:
                            if colors[idx] == 0:  # Red triple in mostly-blue K5
                                candidate_triples.add(idx)
                
                if candidate_triples:
                    i = random.choice(list(candidate_triples))
                else:
                    i = random.randint(0, num_triples - 1)
                    
                old_color = colors[i]
                new_color = 1 - old_color
                
                delta_hard = 0
                delta_soft = 0.0
                
                for ki in triple_to_k4s[i]:
                    old_red = k4_red_count[ki]
                    if old_color == 0:
                        new_red = old_red - 1
                        if old_red == 4:
                            delta_hard += 1
                        if new_red == 4:
                            delta_hard -= 1
                        if old_red >= 2:
                            delta_soft += (4 - new_red) ** 2 - (4 - old_red) ** 2
                        elif new_red >= 2:
                            delta_soft += (4 - new_red) ** 2
                        k4_red_count[ki] = new_red
                    else:
                        new_red = old_red + 1
                        if old_red == 4:
                            delta_hard -= 1
                        if new_red == 4:
                            delta_hard += 1
                        if old_red >= 2:
                            delta_soft += (4 - new_red) ** 2 - (4 - old_red) ** 2
                        elif new_red >= 2:
                            delta_soft += (4 - new_red) ** 2
                        k4_red_count[ki] = new_red
                        
                for ki in triple_to_k5s[i]:
                    old_blue = k5_blue_count[ki]
                    if old_color == 1:
                        new_blue = old_blue - 1
                        if old_blue == 10:
                            delta_hard += 1
                        if new_blue == 10:
                            delta_hard -= 1
                        if old_blue >= 7:
                            delta_soft += (10 - new_blue) ** 2 - (10 - old_blue) ** 2
                        elif new_blue >= 7:
                            delta_soft += (10 - new_blue) ** 2
                        k5_blue_count[ki] = new_blue
                    else:
                        new_blue = old_blue + 1
                        if old_blue == 10:
                            delta_hard -= 1
                        if new_blue == 10:
                            delta_hard += 1
                        if old_blue >= 7:
                            delta_soft += (10 - new_blue) ** 2 - (10 - old_blue) ** 2
                        elif new_blue >= 7:
                            delta_soft += (10 - new_blue) ** 2
                        k5_blue_count[ki] = new_blue
                
                energy += delta_hard
                soft += delta_soft
                
                if energy < 0:
                    energy = 0
                if soft < 0:
                    soft = 0
                
                if delta_soft <= 0 or (temperature > 0.01 and random.random() < math.exp(-delta_soft / temperature)):
                    colors[i] = new_color
                    if delta_hard < 0 or (delta_hard == 0 and delta_soft < 0):
                        no_improvement = 0
                    else:
                        no_improvement += 1
                else:
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
                    energy -= delta_hard
                    soft -= delta_soft
                    no_improvement += 1
                
                if energy == 0:
                    return True, colors
                    
                temperature *= cooling
            
            if energy == 0:
                return True, colors
                
    return False, None
