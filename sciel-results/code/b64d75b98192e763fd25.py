import time
import random
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a valid coloring for n=34 down to n=1
    # The known lower bound is 34.
    
    for n in range(34, 0, -1):
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
    # Precompute structures for fast incremental energy calculation
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # For each triple, precompute which K4s and K5s contain it
    # K4 containing triple (a,b,c): add vertex d, where d is any other vertex
    # K5 containing triple (a,b,c): add two vertices d,e, where d,e are any other vertices
    
    # Build mapping: triple_idx -> list of (k4_idx, position_in_k4)
    # and triple_idx -> list of (k5_idx, position_in_k5)
    
    # First, enumerate all K4s and K5s
    k4s = []
    k4_list = []
    for quad in combinations(range(n), 4):
        idxs = [triple_index[t] for t in combinations(quad, 3)]
        k4s.append(idxs)
        k4_list.append(quad)
    num_k4s = len(k4s)
    
    k5s = []
    k5_list = []
    for quint in combinations(range(n), 5):
        idxs = [triple_index[t] for t in combinations(quint, 3)]
        k5s.append(idxs)
        k5_list.append(quint)
    num_k5s = len(k5s)
    
    # Build reverse mappings: for each triple, which K4s/K5s contain it
    # triple_to_k4s[i] = list of k4 indices that contain triple i
    # triple_to_k5s[i] = list of k5 indices that contain triple i
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for k4_idx, idxs in enumerate(k4s):
        for tri_idx in idxs:
            triple_to_k4s[tri_idx].append(k4_idx)
            
    for k5_idx, idxs in enumerate(k5s):
        for tri_idx in idxs:
            triple_to_k5s[tri_idx].append(k5_idx)
    
    # Simulated Annealing with incremental energy
    best_energy = float('inf')
    best_colors = None
    
    while time.time() < deadline - 5:
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts: k4_red_count[k4_idx] = number of red triples in that K4
        # k5_blue_count[k5_idx] = number of blue triples in that K5
        k4_red_count = [0] * num_k4s
        k5_blue_count = [0] * num_k5s
        
        for tri_idx, color in enumerate(colors):
            if color == 0:  # red
                for k4_idx in triple_to_k4s[tri_idx]:
                    k4_red_count[k4_idx] += 1
            else:  # blue
                for k5_idx in triple_to_k5s[tri_idx]:
                    k5_blue_count[k5_idx] += 1
        
        # Energy = number of K4s with all 4 red + number of K5s with all 10 blue
        energy = 0
        for cnt in k4_red_count:
            if cnt == 4:
                energy += 1
        for cnt in k5_blue_count:
            if cnt == 10:
                energy += 1
        
        if energy == 0:
            return True, colors
            
        if energy < best_energy:
            best_energy = energy
            best_colors = colors[:]
        
        # Local Search with incremental updates
        current_energy = energy
        no_improvement = 0
        max_no_improvement = 500
        temperature = 1.0
        cooling = 0.9995
        
        while no_improvement < max_no_improvement and time.time() < deadline - 5:
            # Pick random triple to flip
            i = random.randint(0, num_triples - 1)
            
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Calculate delta energy by updating counts
            # If flipping from red to blue:
            #   For each K4 containing triple i: k4_red_count decreases by 1
            #   For each K5 containing triple i: k5_blue_count increases by 1
            # If flipping from blue to red:
            #   For each K4 containing triple i: k4_red_count increases by 1
            #   For each K5 containing triple i: k5_blue_count decreases by 1
            
            delta_energy = 0
            
            if old_color == 0:  # red -> blue
                # K4s: red count decreases by 1
                for k4_idx in triple_to_k4s[i]:
                    old_cnt = k4_red_count[k4_idx]
                    new_cnt = old_cnt - 1
                    # Energy contribution: was 1 if old_cnt==4, else 0
                    # Now: 1 if new_cnt==4, else 0
                    if old_cnt == 4:
                        delta_energy -= 1  # removed a violating K4
                    if new_cnt == 4:
                        delta_energy += 1  # added a violating K4 (can't happen since decreasing)
                    k4_red_count[k4_idx] = new_cnt
                # K5s: blue count increases by 1
                for k5_idx in triple_to_k5s[i]:
                    old_cnt = k5_blue_count[k5_idx]
                    new_cnt = old_cnt + 1
                    if old_cnt == 10:
                        delta_energy -= 1  # removed a violating K5 (can't happen since increasing)
                    if new_cnt == 10:
                        delta_energy += 1  # added a violating K5
                    k5_blue_count[k5_idx] = new_cnt
            else:  # blue -> red
                # K4s: red count increases by 1
                for k4_idx in triple_to_k4s[i]:
                    old_cnt = k4_red_count[k4_idx]
                    new_cnt = old_cnt + 1
                    if old_cnt == 4:
                        delta_energy -= 1  # removed a violating K4
                    if new_cnt == 4:
                        delta_energy += 1  # added a violating K4
                    k4_red_count[k4_idx] = new_cnt
                # K5s: blue count decreases by 1
                for k5_idx in triple_to_k5s[i]:
                    old_cnt = k5_blue_count[k5_idx]
                    new_cnt = old_cnt - 1
                    if old_cnt == 10:
                        delta_energy -= 1  # removed a violating K5
                    if new_cnt == 10:
                        delta_energy += 1  # added a violating K5 (can't happen since decreasing)
                    k5_blue_count[k5_idx] = new_cnt
            
            new_energy = current_energy + delta_energy
            
            # Accept if improvement or with probability exp(-delta/temperature)
            if delta_energy <= 0 or random.random() < (2.718281828 ** (-delta_energy / max(temperature, 1e-10))):
                colors[i] = new_color
                current_energy = new_energy
                no_improvement = 0
            else:
                # Reject: revert counts
                if old_color == 0:  # we tried red->blue, now revert blue->red
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_count[k4_idx] += 1
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_count[k5_idx] -= 1
                else:  # we tried blue->red, now revert red->blue
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_count[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_count[k5_idx] += 1
                no_improvement += 1
            
            temperature *= cooling
            
            if current_energy == 0:
                return True, colors
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_colors = colors[:]
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # If we didn't find 0, return False
    return False, None

def calculate_energy(colors, k4s, k5s):
    e = 0
    for idxs in k4s:
        if all(colors[i] == 0 for i in idxs):
            e += 1
    for idxs in k5s:
        if all(colors[i] == 1 for i in idxs):
            e += 1
    return e
