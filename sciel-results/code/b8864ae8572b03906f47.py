import itertools
import random
import time
import math
import sys

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin for 150s limit
    
    # Target n=35 first, then fallback to 34
    target_ns = [35, 34, 33]
    
    for n in target_ns:
        # Time budget per n
        if n == 35:
            budget = 75.0
        elif n == 34:
            budget = 35.0
        else:
            budget = 15.0
            
        # Check if we have enough time
        elapsed = time.time() - start_time
        if elapsed + budget > time_limit:
            budget = time_limit - elapsed - 5.0
            if budget < 5.0:
                break
        
        colors = find_coloring(n, budget, start_time, time_limit)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback: trivial coloring for small n
    n = 10
    triples = list(itertools.combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}

def find_coloring(n, time_budget, start_time, time_limit):
    """
    Finds a valid coloring for n vertices.
    Returns colors list if found, None otherwise.
    """
    if n < 4:
        triples = list(itertools.combinations(range(n), 3))
        return [0] * len(triples)
        
    triples = list(itertools.combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and 5-sets
    four_sets = list(itertools.combinations(range(n), 4))
    five_sets = list(itertools.combinations(range(n), 5))
    
    # Map triple -> list of indices of 4-sets containing it
    triple_to_fours = [[] for _ in range(num_triples)]
    for idx, quad in enumerate(four_sets):
        t1 = (quad[0], quad[1], quad[2])
        t2 = (quad[0], quad[1], quad[3])
        t3 = (quad[0], quad[2], quad[3])
        t4 = (quad[1], quad[2], quad[3])
        triple_to_fours[triple_to_idx[t1]].append(idx)
        triple_to_fours[triple_to_idx[t2]].append(idx)
        triple_to_fours[triple_to_idx[t3]].append(idx)
        triple_to_fours[triple_to_idx[t4]].append(idx)
        
    # Map triple -> list of indices of 5-sets containing it
    triple_to_fives = [[] for _ in range(num_triples)]
    for idx, quint in enumerate(five_sets):
        for t in itertools.combinations(quint, 3):
            triple_to_fives[triple_to_idx[t]].append(idx)
            
    # Helper to compute initial counts and cost
    def init_state():
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        red_counts = [0] * len(four_sets)
        blue_counts = [0] * len(five_sets)
        
        cost = 0
        for i, quad in enumerate(four_sets):
            t1 = triple_to_idx[(quad[0], quad[1], quad[2])]
            t2 = triple_to_idx[(quad[0], quad[1], quad[3])]
            t3 = triple_to_idx[(quad[0], quad[2], quad[3])]
            t4 = triple_to_idx[(quad[1], quad[2], quad[3])]
            rc = (1 if colors[t1]==0 else 0) + (1 if colors[t2]==0 else 0) + (1 if colors[t3]==0 else 0) + (1 if colors[t4]==0 else 0)
            red_counts[i] = rc
            if rc == 4:
                cost += 1
                
        for i, quint in enumerate(five_sets):
            bc = 0
            for t in itertools.combinations(quint, 3):
                if colors[triple_to_idx[t]] == 1:
                    bc += 1
            blue_counts[i] = bc
            if bc == 10:
                cost += 1
                
        return colors, red_counts, blue_counts, cost

    best_cost_found = float('inf')
    best_colors = None
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.05
    cooling = 0.999
    
    # Time management for the whole find_coloring call
    end_time = start_time + time_budget
    
    while time.time() < end_time - 2.0:
        colors, red_counts, blue_counts, cost = init_state()
        
        if cost == 0:
            return colors
            
        if cost < best_cost_found:
            best_cost_found = cost
            best_colors = colors[:]
            
        # Tabu list: store (triple_idx, time_added)
        tabu = []
        no_improve_steps = 0
        max_no_improve = 20000
        T = T_start
        
        while no_improve_steps < max_no_improve and time.time() < end_time - 2.0:
            if cost == 0:
                return colors
                
            # Clean tabu list (keep only last 100ms)
            # This is O(L) every step, might be slow.
            # Better: just check if t_idx is in the last 100 entries.
            
            t_idx = random.randint(0, num_triples - 1)
            
            # Check tabu: is t_idx in the recent history?
            # We'll just check the last 50 entries of tabu
            in_tabu = False
            for i in range(len(tabu)):
                if tabu[i][0] == t_idx:
                    # Check if it's recent (within last 100 steps approx)
                    # We store step count instead of time for faster comparison?
                    # Let's use step count.
                    # We need a global step counter for this restart.
                    pass # Logic needs step counter
                    
            # Let's simplify: No tabu, just SA with restarts.
            # The current seed program doesn't use tabu.
            # Let's stick to SA with good restarts.
            
            current_color = colors[t_idx]
            
            delta_cost = 0
            
            for quad_idx in triple_to_fours[t_idx]:
                rc = red_counts[quad_idx]
                if current_color == 0:
                    if rc == 4:
                        delta_cost -= 1
                else:
                    if rc == 3:
                        delta_cost += 1
                        
            for quint_idx in triple_to_fives[t_idx]:
                bc = blue_counts[quint_idx]
                if current_color == 1:
                    if bc == 10:
                        delta_cost -= 1
                else:
                    if bc == 9:
                        delta_cost += 1
                        
            accept = False
            if delta_cost <= 0:
                accept = True
            else:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    accept = True
                
            if accept:
                colors[t_idx] = 1 - current_color
                cost += delta_cost
                
                for quad_idx in triple_to_fours[t_idx]:
                    if current_color == 0:
                        red_counts[quad_idx] -= 1
                    else:
                        red_counts[quad_idx] += 1
                            
                for quint_idx in triple_to_fives[t_idx]:
                    if current_color == 1:
                        blue_counts[quint_idx] -= 1
                    else:
                        blue_counts[quint_idx] += 1
                            
                if cost < best_cost_found:
                    best_cost_found = cost
                    best_colors = colors[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    
                T *= cooling
            else:
                no_improve_steps += 1
                T *= cooling

    return best_colors if best_cost_found == 0 else None
