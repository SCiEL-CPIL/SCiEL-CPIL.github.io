import time
import random
from itertools import combinations
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    def precompute_sets(n):
        triples = get_triples(n)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        triple_to_4sets = [[] for _ in range(len(triples))]
        triple_to_5sets = [[] for _ in range(len(triples))]
        
        for i, s4 in enumerate(four_sets):
            a, b, c, d = s4
            t1 = (a, b, c)
            t2 = (a, b, d)
            t3 = (a, c, d)
            t4 = (b, c, d)
            triple_to_4sets[triple_index[t1]].append(i)
            triple_to_4sets[triple_index[t2]].append(i)
            triple_to_4sets[triple_index[t3]].append(i)
            triple_to_4sets[triple_index[t4]].append(i)
            
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                triple_to_5sets[triple_index[t]].append(i)
                
        return triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets

    def solve_for_n(n, time_limit):
        if n < 4:
            return [0] * 0
        
        triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets = precompute_sets(n)
        num_triples = len(triples)
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        # 1. Structured Initialization: Lexicographic Mod-2 Rule
        # Color (a,b,c) is Red (0) if (a+b+c) % 2 == 0, else Blue (1)
        # This creates a highly structured pattern that often reduces large-scale monochromatic cliques.
        colors = [0] * num_triples
        for i, t in enumerate(triples):
            a, b, c = t
            colors[i] = (a + b + c) % 2
            
        # Alternative initialization: Random
        # colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # counts
        red_count_4 = [0] * num_4sets
        blue_count_5 = [0] * num_5sets
        
        # Initialize counts
        for t_idx, t in enumerate(triples):
            is_red = (colors[t_idx] == 0)
            for s4_idx in triple_to_4sets[t_idx]:
                if is_red:
                    red_count_4[s4_idx] += 1
            for s5_idx in triple_to_5sets[t_idx]:
                if not is_red: # blue
                    blue_count_5[s5_idx] += 1
                    
        def calc_cost():
            cost = 0
            for c in red_count_4:
                if c == 4:
                    cost += 1
            for c in blue_count_5:
                if c == 10:
                    cost += 1
            return cost
            
        current_cost = calc_cost()
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors.copy()
        
        # 2. Targeted Conflict Resolution Local Search
        # Instead of random flips, we identify violated cliques and flip a random triple within them.
        # This is a "greedy repair" strategy.
        
        # Precompute list of triples for each 4-set and 5-set for fast access
        # We already have triple_to_4sets, but we need 4set_to_triples and 5set_to_triples
        # Let's build them quickly if needed, or just iterate.
        # For n=34, 4sets ~ 46k, 5sets ~ 278k. Iterating all to find violations is O(N^4) which is ~ 1M ops. Fast.
        
        def find_violations():
            red_violations = []
            blue_violations = []
            for i in range(num_4sets):
                if red_count_4[i] == 4:
                    red_violations.append(i)
            for j in range(num_5sets):
                if blue_count_5[j] == 10:
                    blue_violations.append(j)
            return red_violations, blue_violations
            
        temp = 1.0
        cooling_rate = 0.9995
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < time_limit:
            if temp < 0.01:
                break
                
            if no_improve_count > max_no_improve:
                # Restart with randomization if stuck
                temp = 1.0
                no_improve_count = 0
                # Optionally re-randomize a few bits? Or just keep going.
                # Let's try a few random flips to escape local minima
                for _ in range(10):
                    t_idx = random.randint(0, num_triples - 1)
                    # Apply flip
                    old_color = colors[t_idx]
                    new_color = 1 - old_color
                    colors[t_idx] = new_color
                    for s4_idx in triple_to_4sets[t_idx]:
                        if old_color == 0: red_count_4[s4_idx] -= 1
                        else: red_count_4[s4_idx] += 1
                    for s5_idx in triple_to_5sets[t_idx]:
                        if old_color == 0: blue_count_5[s5_idx] += 1
                        else: blue_count_5[s5_idx] -= 1
                    current_cost = calc_cost()
                continue
            
            red_violations, blue_violations = find_violations()
            
            if not red_violations and not blue_violations:
                return colors
                
            # Prioritize Red K4 violations (harder to satisfy? or just pick one)
            # The prompt suggests prioritizing Red K4s.
            if red_violations:
                # Pick a random violated 4-set
                s4_idx = random.choice(red_violations)
                # Get the 4 triples in this 4-set
                s4 = four_sets[s4_idx]
                a, b, c, d = s4
                t_indices = [
                    triple_index[(a, b, c)],
                    triple_index[(a, b, d)],
                    triple_index[(a, c, d)],
                    triple_index[(b, c, d)]
                ]
                # Pick a random triple to flip
                t_idx = random.choice(t_indices)
            else:
                # Pick a random violated 5-set
                s5_idx = random.choice(blue_violations)
                s5 = five_sets[s5_idx]
                t_indices = [triple_index[t] for t in combinations(s5, 3)]
                t_idx = random.choice(t_indices)
                
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta cost
            delta_cost = 0
            affected_4 = triple_to_4sets[t_idx]
            affected_5 = triple_to_5sets[t_idx]
            
            if old_color == 0: # Red -> Blue
                for s4_idx in affected_4:
                    old_c = red_count_4[s4_idx]
                    new_c = old_c - 1
                    if old_c == 4: delta_cost -= 1
                    if new_c == 4: delta_cost += 1
                for s5_idx in affected_5:
                    old_c = blue_count_5[s5_idx]
                    new_c = old_c + 1
                    if old_c == 10: delta_cost -= 1
                    if new_c == 10: delta_cost += 1
            else: # Blue -> Red
                for s4_idx in affected_4:
                    old_c = red_count_4[s4_idx]
                    new_c = old_c + 1
                    if old_c == 4: delta_cost -= 1
                    if new_c == 4: delta_cost += 1
                for s5_idx in affected_5:
                    old_c = blue_count_5[s5_idx]
                    new_c = old_c - 1
                    if old_c == 10: delta_cost -= 1
                    if new_c == 10: delta_cost += 1
            
            accept = False
            if delta_cost < 0:
                accept = True
            elif delta_cost == 0:
                accept = True
            else:
                if temp > 0:
                    if random.random() < math.exp(-delta_cost / temp):
                        accept = True
            
            if accept:
                colors[t_idx] = new_color
                for s4_idx in affected_4:
                    if old_color == 0:
                        red_count_4[s4_idx] -= 1
                    else:
                        red_count_4[s4_idx] += 1
                for s5_idx in affected_5:
                    if old_color == 0:
                        blue_count_5[s5_idx] += 1
                    else:
                        blue_count_5[s5_idx] -= 1
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve_count = 0
                    if best_cost == 0:
                        return best_colors
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            temp *= cooling_rate
            
        if best_cost == 0:
            return best_colors
        return None

    # Try n from 34 downwards
    # We have ~110s. 
    # n=34 is the target. 
    # If we fail 34 quickly, try 33, etc.
    # But usually, if the method is good, it should find 34.
    
    # Let's allocate time dynamically.
    # Try 34 for 60s. If fail, try 33 for 30s, etc.
    
    n_to_try = 34
    time_left = deadline - time.time()
    
    # Attempt 34
    if time_left > 5:
        res = solve_for_n(34, time.time() + min(60, time_left - 1))
        if res is not None:
            return {'n': 34, 'colors': res}
            
    # If 34 failed, try smaller n's with remaining time
    time_left = deadline - time.time()
    if time_left > 5:
        # Try 33
        res = solve_for_n(33, time.time() + min(20, time_left - 1))
        if res is not None:
            return {'n': 33, 'colors': res}
            
    time_left = deadline - time.time()
    if time_left > 5:
        # Try 32
        res = solve_for_n(32, time.time() + min(10, time_left - 1))
        if res is not None:
            return {'n': 32, 'colors': res}
            
    # Fallback
    n = 4
    colors = [0, 0, 0, 1]
    return {'n': n, 'colors': colors}
