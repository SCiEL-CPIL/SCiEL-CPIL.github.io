import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 145.0  # 5s margin for safety
    
    def try_solve(n, time_budget):
        if n < 1:
            return None
        
        # Precomputation
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # 4-sets
        quads = list(combinations(range(n), 4))
        num_quads = len(quads)
        quad_triples = []
        for q in quads:
            ts = [triple_index[t] for t in combinations(q, 3)]
            quad_triples.append(ts)
            
        # 5-sets
        quints = list(combinations(range(n), 5))
        num_quints = len(quints)
        quint_triples = []
        for q in quints:
            ts = [triple_index[t] for t in combinations(q, 3)]
            quint_triples.append(ts)
            
        # Map triple to quads and quints
        triple_to_quads = [[] for _ in range(num_triples)]
        for i, ts in enumerate(quad_triples):
            for t in ts:
                triple_to_quads[t].append(i)
                
        triple_to_quints = [[] for _ in range(num_triples)]
        for i, ts in enumerate(quint_triples):
            for t in ts:
                triple_to_quints[t].append(i)
                
        # Initialize random coloring
        colors = [random.choice([0, 1]) for _ in range(num_triples)]
        
        # Initial violation counts
        red_states = [False] * num_quads
        blue_states = [False] * num_quints
        red_count = 0
        blue_count = 0
        
        for i, ts in enumerate(quad_triples):
            # Check if all 4 are 0
            if colors[ts[0]] == 0 and colors[ts[1]] == 0 and colors[ts[2]] == 0 and colors[ts[3]] == 0:
                red_states[i] = True
                red_count += 1
                
        for i, ts in enumerate(quint_triples):
            # Check if all 10 are 1
            # Using sum is often faster than all for small lists in Py if not optimized
            # But min/max with early exit is good
            if min(colors[t] for t in ts) == 1:
                blue_states[i] = True
                blue_count += 1
                
        if red_count == 0 and blue_count == 0:
            return {'n': n, 'colors': colors}
            
        # Simulated Annealing
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        exp = math.exp
        
        while time.time() - start_time < time_limit - 5:
            if T < min_T:
                T = min_T
            else:
                T *= cooling_rate
                
            # Pick random triple
            t_idx = random.randint(0, num_triples - 1)
            
            # Flip color
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            delta_red = 0
            delta_blue = 0
            
            # Update quads
            for q_idx in triple_to_quads[t_idx]:
                ts = quad_triples[q_idx]
                is_red = (colors[ts[0]] == 0 and colors[ts[1]] == 0 and colors[ts[2]] == 0 and colors[ts[3]] == 0)
                if is_red and not red_states[q_idx]:
                    red_states[q_idx] = True
                    delta_red += 1
                elif not is_red and red_states[q_idx]:
                    red_states[q_idx] = False
                    delta_red -= 1
                    
            # Update quints
            for q_idx in triple_to_quints[t_idx]:
                ts = quint_triples[q_idx]
                is_blue = (min(colors[t] for t in ts) == 1)
                if is_blue and not blue_states[q_idx]:
                    blue_states[q_idx] = True
                    delta_blue += 1
                elif not is_blue and blue_states[q_idx]:
                    blue_states[q_idx] = False
                    delta_blue -= 1
                    
            delta_V = delta_red + delta_blue
            
            # Acceptance
            if delta_V < 0:
                red_count += delta_red
                blue_count += delta_blue
            else:
                # Boltzmann probability
                if random.random() < exp(-delta_V / T):
                    red_count += delta_red
                    blue_count += delta_blue
                else:
                    colors[t_idx] = old_color
            
            if red_count == 0 and blue_count == 0:
                return {'n': n, 'colors': colors}
                
        return None

    # Try decreasing n
    for n in [34, 33, 32, 30, 25, 20, 15, 10, 9]:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5
        if remaining <= 1:
            break
            
        # Allocate time based on n
        if n == 34:
            budget = min(100, remaining)
        elif n == 33:
            budget = min(30, remaining)
        elif n == 32:
            budget = min(20, remaining)
        else:
            budget = min(5, remaining)
            
        result = try_solve(n, budget)
        if result:
            return result
            
    # Fallback: Return a valid small n
    # n=9 with mod 3 coloring from seed
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
