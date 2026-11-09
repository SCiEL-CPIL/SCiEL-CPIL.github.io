import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structures(target_n):
        """Precompute structures for fast local search."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        quad_neighbors = [[] for _ in range(num_triples)]
        quint_neighbors = [[] for _ in range(num_triples)]
        
        # Iterate over all quads
        for a, b, c, d in combinations(range(target_n), 4):
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            
            quad_neighbors[t1].append([t2, t3, t4])
            quad_neighbors[t2].append([t1, t3, t4])
            quad_neighbors[t3].append([t1, t2, t4])
            quad_neighbors[t4].append([t1, t2, t3])
            
        # Iterate over all quints
        for quint in combinations(range(target_n), 5):
            quads_in_quint = list(combinations(quint, 3))
            triple_indices_in_quint = [triple_to_idx[t] for t in quads_in_quint]
            
            for i, t_idx in enumerate(triple_indices_in_quint):
                others = [j for j in range(10) if j != i]
                quint_neighbors[t_idx].append([triple_indices_in_quint[j] for j in others])
                
        return triple_list, num_triples, triple_to_idx, quad_neighbors, quint_neighbors

    def run_local_search(target_n, initial_colors, deadline, seed_offset=0):
        """
        Runs simulated annealing to minimize violations.
        Returns (colors, num_violations)
        """
        triple_list, num_triples, triple_to_idx, quad_neighbors, quint_neighbors = build_structures(target_n)
        
        # Color array: 0 for Red, 1 for Blue
        colors = np.array(initial_colors, dtype=np.int8)
        
        # Initial violation count
        red_k4_count = 0
        blue_k5_count = 0
        
        # Initial full scan
        for a, b, c, d in combinations(range(target_n), 4):
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            if colors[t1]==0 and colors[t2]==0 and colors[t3]==0 and colors[t4]==0:
                red_k4_count += 1
        
        for quint in combinations(range(target_n), 5):
            quads_in_quint = list(combinations(quint, 3))
            triple_indices_in_quint = [triple_to_idx[t] for t in quads_in_quint]
            if all(colors[i] == 1 for i in triple_indices_in_quint):
                blue_k5_count += 1
                
        # Simulated Annealing
        T = 5.0
        T_min = 0.001
        cooling = 0.995
        max_iters = 500000
        current_cost = red_k4_count + blue_k5_count
        best_cost = current_cost
        best_colors = colors.copy()
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Choose a random triple to flip
            t_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[t_idx]
            
            # Calculate delta red_k4
            delta_r4 = 0
            if old_color == 0: # Currently Red
                for others in quad_neighbors[t_idx]:
                    if colors[others[0]] == 0 and colors[others[1]] == 0 and colors[others[2]] == 0:
                        delta_r4 -= 1
            else: # Currently Blue
                for others in quad_neighbors[t_idx]:
                    if colors[others[0]] == 0 and colors[others[1]] == 0 and colors[others[2]] == 0:
                        delta_r4 += 1
                        
            # Calculate delta blue_k5
            delta_b5 = 0
            if old_color == 1: # Currently Blue
                for others in quint_neighbors[t_idx]:
                    all_blue = True
                    for idx in others:
                        if colors[idx] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        delta_b5 -= 1
            else: # Currently Red
                for others in quint_neighbors[t_idx]:
                    all_blue = True
                    for idx in others:
                        if colors[idx] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        delta_b5 += 1
                        
            delta_cost = delta_r4 + delta_b5
            
            # Accept?
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Apply flip
                colors[t_idx] = 1 - old_color
                red_k4_count += delta_r4
                blue_k5_count += delta_b5
                current_cost = red_k4_count + blue_k5_count
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors.tolist(), 0
            
            T *= cooling
            if T < T_min:
                T = T_min
                if i % 10000 == 0:
                     T = 5.0 # Reheat
            
        return best_colors.tolist(), best_cost

    # Strategy: Try increasing n
    # We prioritize higher n.
    
    # Let's define the search order.
    # Given the difficulty, jumping straight to 34 might waste time if it's very hard.
    # But the goal is MAX n.
    
    targets = [34, 33, 32, 31, 30, 29, 28, 27]
    time_allocations = [10, 10, 8, 5, 5, 3, 3, 3]
    
    best_found = (9, mod_color(9, 3))
    
    for i, n in enumerate(targets):
        if time.time() > deadline - 10:
            break
            
        alloc = time_allocations[i] if i < len(time_allocations) else 3
        current_deadline = time.time() + alloc
        
        # Multiple restarts
        max_restarts = 10
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() > current_deadline:
                break
                
            # Generate initial coloring
            # Modular arithmetic is a good start
            if r == 0:
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 1:
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 2:
                init = [0 if (a+b+c) % 7 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 3:
                init = [0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            else:
                triple_list_n = list(combinations(range(n), 3))
                init = [0 if random.random() < 0.4 else 1 for _ in triple_list_n]

            colors, cost = run_local_search(n, init, current_deadline, seed_offset=r)
            
            if cost == 0:
                # Verify strictly
                if is_valid(n, colors):
                    best_found = (n, colors)
                    found_valid = True
                    break
        
        if found_valid:
            break

    return {'n': best_found[0], 'colors': best_found[1]}

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
