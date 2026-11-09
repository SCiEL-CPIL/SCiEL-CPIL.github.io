import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

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

    def solve_walksat(n, deadline):
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # Precompute for each triple, the other triples that form a K4^(3) with it
        # For a triple (a,b,c), the K4^(3) is formed with any d > c
        # The other triples are (a,b,d), (a,c,d), (b,c,d)
        k4_partners = [[] for _ in range(num_triples)]
        for i, (a, b, c) in enumerate(triple_list):
            for d in range(c + 1, n):
                t1 = triple_to_idx[(a, b, d)]
                t2 = triple_to_idx[(a, c, d)]
                t3 = triple_to_idx[(b, c, d)]
                k4_partners[i].extend([t1, t2, t3])
        
        # Precompute for each triple, the other triples that form a K5^(3) with it
        # For a triple (a,b,c), the K5^(3) is formed with any d, e > c
        # The other triples are all triples from {a,b,c,d,e} except (a,b,c)
        k5_partners = [[] for _ in range(num_triples)]
        for i, (a, b, c) in enumerate(triple_list):
            for d in range(c + 1, n):
                for e in range(d + 1, n):
                    for t in combinations([a, b, c, d, e], 3):
                        if t == (a, b, c):
                            continue
                        k5_partners[i].append(triple_to_idx[t])
        
        # Initialize with modular coloring
        color = np.array([0 if (a + b + c) % 3 == 0 else 1 for (a, b, c) in triple_list], dtype=np.int8)
        
        def count_violations(color_arr):
            red_k4_count = 0
            for i in range(num_triples):
                if color_arr[i] == 0:  # Red
                    all_red = True
                    for partner in k4_partners[i]:
                        if color_arr[partner] != 0:
                            all_red = False
                            break
                    if all_red:
                        red_k4_count += 1
            blue_k5_count = 0
            for i in range(num_triples):
                if color_arr[i] == 1:  # Blue
                    all_blue = True
                    for partner in k5_partners[i]:
                        if color_arr[partner] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        blue_k5_count += 1
            return red_k4_count, blue_k5_count
        
        current_red_k4, current_blue_k5 = count_violations(color)
        best_red_k4, best_blue_k5 = current_red_k4, current_blue_k5
        best_color = color.copy()
        
        T = 10.0
        cooling = 0.9999
        min_T = 0.01
        p_walk = 0.1  # Probability of random walk
        
        step = 0
        max_steps = 5000000
        
        while time.time() < deadline - 2 and T > min_T:
            step += 1
            
            if random.random() < p_walk:
                var_idx = random.randint(0, num_triples - 1)
            else:
                # Find the triple that reduces violations the most
                best_delta = None
                best_var = -1
                for var_idx in range(num_triples):
                    old_val = color[var_idx]
                    new_val = 1 - old_val
                    color[var_idx] = new_val
                    new_red_k4, new_blue_k5 = count_violations(color)
                    color[var_idx] = old_val
                    delta = (new_red_k4 - current_red_k4) + 3 * (new_blue_k5 - current_blue_k5)
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                var_idx = best_var
            
            old_val = color[var_idx]
            new_val = 1 - old_val
            color[var_idx] = new_val
            new_red_k4, new_blue_k5 = count_violations(color)
            delta = (new_red_k4 - current_red_k4) + 3 * (new_blue_k5 - current_blue_k5)
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_red_k4, current_blue_k5 = new_red_k4, new_blue_k5
                if current_red_k4 + 3 * current_blue_k5 < best_red_k4 + 3 * best_blue_k5:
                    best_red_k4, best_blue_k5 = current_red_k4, current_blue_k5
                    best_color = color.copy()
                    if best_red_k4 == 0 and best_blue_k5 == 0:
                        return best_color.tolist(), 0
            else:
                color[var_idx] = old_val
            
            T *= cooling
            
            if step % 10000 == 0 and best_red_k4 == 0 and best_blue_k5 == 0:
                return best_color.tolist(), 0
        
        return best_color.tolist(), best_red_k4 + 3 * best_blue_k5

    # Strategy: Try n=34, 33, 32 with the WalkSAT solver
    targets = [34, 33, 32]
    time_allocations = [35, 25, 20]
    
    best_found = (9, [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)])
    
    for i, n in enumerate(targets):
        if time.time() > deadline - 15:
            break
            
        alloc = time_allocations[i] if i < len(time_allocations) else 15
        current_deadline = time.time() + alloc
        
        # Multiple restarts with different initializations
        max_restarts = 10
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() > current_deadline - 3:
                break
            
            # Different initializations
            if r == 0:
                # Modular by 3
                init_color = np.array([0 if (a + b + c) % 3 == 0 else 1 for (a, b, c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 1:
                # Modular by 5
                init_color = np.array([0 if (a + b + c) % 5 == 0 else 1 for (a, b, c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 2:
                # Modular by 7
                init_color = np.array([0 if (a + b + c) % 7 == 0 else 1 for (a, b, c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 3:
                # Vertex-based: color by sum of vertices mod 4
                init_color = np.array([0 if (a + b + c) % 4 == 0 else 1 for (a, b, c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 4:
                # Random with 40% red
                init_color = np.array([0 if random.random() < 0.4 else 1 for _ in range(len(list(combinations(range(n), 3))))], dtype=np.int8)
            elif r == 5:
                # Random with 50% red
                init_color = np.array([0 if random.random() < 0.5 else 1 for _ in range(len(list(combinations(range(n), 3))))], dtype=np.int8)
            elif r == 6:
                # Pattern: red if a+b+c is even
                init_color = np.array([0 if (a + b + c) % 2 == 0 else 1 for (a, b, c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 7:
                # Pattern: red if (a mod 3) == (b mod 3)
                init_color = np.array([0 if a % 3 == b % 3 else 1 for (a, b, c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 8:
                # Pattern: red if (a mod 4) < (b mod 4)
                init_color = np.array([0 if a % 4 < b % 4 else 1 for (a, b, c) in combinations(range(n), 3)], dtype=np.int8)
            else:
                # Random with 35% red
                init_color = np.array([0 if random.random() < 0.35 else 1 for _ in range(len(list(combinations(range(n), 3))))], dtype=np.int8)
            
            # Run the solver with this initialization
            def solve_walksat_with_init(n_val, deadline_val, init_col):
                triple_list = list(combinations(range(n_val), 3))
                num_triples = len(triple_list)
                triple_to_idx = {t: i for i, t in enumerate(triple_list)}
                
                # Precompute for each triple, the other triples that form a K4^(3) with it
                k4_partners = [[] for _ in range(num_triples)]
                for i, (a, b, c) in enumerate(triple_list):
                    for d in range(c + 1, n_val):
                        t1 = triple_to_idx[(a, b, d)]
                        t2 = triple_to_idx[(a, c, d)]
                        t3 = triple_to_idx[(b, c, d)]
                        k4_partners[i].extend([t1, t2, t3])
                
                # Precompute for each triple, the other triples that form a K5^(3) with it
                k5_partners = [[] for _ in range(num_triples)]
                for i, (a, b, c) in enumerate(triple_list):
                    for d in range(c + 1, n_val):
                        for e in range(d + 1, n_val):
                            for t in combinations([a, b, c, d, e], 3):
                                if t == (a, b, c):
                                    continue
                                k5_partners[i].append(triple_to_idx[t])
                
                color = init_col.copy()
                
                def count_violations_local(color_arr):
                    red_k4_count = 0
                    for i in range(num_triples):
                        if color_arr[i] == 0:  # Red
                            all_red = True
                            for partner in k4_partners[i]:
                                if color_arr[partner] != 0:
                                    all_red = False
                                    break
                            if all_red:
                                red_k4_count += 1
                    blue_k5_count = 0
                    for i in range(num_triples):
                        if color_arr[i] == 1:  # Blue
                            all_blue = True
                            for partner in k5_partners[i]:
                                if color_arr[partner] != 1:
                                    all_blue = False
                                    break
                            if all_blue:
                                blue_k5_count += 1
                    return red_k4_count, blue_k5_count
                
                current_red_k4, current_blue_k5 = count_violations_local(color)
                best_red_k4, best_blue_k5 = current_red_k4, current_blue_k5
                best_color = color.copy()
                
                T = 10.0
                cooling = 0.9999
                min_T = 0.01
                p_walk = 0.1  # Probability of random walk
                
                step = 0
                max_steps = 5000000
                
                while time.time() < deadline_val - 2 and T > min_T:
                    step += 1
                    
                    if random.random() < p_walk:
                        var_idx = random.randint(0, num_triples - 1)
                    else:
                        # Find the triple that reduces violations the most
                        best_delta = None
                        best_var = -1
                        for var_idx in range(num_triples):
                            old_val = color[var_idx]
                            new_val = 1 - old_val
                            color[var_idx] = new_val
                            new_red_k4, new_blue_k5 = count_violations_local(color)
                            color[var_idx] = old_val
                            delta = (new_red_k4 - current_red_k4) + 3 * (new_blue_k5 - current_blue_k5)
                            if best_delta is None or delta < best_delta:
                                best_delta = delta
                                best_var = var_idx
                        var_idx = best_var
                    
                    old_val = color[var_idx]
                    new_val = 1 - old_val
                    color[var_idx] = new_val
                    new_red_k4, new_blue_k5 = count_violations_local(color)
                    delta = (new_red_k4 - current_red_k4) + 3 * (new_blue_k5 - current_blue_k5)
                    
                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        current_red_k4, current_blue_k5 = new_red_k4, new_blue_k5
                        if current_red_k4 + 3 * current_blue_k5 < best_red_k4 + 3 * best_blue_k5:
                            best_red_k4, best_blue_k5 = current_red_k4, current_blue_k5
                            best_color = color.copy()
                            if best_red_k4 == 0 and best_blue_k5 == 0:
                                return best_color.tolist(), 0
                    else:
                        color[var_idx] = old_val
                    
                    T *= cooling
                    
                    if step % 10000 == 0 and best_red_k4 == 0 and best_blue_k5 == 0:
                        return best_color.tolist(), 0
                
                return best_color.tolist(), best_red_k4 + 3 * best_blue_k5
            
            colors, cost = solve_walksat_with_init(n, current_deadline, init_color)
            
            if cost == 0:
                if is_valid(n, colors):
                    best_found = (n, colors)
                    found_valid = True
                    break
        
        if found_valid:
            break

    return {'n': best_found[0], 'colors': best_found[1]}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
