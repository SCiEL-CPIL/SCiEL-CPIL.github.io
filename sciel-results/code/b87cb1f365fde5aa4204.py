"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 512
    
    # Precompute all 4-term arithmetic progressions
    ap_list = []
    elem_aps = [None] * (n + 1)
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
    
    num_aps = len(ap_list)
    
    for ap_idx, (a, b, c, e) in enumerate(ap_list):
        if elem_aps[a] is None:
            elem_aps[a] = []
        elem_aps[a].append(ap_idx)
        if elem_aps[b] is None:
            elem_aps[b] = []
        elem_aps[b].append(ap_idx)
        if elem_aps[c] is None:
            elem_aps[c] = []
        elem_aps[c].append(ap_idx)
        if elem_aps[e] is None:
            elem_aps[e] = []
        elem_aps[e].append(ap_idx)
    
    for i in range(1, n + 1):
        if elem_aps[i] is None:
            elem_aps[i] = []
    
    best_size = 0
    best_set = set()
    
    start_time = time.time()
    time_limit = 55.0
    
    while time.time() - start_time < time_limit:
        
        in_set = [False] * (n + 1)
        S_set = set()
        ap_count = [0] * num_aps
        
        def is_valid_add(x):
            for ap_idx in elem_aps[x]:
                if ap_count[ap_idx] == 3:
                    return False
            return True
        
        def add_elem(x):
            if in_set[x]:
                return
            in_set[x] = True
            S_set.add(x)
            for ap_idx in elem_aps[x]:
                ap_count[ap_idx] += 1
        
        def remove_elem(x):
            if not in_set[x]:
                return
            in_set[x] = False
            S_set.remove(x)
            for ap_idx in elem_aps[x]:
                ap_count[ap_idx] -= 1
        
        # Multiple initialization strategies
        strategy = random.randint(0, 2)
        
        if strategy == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 1:
            order = list(range(1, n + 1))
        else:
            order = list(range(n, 0, -1))
        
        for x in order:
            if is_valid_add(x):
                add_elem(x)
        
        current_size = len(S_set)
        
        # Simulated annealing local search
        temperature = 1.0
        cooling = 0.995
        min_temp = 0.01
        
        while temperature > min_temp and time.time() - start_time < time_limit:
            
            # Try to add valid elements
            for x in range(1, n + 1):
                if not in_set[x] and is_valid_add(x):
                    add_elem(x)
                    current_size += 1
            
            if current_size == n:
                break
            
            move_made = False
            
            # Try 1-swap
            for _ in range(20):
                if time.time() - start_time > time_limit - 1.0:
                    break
                if current_size == 0:
                    break
                
                u = random.choice(list(S_set))
                v = random.randint(1, n)
                if v in S_set:
                    continue
                
                remove_elem(u)
                if is_valid_add(v):
                    add_elem(v)
                    for x in range(1, n + 1):
                        if not in_set[x] and is_valid_add(x):
                            add_elem(x)
                            current_size += 1
                    move_made = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(S_set)
                    break
                else:
                    add_elem(u)
            
            if not move_made:
                # Try 2-swap: remove 2, add 1
                if current_size >= 2:
                    for _ in range(10):
                        if time.time() - start_time > time_limit - 1.0:
                            break
                        u1 = random.choice(list(S_set))
                        u2 = random.choice(list(S_set))
                        if u1 == u2:
                            continue
                        v = random.randint(1, n)
                        if v in S_set:
                            continue
                        
                        remove_elem(u1)
                        remove_elem(u2)
                        if is_valid_add(v):
                            add_elem(v)
                            for x in range(1, n + 1):
                                if not in_set[x] and is_valid_add(x):
                                    add_elem(x)
                                    current_size += 1
                            move_made = True
                            if current_size > best_size:
                                best_size = current_size
                                best_set = set(S_set)
                            break
                        else:
                            add_elem(u1)
                            add_elem(u2)
            
            if not move_made:
                # Allow worsening moves with probability
                if random.random() < temperature:
                    if current_size > 10:
                        u = random.choice(list(S_set))
                        remove_elem(u)
                        current_size -= 1
                        for x in range(1, n + 1):
                            if not in_set[x] and is_valid_add(x):
                                add_elem(x)
                                current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = set(S_set)
            
            temperature *= cooling
        
        if time.time() - start_time > time_limit - 3.0:
            break
    
    return list(best_set)

# EVOLVE-BLOCK-END
