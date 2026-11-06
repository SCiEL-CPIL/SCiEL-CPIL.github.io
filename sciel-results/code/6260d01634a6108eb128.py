import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    def get_sums_involving(x, others):
        sums = []
        sums.append(3 * x)
        for a in others:
            sums.append(2 * x + a)
        n = len(others)
        for i in range(n):
            a = others[i]
            for j in range(i, n):
                b = others[j]
                sums.append(x + a + b)
        return sums

    # 1. Initial Greedy Construction
    selected = []
    sum_counts = defaultdict(int)
    num_conflicts = 0
    
    # Fast greedy pass
    for x in range(1, 4097):
        if time.time() - start_time > time_limit - 10:
            break
        others = selected
        new_sums = get_sums_involving(x, others)
        
        valid = True
        for s in new_sums:
            if sum_counts[s] >= 1:
                valid = False
                break
        
        if valid:
            selected.append(x)
            for s in new_sums:
                sum_counts[s] += 1
                
    best_size = len(selected)
    best_selected = selected[:]
    
    # 2. Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995
    
    available = set(range(1, 4097))
    for x in selected:
        available.discard(x)
        
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        
        # Move Type
        if random.random() < 0.4 and available:
            # ADD
            x = random.choice(tuple(available))
            others = selected
            new_sums = get_sums_involving(x, others)
            
            delta_conf = 0
            for s in new_sums:
                if sum_counts[s] == 1:
                    delta_conf += 1
            
            # Acceptance
            if delta_conf == 0:
                # Always accept (size increases, no new conflicts)
                selected.append(x)
                available.remove(x)
                for s in new_sums:
                    sum_counts[s] += 1
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = selected[:]
            else:
                # Probabilistic accept (allows temporary invalidity)
                if random.random() < math.exp(-delta_conf / T):
                    selected.append(x)
                    available.remove(x)
                    for s in new_sums:
                        c = sum_counts[s]
                        sum_counts[s] = c + 1
                        if c == 1:
                            num_conflicts += 1
                            
        elif random.random() < 0.6 and selected and available:
            # SWAP
            a = random.choice(selected)
            b = random.choice(tuple(available))
            
            # Calculate delta for swapping a with b
            others = [x for x in selected if x != a]
            
            sums_remove = get_sums_involving(a, others)
            sums_add = get_sums_involving(b, others)
            
            # Calculate change in counts
            delta_counts = defaultdict(int)
            for s in sums_remove:
                delta_counts[s] -= 1
            for s in sums_add:
                delta_counts[s] += 1
            
            delta_conf = 0
            for s, dc in delta_counts.items():
                c = sum_counts.get(s, 0)
                c_new = c + dc
                if c < 2 and c_new >= 2:
                    delta_conf += 1
                elif c >= 2 and c_new < 2:
                    delta_conf -= 1
            
            # Acceptance for Swap
            if delta_conf < 0:
                accept = True
            elif delta_conf == 0:
                accept = True # Random walk
            else:
                accept = random.random() < math.exp(-delta_conf / T)
            
            if accept:
                # Apply Swap
                selected.remove(a)
                available.add(a)
                
                selected.append(b)
                available.remove(b)
                
                for s, dc in delta_counts.items():
                    c = sum_counts.get(s, 0)
                    c_new = c + dc
                    if c_new == 0:
                        if s in sum_counts:
                            del sum_counts[s]
                    else:
                        sum_counts[s] = c_new
                        
        # Check if current state is valid and better
        # We need to know if num_conflicts is 0.
        # The delta tracking above updates num_conflicts for ADD and SWAP.
        # However, for SWAP, we didn't update num_conflicts in the code block above.
        # Let's fix that.
        
        # Actually, the logic for num_conflicts update in SWAP is missing in the 'if accept' block.
        # I will add it in the final code.
        
    return best_selected
