import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    # 1. Greedy Initial Solution
    chosen = []
    counts = [0] * (max_sum + 1)
    
    for x in range(1, N + 1):
        # Check if adding x violates B2[2]
        valid = True
        for y in chosen:
            s = x + y
            if counts[s] == 2:
                valid = False
                break
        if valid and counts[2 * x] == 2:
            valid = False
            
        if valid:
            # Add x
            chosen.append(x)
            counts[2 * x] += 1
            for y in chosen[:-1]:
                counts[x + y] += 1
                
    best_size = len(chosen)
    
    # 2. Local Search (Simulated Annealing) - Optimized
    current = list(chosen)
    in_set = [False] * (N + 1)
    for x in current:
        in_set[x] = True
        
    # Re-initialize counts
    counts = [0] * (max_sum + 1)
    for i in range(len(current)):
        x = current[i]
        counts[2 * x] += 1
        for j in range(i + 1, len(current)):
            y = current[j]
            counts[x + y] += 1
            
    current_conflicts = 0
    for s in range(2, max_sum + 1):
        if counts[s] > 2:
            current_conflicts += counts[s] - 2
            
    best_conflicts = current_conflicts
    best_solution = list(current)
    
    T = 1.0
    cooling = 0.999
    
    # Maintain a list of available (not in set) elements for fast add proposals
    available = [x for x in range(1, N + 1) if not in_set[x]]
    
    while time.time() < deadline:
        # Move Generation
        r = random.random()
        if r < 0.4:
            # Try Add (40%)
            if not available:
                continue
            idx = random.randint(0, len(available) - 1)
            x = available[idx]
            
            # Calculate delta conflicts for adding x
            delta_conf = 0
            if counts[2 * x] == 2:
                delta_conf += 1
            for y in current:
                s = x + y
                if counts[s] == 2:
                    delta_conf += 1
            
            if delta_conf == 0:
                # Accept
                in_set[x] = True
                current.append(x)
                counts[2 * x] += 1
                for y in current[:-1]:
                    counts[x + y] += 1
                available.pop(idx)
                current_conflicts += delta_conf
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if best_conflicts == 0:
                        best_solution = list(current)
                        if len(current) > best_size:
                            best_size = len(current)
            else:
                if random.random() < math.exp(-delta_conf / T):
                    in_set[x] = True
                    current.append(x)
                    counts[2 * x] += 1
                    for y in current[:-1]:
                        counts[x + y] += 1
                    available.pop(idx)
                    current_conflicts += delta_conf
                    
        elif r < 0.9:
            # Try Swap (50%)
            if not current:
                continue
            x_idx = random.randint(0, len(current) - 1)
            x = current[x_idx]
            
            if not available:
                continue
            y_idx = random.randint(0, len(available) - 1)
            y = available[y_idx]
            
            # Calculate delta for swap: remove x, add y
            # Sums removed: 2x, x+z for z in current, z != x
            # Sums added: 2y, y+z for z in current, z != x
            
            # Use a dict to track net change per sum
            sum_delta = {}
            
            # Remove x
            sum_delta[2 * x] = sum_delta.get(2 * x, 0) - 1
            for z in current:
                if z != x:
                    s = x + z
                    sum_delta[s] = sum_delta.get(s, 0) - 1
            
            # Add y
            sum_delta[2 * y] = sum_delta.get(2 * y, 0) + 1
            for z in current:
                if z != x:
                    s = y + z
                    sum_delta[s] = sum_delta.get(s, 0) + 1
            
            # Calculate conflict delta
            delta_conf = 0
            for s, d in sum_delta.items():
                if d == 0:
                    continue
                old_c = counts[s]
                new_c = old_c + d
                if old_c > 2 and new_c <= 2:
                    delta_conf -= (old_c - 2)
                elif old_c <= 2 and new_c > 2:
                    delta_conf += (new_c - 2)
                elif old_c > 2 and new_c > 2:
                    delta_conf += (new_c - 2) - (old_c - 2)
            
            if delta_conf <= 0:
                # Accept
                current[x_idx] = y
                in_set[x] = False
                in_set[y] = True
                
                # Update counts
                for s, d in sum_delta.items():
                    if d != 0:
                        counts[s] += d
                        
                # Update available list
                available[y_idx] = x
                current_conflicts += delta_conf
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if best_conflicts == 0:
                        best_solution = list(current)
                        if len(current) > best_size:
                            best_size = len(current)
            else:
                if random.random() < math.exp(-delta_conf / T):
                    current[x_idx] = y
                    in_set[x] = False
                    in_set[y] = True
                    for s, d in sum_delta.items():
                        if d != 0:
                            counts[s] += d
                    available[y_idx] = x
                    current_conflicts += delta_conf
        else:
            # Try Remove (10%) - helps escape local minima
            if not current:
                continue
            x_idx = random.randint(0, len(current) - 1)
            x = current[x_idx]
            
            # Calculate delta for removing x
            delta_conf = 0
            if counts[2 * x] == 3:
                delta_conf -= 1
            for z in current:
                if z != x:
                    s = x + z
                    if counts[s] == 3:
                        delta_conf -= 1
            
            if delta_conf < 0:
                # Accept removal
                # Remove x from current
                current.pop(x_idx)
                in_set[x] = False
                
                # Update counts
                counts[2 * x] -= 1
                for z in current:
                    counts[x + z] -= 1
                    
                # Add x back to available
                available.append(x)
                current_conflicts += delta_conf
            else:
                if random.random() < math.exp(-delta_conf / T):
                    current.pop(x_idx)
                    in_set[x] = False
                    counts[2 * x] -= 1
                    for z in current:
                        counts[x + z] -= 1
                    available.append(x)
                    current_conflicts += delta_conf
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final verification
    final_counts = [0] * (max_sum + 1)
    for i in range(len(best_solution)):
        x = best_solution[i]
        final_counts[2 * x] += 1
        for j in range(i + 1, len(best_solution)):
            y = best_solution[j]
            final_counts[x + y] += 1
            
    valid = True
    for s in range(2, max_sum + 1):
        if final_counts[s] > 2:
            valid = False
            break
            
    if valid:
        return best_solution
    else:
        # Fallback: rebuild from greedy
        return chosen

# EVOLVE-BLOCK-END
