import random
import time
import math

def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    
    best_solution = []
    best_size = 0
    
    # Initial greedy with fixed order to get a baseline
    # We try to add moduli in increasing order
    current_set = []
    for m in moduli:
        valid_residues = []
        for r in range(m):
            compatible = True
            for r2, m2 in current_set:
                g = math.gcd(m, m2)
                if (r - r2) % g == 0:
                    compatible = False
                    break
            if compatible:
                valid_residues.append(r)
        
        if valid_residues:
            # Pick the first valid residue
            current_set.append([valid_residues[0], m])
            
    best_solution = current_set
    best_size = len(current_set)
    
    def local_search_improve(sol, time_limit):
        """Try to improve solution using remove-one-add-two and remove-one-add-one moves."""
        sol = [pair[:] for pair in sol]
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            n = len(sol)
            # Try remove-one-add-two
            for i in range(n):
                if time.time() - start_time >= time_limit:
                    break
                # Remove sol[i]
                temp_sol = sol[:i] + sol[i+1:]
                # Try to add two new elements
                added = 0
                new_elements = []
                for m in moduli:
                    if m in [p[1] for p in temp_sol] or m in [p[1] for p in new_elements]:
                        continue
                    valid_residues = []
                    for r in range(m):
                        compatible = True
                        for r2, m2 in temp_sol:
                            g = math.gcd(m, m2)
                            if (r - r2) % g == 0:
                                compatible = False
                                break
                        if compatible:
                            for r3, m3 in new_elements:
                                g = math.gcd(m, m3)
                                if (r - r3) % g == 0:
                                    compatible = False
                                    break
                        if compatible:
                            valid_residues.append(r)
                    if valid_residues and added < 2:
                        new_elements.append([random.choice(valid_residues), m])
                        added += 1
                if added == 2:
                    sol = temp_sol + new_elements
                    improved = True
                    break
            # Try remove-one-add-one
            if not improved:
                for i in range(n):
                    if time.time() - start_time >= time_limit:
                        break
                    temp_sol = sol[:i] + sol[i+1:]
                    for m in moduli:
                        if m in [p[1] for p in temp_sol]:
                            continue
                        valid_residues = []
                        for r in range(m):
                            compatible = True
                            for r2, m2 in temp_sol:
                                g = math.gcd(m, m2)
                                if (r - r2) % g == 0:
                                    compatible = False
                                    break
                            if compatible:
                                valid_residues.append(r)
                        if valid_residues:
                            # Check if this gives net gain (we removed 1, add 1, so no gain)
                            # Only do this if it allows future improvements - skip for now
                            pass
                    # Try just adding one more without removing
                    for m in moduli:
                        if m in [p[1] for p in sol]:
                            continue
                        valid_residues = []
                        for r in range(m):
                            compatible = True
                            for r2, m2 in sol:
                                g = math.gcd(m, m2)
                                if (r - r2) % g == 0:
                                    compatible = False
                                    break
                            if compatible:
                                valid_residues.append(r)
                        if valid_residues:
                            sol.append([random.choice(valid_residues), m])
                            improved = True
                            break
                    if improved:
                        break
        return sol
    
    # Randomized local search / greedy with restarts
    restarts = 0
    while time.time() - start_time < max_time:
        restarts += 1
        random.shuffle(moduli)
        current_set = []
        
        for m in moduli:
            valid_residues = []
            for r in range(m):
                compatible = True
                for r2, m2 in current_set:
                    g = math.gcd(m, m2)
                    if (r - r2) % g == 0:
                        compatible = False
                        break
                if compatible:
                    valid_residues.append(r)
            
            if valid_residues:
                # Pick a random valid residue to diversify
                current_set.append([random.choice(valid_residues), m])
        
        # Apply local search improvement
        if time.time() - start_time < max_time - 5.0:
            current_set = local_search_improve(current_set, max_time - 2.0)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = [pair[:] for pair in current_set]
            
        # Also try local search on the best solution found so far
        if time.time() - start_time < max_time - 5.0:
            improved_best = local_search_improve(best_solution, max_time - 2.0)
            if len(improved_best) > best_size:
                best_size = len(improved_best)
                best_solution = [pair[:] for pair in improved_best]
        
    return best_solution
