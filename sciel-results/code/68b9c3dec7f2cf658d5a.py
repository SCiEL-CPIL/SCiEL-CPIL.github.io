import random
import time
import math

def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    
    def is_compatible(r, m, existing_set):
        for r2, m2 in existing_set:
            g = math.gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_valid_residues(m, existing_set):
        valid = []
        for r in range(m):
            if is_compatible(r, m, existing_set):
                valid.append(r)
        return valid
    
    def greedy_build(order):
        current_set = []
        for m in order:
            valid_residues = find_valid_residues(m, current_set)
            if valid_residues:
                current_set.append([random.choice(valid_residues), m])
        return current_set
    
    def local_search_improve(solution, max_iterations=1000):
        """Try remove-one-add-two moves to improve the solution."""
        current = [pair[:] for pair in solution]
        current_size = len(current)
        
        for _ in range(max_iterations):
            if time.time() - start_time > max_time - 5.0:
                break
                
            # Try remove-one-add-two: remove one element, try to add two others
            improved = False
            for i in range(len(current)):
                if time.time() - start_time > max_time - 5.0:
                    break
                    
                removed = current[i]
                temp_set = [pair[:] for pair in current[:i] + current[i+1:]]
                
                # Try to add two moduli from the remaining pool
                used_moduli = {m for _, m in temp_set}
                available_moduli = [m for m in moduli if m not in used_moduli]
                
                # Try pairs of available moduli
                for j in range(len(available_moduli)):
                    if time.time() - start_time > max_time - 5.0:
                        break
                    m1 = available_moduli[j]
                    valid_r1 = find_valid_residues(m1, temp_set)
                    if not valid_r1:
                        continue
                    
                    for r1 in valid_r1[:3]:  # Limit to first 3 for speed
                        temp_set2 = temp_set + [[r1, m1]]
                        
                        # Now try to add one more
                        used_moduli2 = {m for _, m in temp_set2}
                        remaining = [m for m in available_moduli if m not in used_moduli2 and m != m1]
                        
                        for m2 in remaining[:20]:  # Limit search
                            valid_r2 = find_valid_residues(m2, temp_set2)
                            if valid_r2:
                                new_solution = temp_set2 + [[random.choice(valid_r2), m2]]
                                if len(new_solution) > current_size:
                                    current = new_solution
                                    current_size = len(current)
                                    improved = True
                                    break
                        
                        if improved:
                            break
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current
    
    # Initial greedy with fixed order to get a baseline
    current_set = greedy_build(moduli)
    best_solution = current_set
    best_size = len(current_set)
    
    # Randomized local search / greedy with restarts
    restarts = 0
    while time.time() - start_time < max_time:
        restarts += 1
        random.shuffle(moduli)
        current_set = greedy_build(moduli)
        
        # Apply local search improvement
        improved_set = local_search_improve(current_set, max_iterations=500)
        
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_solution = [pair[:] for pair in improved_set]
            
        # If we've found a large solution, do more intensive local search on it
        if best_size >= 30 and time.time() - start_time < max_time - 10.0:
            further_improved = local_search_improve(best_solution, max_iterations=2000)
            if len(further_improved) > best_size:
                best_size = len(further_improved)
                best_solution = [pair[:] for pair in further_improved]
        
    return best_solution
