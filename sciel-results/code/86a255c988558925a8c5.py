import random
import time
import math

def construct():
    start_time = time.time()
    max_time = 50.0  # Safety margin for 90s limit
    
    max_modulus = 128
    all_moduli = list(range(2, max_modulus + 1))
    
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
    
    def try_add_modulus(m, current_set):
        """Try to find a valid residue for modulus m given current_set."""
        valid = find_valid_residues(m, current_set)
        if valid:
            return random.choice(valid)
        return None
    
    def local_search_improve(solution, time_budget=5.0):
        """Try remove-one-add-two moves to improve the solution."""
        current = [pair[:] for pair in solution]
        current_size = len(current)
        deadline = time.time() + time_budget
        
        improved_any = True
        while improved_any and time.time() < deadline:
            improved_any = False
            used_moduli = {m for _, m in current}
            available_moduli = [m for m in all_moduli if m not in used_moduli]
            
            # Try each removal
            for i in range(len(current)):
                if time.time() >= deadline:
                    break
                    
                removed = current[i]
                temp_set = current[:i] + current[i+1:]
                
                # Try to add two moduli from the remaining pool
                # Sort available by how "easy" they are (fewer conflicts)
                def conflict_count(m):
                    cnt = 0
                    for _, m2 in temp_set:
                        g = math.gcd(m, m2)
                        if g > 1:
                            cnt += 1
                    return cnt
                
                available_sorted = sorted(available_moduli, key=conflict_count)
                
                # Try pairs
                for j in range(min(len(available_sorted), 30)):
                    if time.time() >= deadline:
                        break
                    m1 = available_sorted[j]
                    r1 = try_add_modulus(m1, temp_set)
                    if r1 is None:
                        continue
                    
                    temp_set2 = temp_set + [[r1, m1]]
                    
                    # Now try to add one more
                    used_moduli2 = {m for _, m in temp_set2}
                    remaining = [m for m in available_sorted if m not in used_moduli2 and m != m1]
                    
                    for m2 in remaining[:25]:
                        if time.time() >= deadline:
                            break
                        r2 = try_add_modulus(m2, temp_set2)
                        if r2 is not None:
                            new_solution = temp_set2 + [[r2, m2]]
                            if len(new_solution) > current_size:
                                current = new_solution
                                current_size = len(current)
                                improved_any = True
                                break
                    
                    if improved_any:
                        break
                
                if improved_any:
                    break
        
        return current
    
    def polish_solution(solution, time_budget=3.0):
        """Final polish: try all possible 1-for-2 swaps exhaustively."""
        current = [pair[:] for pair in solution]
        current_size = len(current)
        deadline = time.time() + time_budget
        
        improved_any = True
        while improved_any and time.time() < deadline:
            improved_any = False
            used_moduli = {m for _, m in current}
            available_moduli = [m for m in all_moduli if m not in used_moduli]
            
            for i in range(len(current)):
                if time.time() >= deadline:
                    break
                    
                removed = current[i]
                temp_set = current[:i] + current[i+1:]
                
                # Try all available moduli for first addition
                for m1 in available_moduli:
                    if time.time() >= deadline:
                        break
                    r1 = try_add_modulus(m1, temp_set)
                    if r1 is None:
                        continue
                    
                    temp_set2 = temp_set + [[r1, m1]]
                    used_moduli2 = {m for _, m in temp_set2}
                    remaining = [m for m in available_moduli if m not in used_moduli2 and m != m1]
                    
                    for m2 in remaining:
                        if time.time() >= deadline:
                            break
                        r2 = try_add_modulus(m2, temp_set2)
                        if r2 is not None:
                            new_solution = temp_set2 + [[r2, m2]]
                            if len(new_solution) > current_size:
                                current = new_solution
                                current_size = len(current)
                                improved_any = True
                                break
                    
                    if improved_any:
                        break
                
                if improved_any:
                    break
        
        return current
    
    # Initial greedy with sorted order (smaller moduli first)
    sorted_moduli = sorted(all_moduli)
    current_set = greedy_build(sorted_moduli)
    best_solution = current_set
    best_size = len(current_set)
    
    # Randomized local search / greedy with restarts
    restarts = 0
    while time.time() - start_time < max_time - 8.0:
        restarts += 1
        random.shuffle(all_moduli)
        current_set = greedy_build(all_moduli)
        
        # Apply local search improvement
        improved_set = local_search_improve(current_set, time_budget=2.0)
        
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_solution = [pair[:] for pair in improved_set]
    
    # Final polish on the best solution found
    if time.time() - start_time < max_time - 2.0:
        polished = polish_solution(best_solution, time_budget=5.0)
        if len(polished) > best_size:
            best_size = len(polished)
            best_solution = [pair[:] for pair in polished]
    
    return best_solution
