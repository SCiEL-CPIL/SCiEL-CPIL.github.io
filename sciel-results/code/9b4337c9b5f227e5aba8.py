import math
import time
import random

def construct():
    deadline = time.time() + 145
    
    min_m = 20
    max_m = 200
    
    # Precompute gcd
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    def is_valid(m, r, selected):
        """Check if (m, r) is valid against selected classes."""
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)

    def greedy_construct():
        """Greedy construction: iterate moduli in increasing order, pick first valid residue."""
        selected = []
        for m in range(min_m, max_m + 1):
            found = False
            # Try residues 0..m-1
            # Optimization: If we just need ANY valid residue, we can stop early.
            # To maximize flexibility, maybe randomize order? No, deterministic is safer for reproducibility, 
            # but we want to find a valid one quickly.
            # Let's try residues in a pseudo-random order or just 0..m-1.
            # 0..m-1 is fine.
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append((m, r))
                    found = True
                    break
            if time.time() > deadline:
                break
        return selected

    def local_search(selected, deadline):
        """Improve the solution via local moves."""
        best_sol = [list(x) for x in selected]
        best_score = calc_score(best_sol)
        
        # Helper to check if a set of tuples is valid
        def is_set_valid(sol):
            for i in range(len(sol)):
                for j in range(i+1, len(sol)):
                    m1, r1 = sol[i]
                    m2, r2 = sol[j]
                    if r1 % gcd_table[m1][m2] == r2 % gcd_table[m1][m2]:
                        return False
            return True
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing modulus
            current_moduli = set(m for m, _ in best_sol)
            for m in range(min_m, max_m + 1):
                if m in current_moduli:
                    continue
                # Try to find a valid residue for m
                for r in range(m):
                    if is_valid(m, r, best_sol):
                        best_sol.append((m, r))
                        new_score = calc_score(best_sol)
                        if new_score > best_score:
                            best_score = new_score
                            improved = True
                            break # Restart loop to pick up other additions? Or continue?
                            # Continue to see if we can add more
                        break # If we found one, we added it. No point trying other r for this m.
                if time.time() > deadline:
                    return best_sol
            
            # 2. Try to swap a large modulus for a smaller one (if it fits)
            # Sort by modulus descending to try removing the "worst" contributors first?
            # Or just iterate all.
            # Actually, replacing a large m with a small m increases score.
            # So we want to remove a large m and add a small m.
            
            # Let's try: Remove (m_large, r_large). Then try to add any missing small m.
            # If we can add a small m, and 1/m_small > 1/m_large, it's an improvement.
            
            # Iterate over current moduli
            indices = list(range(len(best_sol)))
            random.shuffle(indices) # Randomize order to escape local minima
            
            for idx in indices:
                if time.time() > deadline:
                    return best_sol
                
                m_old, r_old = best_sol[idx]
                
                # Temporarily remove
                temp_sol = [list(x) for x in best_sol if x != best_sol[idx]]
                
                # Try to add any missing modulus
                # Prefer smaller moduli
                for m_new in range(min_m, max_m + 1):
                    if m_new == m_old:
                        continue
                    if any(m == m_new for m, _ in temp_sol):
                        continue
                    
                    # Check if adding m_new is better than m_old
                    if 1.0 / m_new <= 1.0 / m_old:
                        continue # No improvement in score, skip (unless it enables further additions? 
                                 # For simplicity, we only accept if score increases immediately)
                    
                    # Try to find valid residue for m_new
                    found_r = -1
                    for r in range(m_new):
                        if is_valid(m_new, r, temp_sol):
                            found_r = r
                            break
                    
                    if found_r != -1:
                        # Accept the move
                        temp_sol.append((m_new, found_r))
                        new_score = calc_score(temp_sol)
                        if new_score > best_score:
                            best_sol = temp_sol
                            best_score = new_score
                            improved = True
                            break # Restart local search with new best
                            
            # 3. Residue Optimization: Try to change residue of existing moduli to allow adding new ones?
            # This is complex. Let's stick to Add and Swap for now.
            # The "Add" step should catch most gains if we start with a good greedy.
            
        return best_sol

    # 1. Generate initial solution
    sol = greedy_construct()
    
    # 2. Polish
    sol = local_search(sol, deadline)
    
    # 3. Simulated Annealing for global escape if time permits
    # Convert to list of lists for mutability in SA
    current = [list(x) for x in sol]
    current_score = calc_score(current)
    best = [list(x) for x in sol]
    best_score = current_score
    
    T = 0.1
    cooling = 0.999
    min_T = 1e-5
    
    while time.time() < deadline - 5:
        if T < min_T:
            break
            
        # Random Move
        new_sol = [list(x) for x in current]
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            m_add = random.randint(min_m, max_m)
            if any(m == m_add for m, _ in new_sol):
                continue
            r_add = random.randint(0, m_add - 1)
            if is_valid(m_add, r_add, new_sol):
                new_sol.append([m_add, r_add])
        elif move_type < 0.7:
            # Remove
            if not new_sol: continue
            idx = random.randint(0, len(new_sol) - 1)
            new_sol.pop(idx)
        elif move_type < 0.9:
            # Change Residue
            if not new_sol: continue
            idx = random.randint(0, len(new_sol) - 1)
            m, r = new_sol[idx]
            new_r = random.randint(0, m - 1)
            if new_r != r and is_valid(m, new_r, new_sol):
                new_sol[idx][1] = new_r
        else:
            # Swap Modulus (Remove one, Add another)
            if not new_sol: continue
            idx = random.randint(0, len(new_sol) - 1)
            m_old, _ = new_sol[idx]
            new_sol.pop(idx)
            
            m_new = random.randint(min_m, max_m)
            if m_new == m_old or any(m == m_new for m, _ in new_sol):
                # Revert
                new_sol.insert(idx, [m_old, _])
                continue
            
            r_new = random.randint(0, m_new - 1)
            if is_valid(m_new, r_new, new_sol):
                new_sol.append([m_new, r_new])
            else:
                # Revert
                new_sol.insert(idx, [m_old, _])
                continue
        
        new_score = calc_score(new_sol)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current = new_sol
            current_score = new_score
            if current_score > best_score:
                best = [list(x) for x in current]
                best_score = current_score
        
        T *= cooling
        
    # Final Polish
    best = local_search(best, deadline)
    
    return best

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
