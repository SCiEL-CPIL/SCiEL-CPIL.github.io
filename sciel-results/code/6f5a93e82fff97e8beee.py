import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Pre-compute gcd table for moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd_table[i][j] = gcd(moduli[i], moduli[j])
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def compute_score(selected):
        """Compute sum of 1/m for selected classes."""
        return sum(1.0 / m for m, r in selected)
    
    def is_compatible(m, r, selected):
        """Check if (m, r) is compatible with all in selected."""
        idx = mod_to_idx[m]
        for m_i, r_i in selected:
            idx_i = mod_to_idx[m_i]
            g = gcd_table[idx][idx_i]
            if (r - r_i) % g == 0:
                return False
        return True
    
    def greedy_solution():
        """Greedy: add moduli in increasing order, first compatible residue."""
        selected = []
        for m in moduli:
            for r in range(m):
                if is_compatible(m, r, selected):
                    selected.append([m, r])
                    break
        return selected
    
    def simulated_annealing(initial, max_time):
        """SA to improve the solution."""
        if time.time() > max_time:
            return initial
        
        current = [list(x) for x in initial]
        current_score = compute_score(current)
        best = [list(x) for x in current]
        best_score = current_score
        
        # Temperature schedule
        T_start = 0.1
        T_end = 0.001
        steps = 50000
        
        # Build lookup: modulus -> index in current
        mod_to_pos = {}
        for i, (m, r) in enumerate(current):
            mod_to_pos[m] = i
        
        for step in range(steps):
            if time.time() > max_time:
                break
            
            # Cooling
            frac = step / steps
            T = T_start * (T_end / T_start) ** frac
            
            # Choose a random move type
            move_type = random.random()
            
            if move_type < 0.3:
                # Change residue of a random selected modulus
                if not current:
                    continue
                i = random.randint(0, len(current) - 1)
                m = current[i][0]
                old_r = current[i][1]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                # Check if new residue is compatible
                temp_current = current[:]
                temp_current[i] = [m, new_r]
                # Remove old, check new
                temp_without = temp_current[:i] + temp_current[i+1:]
                if is_compatible(m, new_r, temp_without):
                    temp_score = compute_score(temp_current)
                    delta = temp_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                        current[i][1] = new_r
                        current_score = temp_score
                        if current_score > best_score:
                            best = [list(x) for x in current]
                            best_score = current_score
                            
            elif move_type < 0.6:
                # Swap: remove one, add another
                if not current:
                    continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                
                # Pick a modulus not in current to add
                available = [m for m in moduli if m not in mod_to_pos]
                if not available:
                    continue
                m_add = random.choice(available)
                
                # Find a compatible residue for m_add
                temp_without = current[:i] + current[i+1:]
                found_r = None
                # Try a few random residues
                for _ in range(min(m_add, 20)):
                    r_try = random.randint(0, m_add - 1)
                    if is_compatible(m_add, r_try, temp_without):
                        found_r = r_try
                        break
                if found_r is None:
                    # Try all
                    for r_try in range(m_add):
                        if is_compatible(m_add, r_try, temp_without):
                            found_r = r_try
                            break
                if found_r is None:
                    continue
                
                temp_new = temp_without + [[m_add, found_r]]
                temp_score = compute_score(temp_new)
                delta = temp_score - current_score
                if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                    current = temp_new
                    current_score = temp_score
                    # Update mod_to_pos
                    mod_to_pos = {}
                    for j, (m, r) in enumerate(current):
                        mod_to_pos[m] = j
                    if current_score > best_score:
                        best = [list(x) for x in current]
                        best_score = current_score
                        
            else:
                # Remove a random element (if score doesn't drop too much)
                if len(current) <= 1:
                    continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                temp_new = current[:i] + current[i+1:]
                temp_score = compute_score(temp_new)
                delta = temp_score - current_score
                if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                    current = temp_new
                    current_score = temp_score
                    # Update mod_to_pos
                    mod_to_pos = {}
                    for j, (m, r) in enumerate(current):
                        mod_to_pos[m] = j
                    if current_score > best_score:
                        best = [list(x) for x in current]
                        best_score = current_score
        
        return best
    
    # Get initial greedy solution
    initial = greedy_solution()
    initial_score = compute_score(initial)
    
    # Run SA with multiple restarts
    best_overall = [list(x) for x in initial]
    best_overall_score = initial_score
    
    # Use remaining time for SA
    sa_deadline = deadline - 5.0  # Leave 5s buffer
    
    # Multiple SA runs with different random seeds
    n_restarts = 10
    time_per_restart = (sa_deadline - time.time()) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > sa_deadline:
            break
        
        # For first restart, use the greedy solution
        if restart == 0:
            start_sol = initial
        else:
            # Perturb the best solution
            start_sol = [list(x) for x in best_overall]
            # Randomly change a few residues
            for _ in range(min(3, len(start_sol))):
                if not start_sol:
                    break
                i = random.randint(0, len(start_sol) - 1)
                m = start_sol[i][0]
                # Try to find a new compatible residue
                temp_without = start_sol[:i] + start_sol[i+1:]
                for r_try in range(m):
                    if is_compatible(m, r_try, temp_without) and r_try != start_sol[i][1]:
                        start_sol[i][1] = r_try
                        break
        
        # Run SA
        sa_deadline_this = min(sa_deadline, time.time() + time_per_restart)
        result = simulated_annealing(start_sol, sa_deadline_this)
        
        if compute_score(result) > best_overall_score:
            best_overall = result
            best_overall_score = compute_score(result)
    
    return best_overall
