import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Helper to check if (m_idx, r) is compatible with a set of (m_idx, r)
    def is_compatible(m_idx, r, selected_indices, selected_residues):
        for k in range(len(selected_indices)):
            m_old_idx = selected_indices[k]
            r_old = selected_residues[k]
            g = gcds[m_idx][m_old_idx]
            if (r - r_old) % g == 0:
                return False
        return True

    def get_score(selected_indices, selected_residues):
        return sum(1.0 / moduli[idx] for idx in selected_indices)

    # Initial Solution: Greedy
    # We want to pick small moduli.
    # Sort moduli by value (ascending)
    # For each m, try to find a residue.
    
    def initial_greedy():
        selected_indices = []
        selected_residues = []
        # Try to add in order of increasing m
        for m_idx in range(n_moduli):
            m = moduli[m_idx]
            # Try all residues
            # To make it deterministic but varied, we can shuffle residues
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(m_idx, r, selected_indices, selected_residues):
                    selected_indices.append(m_idx)
                    selected_residues.append(r)
                    break
        return selected_indices, selected_residues

    # Simulated Annealing
    def simulated_annealing(initial_indices, initial_residues):
        current_indices = list(initial_indices)
        current_residues = list(initial_residues)
        current_score = get_score(current_indices, current_residues)
        
        best_indices = list(current_indices)
        best_residues = list(current_residues)
        best_score = current_score
        
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        
        while T > min_T and time.time() < deadline:
            # Propose a move
            move_type = random.random()
            
            new_indices = list(current_indices)
            new_residues = list(current_residues)
            
            if move_type < 0.4:
                # Add a random modulus not in set
                if len(current_indices) < n_moduli:
                    # Pick a random modulus not in current_indices
                    # Optimization: maintain a set of used indices? 
                    # For now, just pick a random index and check if it's in set.
                    # Since n_moduli is 181, and set size is maybe 50-100, 
                    # we can just try a few random picks.
                    added = False
                    for _ in range(10): # try up to 10 times to find an unused one
                        m_idx = random.randint(0, n_moduli - 1)
                        if m_idx not in new_indices:
                            m = moduli[m_idx]
                            # Find a valid residue
                            residues = list(range(m))
                            random.shuffle(residues)
                            for r in residues:
                                if is_compatible(m_idx, r, new_indices, new_residues):
                                    new_indices.append(m_idx)
                                    new_residues.append(r)
                                    added = True
                                    break
                            if added:
                                break
                    if not added:
                        # Move failed, skip
                        continue
            elif move_type < 0.8:
                # Change residue of a random existing modulus
                if current_indices:
                    pos = random.randint(0, len(current_indices) - 1)
                    m_idx = current_indices[pos]
                    m = moduli[m_idx]
                    old_r = current_residues[pos]
                    # Pick a new residue different from old_r
                    # To ensure validity, we must check compatibility with OTHERS.
                    # So we temporarily remove it, pick a new r, and check.
                    temp_indices = new_indices[:pos] + new_indices[pos+1:]
                    temp_residues = new_residues[:pos] + new_residues[pos+1:]
                    
                    residues = list(range(m))
                    residues.remove(old_r) # Ensure it changes
                    random.shuffle(residues)
                    
                    found = False
                    for r in residues:
                        if is_compatible(m_idx, r, temp_indices, temp_residues):
                            new_indices[pos] = m_idx # It was already m_idx
                            new_residues[pos] = r
                            found = True
                            break
                    if not found:
                        continue
            else:
                # Remove a random modulus
                if current_indices:
                    pos = random.randint(0, len(current_indices) - 1)
                    new_indices.pop(pos)
                    new_residues.pop(pos)
            
            new_score = get_score(new_indices, new_residues)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_indices = new_indices
                current_residues = new_residues
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_indices = list(current_indices)
                    best_residues = list(current_residues)
            
            T *= cooling_rate
            
        return best_indices, best_residues, best_score

    # Run multiple restarts
    best_overall_score = 0.0
    best_overall_indices = []
    best_overall_residues = []
    
    # First, run a few greedy initializations to find a good start
    # Then run SA from the best start.
    
    # To save time, we can run SA for a fixed number of steps or time.
    # Let's run SA in a loop, restarting if stuck or time allows.
    
    # A better strategy: Run SA once for a long time, or multiple short SAs?
    # Given 115s, we can run one long SA with periodic restarts or just long cooling.
    # Let's do multiple restarts of SA, each with a shorter budget, to explore different basins.
    
    # Initial greedy
    init_indices, init_residues = initial_greedy()
    
    # Run SA
    # We will run SA for ~100 seconds total.
    # Split into chunks? Or just one long run?
    # One long run with adaptive cooling is usually fine.
    
    # Let's optimize the "Add" move probability. 
    # Since we want to maximize sum, we should bias towards adding small moduli.
    # But "Add" picks random modulus. 
    # Let's modify "Add" to pick from a weighted distribution favoring small moduli?
    # Or just rely on the fact that small moduli are more likely to be compatible?
    # Actually, small moduli have fewer residues, so they are "tighter".
    # Let's stick to random for now, but maybe weight the selection of m_idx in Add move.
    
    # Let's refine the Add move to prefer smaller moduli.
    # In the loop `for _ in range(10)`, we can pick m_idx with probability proportional to 1/moduli[m_idx]?
    # No, just picking random is okay if we do many steps.
    
    # Let's run the SA.
    
    # To speed up `is_compatible`, we can use a list of tuples and iterate.
    # The current implementation uses two parallel lists.
    
    # One optimization: 
    # If we add a modulus, we should probably try to add the smallest available one?
    # No, SA handles that.
    
    # Let's just run the SA.
    
    # To ensure we don't timeout, we check time in the main loop.
    
    # We'll do 10 restarts, each with ~10s budget.
    num_restarts = 10
    time_per_restart = 10.0
    
    for i in range(num_restarts):
        restart_start = time.time()
        restart_deadline = restart_start + time_per_restart
        
        # New initial solution
        init_indices, init_residues = initial_greedy()
        
        # Run SA with deadline
        # We need to pass the deadline to SA or modify SA to check global time.
        # Let's modify SA to take a deadline.
        
        # Actually, the previous SA function uses a local deadline. 
        # Let's rewrite SA to accept a deadline.
        
        def run_sa_with_deadline(init_indices, init_residues, deadline):
            current_indices = list(init_indices)
            current_residues = list(init_residues)
            current_score = get_score(current_indices, current_residues)
            
            best_indices = list(current_indices)
            best_residues = list(current_residues)
            best_score = current_score
            
            T = 1.0
            cooling_rate = 0.999
            min_T = 0.001
            
            while T > min_T and time.time() < deadline:
                move_type = random.random()
                
                new_indices = list(current_indices)
                new_residues = list(current_residues)
                
                if move_type < 0.4:
                    # Add
                    if len(current_indices) < n_moduli:
                        added = False
                        # Try to add a random modulus
                        # Bias towards smaller moduli?
                        # Let's pick a random index from the full range, 
                        # but reject if already in set.
                        for _ in range(20):
                            m_idx = random.randint(0, n_moduli - 1)
                            if m_idx not in new_indices:
                                m = moduli[m_idx]
                                residues = list(range(m))
                                random.shuffle(residues)
                                for r in residues:
                                    if is_compatible(m_idx, r, new_indices, new_residues):
                                        new_indices.append(m_idx)
                                        new_residues.append(r)
                                        added = True
                                        break
                                if added:
                                    break
                        if not added:
                            continue
                elif move_type < 0.8:
                    # Change Residue
                    if current_indices:
                        pos = random.randint(0, len(current_indices) - 1)
                        m_idx = current_indices[pos]
                        m = moduli[m_idx]
                        old_r = current_residues[pos]
                        
                        temp_indices = new_indices[:pos] + new_indices[pos+1:]
                        temp_residues = new_residues[:pos] + new_residues[pos+1:]
                        
                        residues = list(range(m))
                        # Filter out old_r? Not strictly necessary, but ensures change.
                        # If we pick old_r, delta is 0.
                        random.shuffle(residues)
                        
                        found = False
                        for r in residues:
                            if r == old_r: continue
                            if is_compatible(m_idx, r, temp_indices, temp_residues):
                                new_residues[pos] = r
                                found = True
                                break
                        if not found:
                            continue
                else:
                    # Remove
                    if current_indices:
                        pos = random.randint(0, len(current_indices) - 1)
                        new_indices.pop(pos)
                        new_residues.pop(pos)
                
                new_score = get_score(new_indices, new_residues)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_indices = new_indices
                    current_residues = new_residues
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_indices = list(current_indices)
                        best_residues = list(current_residues)
                
                T *= cooling_rate
                
            return best_indices, best_residues, best_score

        res_indices, res_residues, res_score = run_sa_with_deadline(init_indices, init_residues, restart_deadline)
        
        if res_score > best_overall_score:
            best_overall_score = res_score
            best_overall_indices = res_indices
            best_overall_residues = res_residues
            
        if time.time() > deadline:
            break

    # Convert to required format
    result_classes = []
    for i in range(len(best_overall_indices)):
        m_idx = best_overall_indices[i]
        r = best_overall_residues[i]
        m = moduli[m_idx]
        result_classes.append([m, r])
        
    return result_classes
