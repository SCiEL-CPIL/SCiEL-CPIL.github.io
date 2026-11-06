import random
import time
import math

def construct():
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    def evaluate(A_set):
        """
        Returns (uncovered_count, peak_multiplicity)
        """
        elems = sorted(A_set)
        counts = [0] * num_targets
        
        for i in range(len(elems)):
            ei = elems[i]
            for j in range(i, len(elems)):
                s = ei + elems[j]
                if target_lo <= s <= target_hi:
                    counts[s - target_lo] += 1
        
        uncovered = 0
        peak = 0
        for c in counts:
            if c == 0:
                uncovered += 1
            else:
                if c > peak:
                    peak = c
        return uncovered, peak

    best_A = None
    best_score = float('inf')
    
    start_time = time.time()
    max_time = 50.0 # Leave 10s buffer
    
    # Use multiple restarts to escape local minima
    for restart in range(15):
        if time.time() - start_time > max_time:
            break
            
        # Initialize A
        # Start with a random set of size ~25
        A = set()
        # Add 0 and 128 to anchor the ends
        A.add(0)
        A.add(128)
        
        # Add random elements
        for _ in range(20):
            A.add(random.randint(0, 128))
            
        current_A = set(A)
        unc, peak = evaluate(current_A)
        current_energy = unc * 10000 + peak * 1000
        
        T = 100.0
        min_T = 0.01
        cooling = 0.995
        
        # Simulated Annealing Loop
        while time.time() - start_time < max_time:
            if T < min_T:
                break
                
            # Generate neighbor
            move_type = random.random()
            new_A = set(current_A)
            valid_move = False
            
            if move_type < 0.8:
                # Swap
                if len(new_A) > 0:
                    a = random.choice(list(new_A))
                    b = random.randint(0, 128)
                    if b not in new_A:
                        new_A.remove(a)
                        new_A.add(b)
                        valid_move = True
            elif move_type < 0.9:
                # Add
                b = random.randint(0, 128)
                if b not in new_A:
                    new_A.add(b)
                    valid_move = True
            else:
                # Remove
                if len(new_A) > 2:
                    a = random.choice(list(new_A))
                    new_A.remove(a)
                    valid_move = True
            
            if not valid_move:
                T *= cooling
                continue
                
            new_unc, new_peak = evaluate(new_A)
            new_energy = new_unc * 10000 + new_peak * 1000
            
            delta = new_energy - current_energy
            
            # Accept if better or with prob exp(-delta/T)
            if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                current_A = new_A
                current_energy = new_energy
                
                # Update global best
                if current_energy < best_score:
                    best_score = current_energy
                    best_A = set(current_A)
            
            T *= cooling
            
        # End of SA for this restart
        
        # Quick local polish on the best found so far
        if best_A is not None:
             improved = True
             polish_start = time.time()
             while improved and (time.time() - polish_start < 2.0) and (time.time() - start_time < max_time):
                 improved = False
                 best_unc, best_peak = evaluate(best_A)
                 base_energy = best_unc * 10000 + best_peak * 1000
                 
                 # Try swaps
                 found_better = False
                 for a in list(best_A):
                     if found_better: break
                     for b in range(129):
                         if b in best_A:
                             continue
                         new_A = set(best_A)
                         new_A.remove(a)
                         new_A.add(b)
                         n_unc, n_peak = evaluate(new_A)
                         n_energy = n_unc * 10000 + n_peak * 1000
                         if n_energy < base_energy:
                             best_A = new_A
                             improved = True
                             found_better = True
                             break
                             
                 # Try removes
                 if not improved:
                     for a in list(best_A):
                         if len(best_A) <= 2: break
                         new_A = set(best_A)
                         new_A.remove(a)
                         n_unc, n_peak = evaluate(new_A)
                         n_energy = n_unc * 10000 + n_peak * 1000
                         if n_energy < base_energy:
                             best_A = new_A
                             improved = True
                             break
                             
                 # Try adds
                 if not improved:
                     for b in range(129):
                         if b in best_A:
                             continue
                         new_A = set(best_A)
                         new_A.add(b)
                         n_unc, n_peak = evaluate(new_A)
                         n_energy = n_unc * 10000 + n_peak * 1000
                         if n_energy < base_energy:
                             best_A = new_A
                             improved = True
                             break

    if best_A is None:
        # Fallback: A set that is guaranteed to work
        return list(range(0, 129))
        
    return sorted(best_A)
