import numpy as np
import random
import time

N = 51975
DIVISORS = [d for d in range(3, N + 1, 2) if N % d == 0]
DIVISORS_ASC = sorted(DIVISORS)
DIVISORS_DESC = sorted(DIVISORS, reverse=True)

def get_uncovered(covered):
    return N - np.sum(covered)

def greedy_init(order):
    covered = np.zeros(N, dtype=bool)
    classes = []
    for m in order:
        # Find best r
        best_r = 0
        best_gain = -1
        # Optimization: if m is large, sampling might be needed, but N is small enough
        # Actually, iterating m times and summing is O(N) total per m.
        for r in range(m):
            # Gain is number of currently uncovered residues in r::m
            gain = np.sum(~covered[r::m])
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append((m, best_r))
        covered[best_r::m] = True
    return classes, covered

def simulated_annealing(initial_classes, time_limit):
    # Setup
    covered = np.zeros(N, dtype=bool)
    mod_to_res = {}
    for m, r in initial_classes:
        covered[r::m] = True
        mod_to_res[m] = r
    
    used_mods = set(mod_to_res.keys())
    available_mods = [m for m in DIVISORS if m not in used_mods]
    
    current_score = N - np.sum(covered)
    best_score = current_score
    best_classes = list(initial_classes)
    
    T = 5.0
    cooling_rate = 0.999
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if current_score == 0: break
        
        # Move
        move_type = random.random()
        if move_type < 0.7: # Change residue
            if not used_mods: continue
            m = random.choice(list(used_mods))
            old_r = mod_to_res[m]
            new_r = random.randint(0, m - 1)
            if new_r == old_r: continue
            
            # Apply
            # Note: We need to be careful. The new AP might overlap with others.
            # We just set the bits.
            # But we need to revert if rejected.
            # Saving the state of covered is expensive (copying 51975 bytes).
            # Instead, we can calculate the delta?
            # Or just copy covered. 51975 bytes is ~50KB. Copying 1000 times is 50MB. Fine.
            covered_old = covered.copy()
            
            # Unset old, Set new? No, just set new.
            # Wait, if we just set new, we don't unset old.
            # The class (m, old_r) is replaced by (m, new_r).
            # So we must unset old_r::m and set new_r::m.
            covered[old_r::m] = False
            covered[new_r::m] = True
            
            new_score = N - np.sum(covered)
            
            if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                mod_to_res[m] = new_r
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [[m, mod_to_res[m]] for m in sorted(used_mods)]
            else:
                covered[:] = covered_old
        
        elif move_type < 0.85: # Add modulus
            if not available_mods: continue
            m = random.choice(available_mods)
            r = random.randint(0, m - 1)
            
            covered_old = covered.copy()
            covered[r::m] = True
            new_score = N - np.sum(covered)
            
            if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                used_mods.add(m)
                mod_to_res[m] = r
                available_mods.remove(m)
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [[m, mod_to_res[m]] for m in sorted(used_mods)]
            else:
                covered[:] = covered_old
                
        else: # Remove modulus
            if not used_mods: continue
            m = random.choice(list(used_mods))
            r = mod_to_res[m]
            
            covered_old = covered.copy()
            covered[r::m] = False
            new_score = N - np.sum(covered)
            
            if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                used_mods.remove(m)
                del mod_to_res[m]
                available_mods.append(m)
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [[m, mod_to_res[m]] for m in sorted(used_mods)]
            else:
                covered[:] = covered_old
                
        T *= cooling_rate
        if T < 0.01: T = 0.01
        
    return best_classes, best_score

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Phase 1: Greedy
    classes_asc, _ = greedy_init(DIVISORS_ASC)
    score_asc = N - np.sum(np.zeros(N, dtype=bool) | np.concatenate([np.zeros(N, dtype=bool) for _ in range(len(classes_asc))])) # Dummy, need actual score
    # Recalculate score properly
    def calc_score(classes):
        c = np.zeros(N, dtype=bool)
        for m, r in classes:
            c[r::m] = True
        return N - np.sum(c)
        
    score_asc = calc_score(classes_asc)
    
    classes_desc, _ = greedy_init(DIVISORS_DESC)
    score_desc = calc_score(classes_desc)
    
    best_classes = classes_asc if score_asc < score_desc else classes_desc
    best_score = min(score_asc, score_desc)
    
    # Phase 2: SA
    # Run SA on best greedy
    elapsed = time.time() - start_time
    time_left = deadline - elapsed
    if time_left > 5:
        sa_classes, sa_score = simulated_annealing(best_classes, time_left - 2)
        if sa_score < best_score:
            best_classes = sa_classes
            best_score = sa_score
            
    # Phase 3: More restarts if time
    while time.time() - start_time < deadline - 5:
        # Random init
        rand_mods = random.sample(DIVISORS, random.randint(10, 20))
        rand_classes = [[m, random.randint(0, m-1)] for m in rand_mods]
        
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 2: break
        
        sa_classes, sa_score = simulated_annealing(rand_classes, time_left - 2)
        if sa_score < best_score:
            best_classes = sa_classes
            best_score = sa_score
            
    return best_classes
