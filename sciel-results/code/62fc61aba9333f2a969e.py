import time
import random
import numpy as np

_BASE = 51975
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _get_divisors():
    ds = []
    for d in range(3, _BASE + 1, 2):
        if _BASE % d == 0:
            ds.append(d)
    return ds

def construct():
    divisors = _get_divisors()
    
    # State variables
    cover_count = np.zeros(_BASE, dtype=np.int8)
    uncovered = _BASE
    classes = []
    used_moduli = set()
    unused_moduli = set(divisors)
    
    def _add(m, r):
        nonlocal uncovered
        for x in range(r, _BASE, m):
            if cover_count[x] == 0:
                uncovered -= 1
            cover_count[x] += 1
        classes.append([m, r])
        used_moduli.add(m)
        unused_moduli.discard(m)

    def _remove(m, r):
        nonlocal uncovered
        for x in range(r, _BASE, m):
            cover_count[x] -= 1
            if cover_count[x] == 0:
                uncovered += 1
        for i, c in enumerate(classes):
            if c[0] == m:
                classes.pop(i)
                break
        used_moduli.discard(m)
        unused_moduli.add(m)

    def _change(m, r_old, r_new):
        nonlocal uncovered
        for x in range(r_old, _BASE, m):
            cover_count[x] -= 1
            if cover_count[x] == 0:
                uncovered += 1
        for x in range(r_new, _BASE, m):
            if cover_count[x] == 0:
                uncovered -= 1
            cover_count[x] += 1
        for c in classes:
            if c[0] == m:
                c[1] = r_new
                break

    best_uncovered = _BASE
    best_classes = []
    
    T = 100.0
    T_min = 0.1
    alpha = 0.999
    
    start_time = time.time()
    
    # Greedy Initialization
    # Add moduli in random order, picking best residue
    # This gives a good starting point for SA
    
    # Shuffle divisors
    init_divs = divisors[:]
    random.shuffle(init_divs)
    
    for m in init_divs:
        best_r = 0
        best_gain = -1
        # Sample a few residues to speed up, or just all if m is small
        # m is at most 51975/3 = 17325. 
        # Iterating all r is O(m). 
        # Total cost O(sum m). Sum of divisors of 51975 is manageable.
        for r in range(m):
            # Estimate gain: count how many 0s in cover_count[r::m]
            # This is O(N/m)
            # Doing this for all r is O(N). 
            # Doing this for all m is O(N * num_divs) ~ 50k * 47 ~ 2.5M. Fast.
            gain = 0
            for x in range(r, _BASE, m):
                if cover_count[x] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            _add(m, best_r)
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_classes = [c[:] for c in classes]
        
        if time.time() - start_time > _DEADLINE - 20:
            break
            
    # Simulated Annealing
    while time.time() - start_time < _DEADLINE - 5:
        if T < T_min:
            # Restart temperature
            T = 100.0
            # Optionally reset to best solution
            # For now, just continue from current, but higher T allows escaping.
            pass
            
        prev_uncovered = uncovered
        
        # Choose move
        n_classes = len(classes)
        
        # Probabilities
        if n_classes < 10:
            p_add = 0.5
            p_change = 0.4
            p_remove = 0.1
        elif n_classes < 30:
            p_add = 0.3
            p_change = 0.6
            p_remove = 0.1
        else:
            p_add = 0.2
            p_change = 0.7
            p_remove = 0.1
            
        r_rand = random.random()
        
        move_made = False
        
        if r_rand < p_add:
            if unused_moduli:
                m = random.choice(list(unused_moduli))
                r = random.randint(0, m - 1)
                _add(m, r)
                move_made = True
                new_uncovered = uncovered
        elif r_rand < p_add + p_change:
            if classes:
                m, r_old = random.choice(classes)
                r_new = random.randint(0, m - 1)
                if r_new != r_old:
                    _change(m, r_old, r_new)
                    move_made = True
                    new_uncovered = uncovered
        else:
            if classes:
                m, r_old = random.choice(classes)
                _remove(m, r_old)
                move_made = True
                new_uncovered = uncovered
        
        if move_made:
            if new_uncovered < best_uncovered:
                best_uncovered = new_uncovered
                best_classes = [c[:] for c in classes]
                
            delta = new_uncovered - prev_uncovered
            if delta > 0:
                # Reject with probability 1 - exp(-delta/T)
                if random.random() > np.exp(-delta / T):
                    # Revert
                    if r_rand < p_add:
                        _remove(m, r)
                    elif r_rand < p_add + p_change:
                        _change(m, r_new, r_old)
                    else:
                        _add(m, r_old)
                        
        T *= alpha
        
    return best_classes
# EVOLVE-BLOCK-END
