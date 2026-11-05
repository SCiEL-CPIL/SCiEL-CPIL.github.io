"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def is_compatible(m_new, r_new, selected_list):
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def get_valid_residues(m_new, selected_list):
        valids = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected_list):
                valids.append(r)
        return valids

    def greedy_build(order, rng):
        selected = []
        for m in order:
            valids = get_valid_residues(m, selected)
            if valids:
                r = rng.choice(valids)
                selected.append([m, r])
        return selected

    def simple_add(selected, all_moduli, rng):
        selected_moduli = set(m for m, r in selected)
        improved = True
        while improved:
            improved = False
            for m in all_moduli:
                if m in selected_moduli:
                    continue
                valids = get_valid_residues(m, selected)
                if valids:
                    r = rng.choice(valids)
                    selected.append([m, r])
                    selected_moduli.add(m)
                    improved = True
        return selected

    best_score = 0.0
    best_classes = []
    
    all_moduli = list(range(20, 201))
    
    def num_pf(n):
        c = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
    
    pf_counts = {m: num_pf(m) for m in all_moduli}
    order_prime_first = sorted(all_moduli, key=lambda m: pf_counts[m])
    order_asc = list(all_moduli)
    order_desc = list(reversed(all_moduli))
    
    for trial in range(10000):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(trial * 7919 + 13)
        
        t = trial % 6
        if t == 0:
            order = order_asc
        elif t == 1:
            order = order_desc
        elif t == 2:
            order = order_prime_first
        elif t == 3:
            order = all_moduli[:]
            rng.shuffle(order)
        elif t == 4:
            order = order_prime_first[:]
            rng.shuffle(order)
        else:
            order = all_moduli[:]
            rng.shuffle(order)
        
        classes = greedy_build(order, rng)
        classes = simple_add(classes, all_moduli, rng)
        score = sum(1.0 / m for m, r in classes)
        
        if score > best_score:
            best_score = score
            best_classes = classes
    
    return best_classes

# EVOLVE-BLOCK-END
