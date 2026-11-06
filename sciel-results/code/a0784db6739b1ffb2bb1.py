import math
import random
import time

def construct():
    moduli = list(range(20, 201))
    
    def greedy_init():
        selected = []
        used_moduli = set()
        for m in moduli:
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in selected:
                    g = math.gcd(m, m2)
                    if (r - r2) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append((m, r))
                    used_moduli.add(m)
                    found = True
                    break
        return selected, used_moduli

    current_sol, current_used = greedy_init()
    current_score = sum(1.0 / m for m, _ in current_sol)
    best_sol = list(current_sol)
    best_score = current_score
    
    def is_valid(m, r, sol, used):
        if m in used:
            return False
        for m2, r2 in sol:
            g = math.gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def get_score(sol):
        return sum(1.0 / m for m, _ in sol)

    start_time = time.time()
    time_limit = 115.0
    
    T = 0.05 # Start with smaller T to focus on improvements
    cooling = 0.9999
    
    # Precompute gcds for all pairs of moduli?
    # 181 moduli. 181*181/2 = 16k pairs.
    # gcd is fast, but precomputing saves time in loop.
    gcd_cache = {}
    for i, m1 in enumerate(moduli):
        for m2 in moduli[i:]:
            gcd_cache[(m1, m2)] = math.gcd(m1, m2)
            if m1 != m2:
                gcd_cache[(m2, m1)] = math.gcd(m1, m2)
                
    def is_valid_fast(m, r, sol, used):
        if m in used:
            return False
        for m2, r2 in sol:
            g = gcd_cache.get((m, m2), math.gcd(m, m2))
            if (r - r2) % g == 0:
                return False
        return True

    while time.time() - start_time < time_limit:
        move_type = random.randint(0, 2)
        new_sol = list(current_sol)
        new_used = set(current_used)
        
        if move_type == 0:
            # Add
            available = [m for m in moduli if m not in new_used]
            if available:
                m = random.choice(available)
                valid_rs = []
                # Optimization: check a few random r's first?
                # Or just find one valid r quickly.
                # Since we just need one, we can iterate r.
                # But we want random r for diversity?
                # For hill climbing, any valid r is fine.
                found_r = None
                for r in range(m):
                    if is_valid_fast(m, r, current_sol, current_used):
                        found_r = r
                        break
                if found_r is not None:
                    new_sol.append((m, found_r))
                    new_used.add(m)
        elif move_type == 1:
            # Remove
            if current_sol:
                idx = random.randint(0, len(current_sol) - 1)
                m, r = new_sol.pop(idx)
                new_used.remove(m)
        else:
            # Change r
            if current_sol:
                idx = random.randint(0, len(current_sol) - 1)
                m, r_old = new_sol[idx]
                # Try a random new r
                r_new = random.randint(0, m - 1)
                if r_new != r_old:
                    # Check against others
                    temp_sol = [x for i, x in enumerate(current_sol) if i != idx]
                    if is_valid_fast(m, r_new, temp_sol, current_used - {m}):
                        new_sol[idx] = (m, r_new)
                    else:
                        continue

        new_score = get_score(new_sol)
        delta = new_score - current_score
        
        if delta > 0 or (T > 1e-6 and random.random() < math.exp(delta / T)):
            current_sol = new_sol
            current_used = new_used
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_sol = list(current_sol)
        
        T *= cooling

    return [[m, r] for m, r in best_sol]
