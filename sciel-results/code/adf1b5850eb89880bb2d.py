import random
import time

def construct():
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)  # 168
    MAX_X = 20000  # Increased to allow finding longer prefixes
    
    def get_coverage(residues):
        cov = [0] * (MAX_X + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            x = start
            while x <= MAX_X:
                cov[x] += 1
                x += p
        return cov

    def find_L(cov):
        L = 0
        for x in range(1, MAX_X + 1):
            if cov[x] < 2:
                break
            L = x
        return L

    def solve_repair(init_residues):
        residues = init_residues[:]
        cov = get_coverage(residues)
        L = find_L(cov)
        
        max_iter = 5000
        iter_count = 0
        best_L_local = L
        best_res_local = residues[:]
        
        while iter_count < max_iter:
            iter_count += 1
            
            # Find first hole
            target_x = L + 1
            if target_x > MAX_X:
                break
                
            # Find safe candidates
            candidates = []
            for i, p in enumerate(primes):
                old_r = residues[i]
                new_r = target_x % p
                if old_r == new_r:
                    # Already covers target_x?
                    # If old_r == new_r, then target_x is already covered by p.
                    # But cov[target_x] < 2, so it's not covered by 2 primes.
                    # If p covers it, we need another prime to cover it.
                    # Changing p to the same residue does nothing.
                    continue
                    
                # Check if changing p to new_r breaks any y < L
                safe = True
                # We need to check all y < L that are covered by (p, old_r)
                # If any such y has cov[y] == 2, it will become 1, breaking prefix.
                if old_r == 0:
                    start_y = p
                else:
                    start_y = old_r
                    
                y = start_y
                while y < L:
                    if cov[y] == 2:
                        safe = False
                        break
                    y += p
                    
                if safe:
                    candidates.append(i)
            
            if candidates:
                # Pick a random candidate to diversify
                idx = random.choice(candidates)
                p = primes[idx]
                old_r = residues[idx]
                new_r = target_x % p
                
                # Update cov
                if old_r == 0:
                    start = p
                else:
                    start = old_r
                y = start
                while y <= MAX_X:
                    cov[y] -= 1
                    y += p
                    
                if new_r == 0:
                    start = p
                else:
                    start = new_r
                y = start
                while y <= MAX_X:
                    cov[y] += 1
                    y += p
                    
                residues[idx] = new_r
                
                # Update L
                new_L = find_L(cov)
                if new_L > best_L_local:
                    best_L_local = new_L
                    best_res_local = residues[:]
                L = new_L
                
            else:
                # Stuck. Perform a kick.
                # Pick a random prime and change its residue to a random value.
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                
                # Update cov
                if old_r == 0:
                    start = p
                else:
                    start = old_r
                y = start
                while y <= MAX_X:
                    cov[y] -= 1
                    y += p
                    
                if new_r == 0:
                    start = p
                else:
                    start = new_r
                y = start
                while y <= MAX_X:
                    cov[y] += 1
                    y += p
                    
                residues[idx] = new_r
                
                # Update L
                L = find_L(cov)
                if L > best_L_local:
                    best_L_local = L
                    best_res_local = residues[:]
                    
        return best_res_local, best_L_local

    best_L = 0
    best_res = None
    start_time = time.time()
    time_budget = 115
    
    # Initial guesses
    inits = [
        [0] * n_primes,
        [1] * n_primes,
        [random.randint(0, p - 1) for p in primes]
    ]
    
    for init in inits:
        if time.time() - start_time > time_budget:
            break
        res, L = solve_repair(init)
        if L > best_L:
            best_L = L
            best_res = res[:]
            
    # Random restarts
    while time.time() - start_time < time_budget:
        init = [random.randint(0, p - 1) for p in primes]
        res, L = solve_repair(init)
        if L > best_L:
            best_L = L
            best_res = res[:]
            
    if best_res is None:
        return [0] * 168
        
    return best_res
