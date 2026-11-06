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
    MAX_X = 100000  # cap for evaluation
    
    # We'll work with a smaller working range for speed, say up to 5000
    WORK_MAX = 5000
    
    def compute_coverage(residues):
        cov = [0] * (WORK_MAX + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            # x ≡ r (mod p) for x in 1..WORK_MAX
            # First x >= 1 with x ≡ r mod p:
            # x = r if r >= 1 else r + p (but r can be 0, then x = p, 2p, ...)
            # Actually x in 1..WORK_MAX, x ≡ r mod p
            # The smallest x >= 1: if r == 0, x = p; else x = r (if r >= 1)
            if r == 0:
                start = p
            else:
                start = r
            # But start might be > WORK_MAX, then no coverage
            x = start
            while x <= WORK_MAX:
                cov[x] += 1
                x += p
        return cov
    
    def objective(residues):
        cov = compute_coverage(residues)
        # Find largest L such that cov[1..L] all >= 2
        L = 0
        for x in range(1, WORK_MAX + 1):
            if cov[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Greedy initialization: try to cover early integers
    def greedy_init():
        residues = [0] * n_primes
        cov = [0] * (WORK_MAX + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            x = start
            while x <= WORK_MAX:
                cov[x] += 1
                x += p
        
        # For each x from 1 upward, if cov[x] < 2, fix it
        for x in range(1, WORK_MAX + 1):
            while cov[x] < 2:
                # Find a prime p where x mod p != residues[i], and changing it helps
                best_i = -1
                best_gain = -1
                for i, p in enumerate(primes):
                    cur_r = residues[i]
                    new_r = x % p
                    if new_r == cur_r:
                        continue
                    # Changing r_i from cur_r to new_r:
                    # Adds 1 to cov[x] (since x ≡ new_r mod p)
                    # Removes 1 from all y where y ≡ cur_r mod p, y != x
                    # Net effect on coverage: we want to minimize damage
                    # Simple heuristic: pick prime with largest p (least frequent hits)
                    # or pick randomly
                    if p > best_gain:
                        best_gain = p
                        best_i = i
                if best_i == -1:
                    # Can't improve, break
                    break
                p = primes[best_i]
                old_r = residues[best_i]
                new_r = x % p
                # Update coverage
                # Remove old contributions
                if old_r == 0:
                    start = p
                else:
                    start = old_r
                y = start
                while y <= WORK_MAX:
                    cov[y] -= 1
                    y += p
                # Add new contributions
                if new_r == 0:
                    start = p
                else:
                    start = new_r
                y = start
                while y <= WORK_MAX:
                    cov[y] += 1
                    y += p
                residues[best_i] = new_r
        return residues
    
    # Local search: hill climbing with random restarts
    def local_search(residues, time_limit):
        start_time = time.time()
        cov = compute_coverage(residues)
        best_L = objective(residues)
        best_residues = residues[:]
        
        while time.time() - start_time < time_limit:
            # Find smallest x with cov[x] < 2
            bottleneck_x = -1
            for x in range(1, WORK_MAX + 1):
                if cov[x] < 2:
                    bottleneck_x = x
                    break
            if bottleneck_x == -1:
                # All covered up to WORK_MAX, try to extend? We're capped.
                break
            
            # Try all possible single-residue changes that increase cov[bottleneck_x]
            best_move = None
            best_score = -1
            for i, p in enumerate(primes):
                cur_r = residues[i]
                new_r = bottleneck_x % p
                if new_r == cur_r:
                    continue
                # This change adds 1 to cov[bottleneck_x]
                # Compute how many other positions lose coverage
                # Positions that had cur_r mod p and become uncovered
                # and positions that gain new_r mod p
                # Net: we care about not creating new bottlenecks
                # Simple: count how many y in 1..WORK_MAX have cov[y] == 2 and y ≡ cur_r mod p (would drop to 1)
                # vs y ≡ new_r mod p (would go from cov to cov+1, fine)
                # Penalty: number of y with cov[y] == 2 and y ≡ cur_r mod p, y != bottleneck_x
                penalty = 0
                if cur_r == 0:
                    start = p
                else:
                    start = cur_r
                y = start
                while y <= WORK_MAX:
                    if y != bottleneck_x and cov[y] == 2:
                        penalty += 1
                    y += p
                # Score: prefer lower penalty
                if penalty < best_score or best_move is None:
                    best_score = penalty
                    best_move = (i, cur_r, new_r)
            
            if best_move is None:
                # No improving move, random kick
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                new_r = random.randint(0, p - 1)
                if new_r == residues[i]:
                    continue
                old_r = residues[i]
                # Update coverage
                if old_r == 0:
                    start = p
                else:
                    start = old_r
                y = start
                while y <= WORK_MAX:
                    cov[y] -= 1
                    y += p
                if new_r == 0:
                    start = p
                else:
                    start = new_r
                y = start
                while y <= WORK_MAX:
                    cov[y] += 1
                    y += p
                residues[i] = new_r
                continue
            
            i, old_r, new_r = best_move
            p = primes[i]
            # Apply move
            if old_r == 0:
                start = p
            else:
                start = old_r
            y = start
            while y <= WORK_MAX:
                cov[y] -= 1
                y += p
            if new_r == 0:
                start = p
            else:
                start = new_r
            y = start
            while y <= WORK_MAX:
                cov[y] += 1
                y += p
            residues[i] = new_r
            
            # Update objective
            L = 0
            for x in range(1, WORK_MAX + 1):
                if cov[x] >= 2:
                    L = x
                else:
                    break
            if L > best_L:
                best_L = L
                best_residues = residues[:]
        
        return best_residues, best_L
    
    # Run local search with multiple restarts
    best_overall = greedy_init()
    best_L_overall = objective(best_overall)
    
    start_time = time.time()
    time_budget = 100  # leave 20s for final evaluation
    
    # First, local search from greedy init
    best_res, best_L = local_search(best_overall, time_budget * 0.5)
    if best_L > best_L_overall:
        best_L_overall = best_L
        best_overall = best_res
    
    # Random restarts
    while time.time() - start_time < time_budget:
        # Random initialization
        residues = [random.randint(0, p - 1) for p in primes]
        # Local search from random
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        res, L = local_search(residues, min(remaining, 10))
        if L > best_L_overall:
            best_L_overall = L
            best_overall = res
    
    # Final answer: best_overall
    return best_overall
