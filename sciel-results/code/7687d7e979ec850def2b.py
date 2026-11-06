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
    MAX_X = 100000  # cap for evaluation, but we target ~1000-5000
    
    # We will work up to a reasonable limit, say 5000, to ensure we can find L >= 1000
    # If we can't reach 1000, we try to maximize whatever we get.
    WORK_MAX = 5000 

    def forward_greedy(init_residues=None):
        if init_residues is None:
            residues = [random.randint(0, p - 1) for p in primes]
        else:
            residues = init_residues[:]
            
        cov = [0] * (WORK_MAX + 1)
        
        # Initial coverage calculation
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
        
        L = 0
        for x in range(1, WORK_MAX + 1):
            if cov[x] >= 2:
                L = x
            else:
                break
        
        # If we already have L, we can try to extend
        # But usually L starts small.
        
        for x in range(1, WORK_MAX + 1):
            if cov[x] >= 2:
                continue
            
            # Find candidates
            best_p = -1
            best_cost = float('inf')
            best_new_r = -1
            
            for i, p in enumerate(primes):
                cur_r = residues[i]
                new_r = x % p
                if new_r == cur_r:
                    continue
                
                # Calculate cost: number of y < x with cov[y] == 2 and y % p == cur_r
                cost = 0
                if cur_r == 0:
                    start_y = p
                else:
                    start_y = cur_r
                
                # Optimize: iterate y from start_y to x-1 step p
                y = start_y
                while y < x:
                    if cov[y] == 2:
                        cost += 1
                        # Early break if cost exceeds best? 
                        # We want min cost. If cost > best_cost, we can break.
                        if cost > best_cost:
                            break
                    y += p
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = i
                    best_new_r = new_r
                    if cost == 0:
                        # Can't get better than 0. 
                        # But we might want to pick a specific p among those with 0.
                        # Let's collect all with cost 0? Or just pick the first?
                        # To diversify, we could pick randomly from cost 0 candidates.
                        # For now, let's keep the first one found, or prefer larger p.
                        pass
            
            if best_p == -1:
                # No prime can cover x? 
                # This implies all primes already cover x? 
                # If all primes cover x, cov[x] >= 168. But cov[x] < 2. Contradiction.
                # So there must be a prime.
                # But maybe all candidates had cost > 0?
                # If best_cost > 0, we cannot extend without breaking prefix.
                break
            
            # Apply change
            p = primes[best_p]
            old_r = residues[best_p]
            new_r = best_new_r
            
            # Update cov
            # Remove old
            if old_r == 0:
                start = p
            else:
                start = old_r
            y = start
            while y <= WORK_MAX:
                cov[y] -= 1
                y += p
                
            # Add new
            if new_r == 0:
                start = p
            else:
                start = new_r
            y = start
            while y <= WORK_MAX:
                cov[y] += 1
                y += p
                
            residues[best_p] = new_r
            
            # Update L
            # Since we only changed cov for x (increased) and others (decreased/increased),
            # and we ensured cost 0 (no y < x with cov 2 dropped to 1),
            # all y < x still have cov >= 2.
            # x now has cov >= 2 (was < 2, added 1, so >= 1? 
            # Wait. If cov[x] was 1, it becomes 2. If 0, becomes 1.
            # We need cov[x] >= 2.
            # If cov[x] was 0, we added 1, it is 1. Still < 2.
            # We need to add *another* prime?
            # The loop `for x` will handle x again? No, we move to x+1.
            # So if cov[x] is still < 2, we failed.
            
            if cov[x] < 2:
                # We need to fix x again. 
                # This means we need to change another prime for the same x.
                # We can just loop while cov[x] < 2.
                pass

        # Refine the loop to handle multiple changes for same x
        # Let's restructure the main loop
        
        return residues, L

    # The logic above is slightly flawed because if cov[x] < 2, we might need multiple primes.
    # Let's rewrite the core loop properly.
    
    def solve_greedy():
        residues = [random.randint(0, p - 1) for p in primes]
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
        
        L = 0
        for x in range(1, WORK_MAX + 1):
            if cov[x] < 2:
                break
            L = x
            
        # Extend L
        x = L + 1
        while x <= WORK_MAX:
            if cov[x] >= 2:
                L = x
                x += 1
                continue
            
            # Need to increase cov[x]
            # Find primes to change
            # We might need to change 1 or 2 primes.
            
            # Try to find a prime p such that changing it increases cov[x] and Cost 0.
            # If found, apply.
            # If cov[x] is still < 2, repeat.
            
            progress = True
            while cov[x] < 2 and progress:
                progress = False
                best_p = -1
                best_cost = float('inf')
                best_new_r = -1
                candidates = []
                
                for i, p in enumerate(primes):
                    cur_r = residues[i]
                    new_r = x % p
                    if new_r == cur_r:
                        continue
                    
                    cost = 0
                    if cur_r == 0:
                        start_y = p
                    else:
                        start_y = cur_r
                    
                    y = start_y
                    while y < x:
                        if cov[y] == 2:
                            cost += 1
                            if cost > best_cost:
                                break
                        y += p
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_p = i
                        best_new_r = new_r
                        candidates = [(i, new_r)]
                    elif cost == best_cost:
                        candidates.append((i, new_r))
                        
                if best_p == -1:
                    break # No move possible
                
                if best_cost > 0:
                    break # Would break prefix
                
                # Pick a candidate. Prefer larger p?
                # Let's pick the one with largest p to reduce future collisions.
                best_candidate = max(candidates, key=lambda t: primes[t[0]])
                i, new_r = best_candidate
                p = primes[i]
                old_r = residues[i]
                
                # Apply
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
                progress = True
            
            if cov[x] < 2:
                break
            
            L = x
            x += 1
            
        return residues, L

    best_L = 0
    best_res = None
    start_time = time.time()
    time_budget = 110
    
    # Run multiple restarts
    while time.time() - start_time < time_budget:
        res, L = solve_greedy()
        if L > best_L:
            best_L = L
            best_res = res
            
    if best_res is None:
        # Fallback
        return [0] * 168
        
    return best_res
