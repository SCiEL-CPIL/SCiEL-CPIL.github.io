import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i in range(2, limit+1) if sieve[i]]
    
    primes = get_primes(1000)
    n = len(primes) # 168
    
    # Initial solution: simple heuristic
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # We will maintain a count array up to a max limit
    # L is capped at 10^5, but we expect to find L ~ 1000-2000
    # Let's use a buffer of 5000 to be safe for local search exploration
    MAX_L = 5000
    counts = [0] * (MAX_L + 1)
    
    def calculate_L():
        """Calculate the current L based on the counts array."""
        L = 0
        for x in range(1, MAX_L + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L

    # Initialize counts
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        for x in range(start, MAX_L + 1, p):
            counts[x] += 1
            
    current_L = calculate_L()
    best_L = current_L
    best_res = residues[:]
    
    random.seed(42)
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        improved = False
        
        # 1. Identify the critical point
        # Find the first x <= current_L + 200 where count < 2
        search_limit = min(current_L + 200, MAX_L)
        x_crit = -1
        for x in range(1, search_limit + 1):
            if counts[x] < 2:
                x_crit = x
                break
        
        # 2. Generate moves
        # Strategy A: Targeted moves to fix x_crit
        if x_crit != -1:
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                p = primes[i]
                target_r = x_crit % p
                old_r = residues[i]
                
                if target_r == old_r:
                    continue
                    
                # Apply move
                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                for x in range(start_old, MAX_L + 1, p):
                    counts[x] -= 1
                    
                if target_r == 0:
                    start_new = p
                else:
                    start_new = target_r
                for x in range(start_new, MAX_L + 1, p):
                    counts[x] += 1
                    
                residues[i] = target_r
                
                new_L = calculate_L()
                
                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_res = residues[:]
                        no_improve_count = 0
                    improved = True
                    break
                else:
                    # Revert
                    if target_r == 0:
                        start_new = p
                    else:
                        start_new = target_r
                    for x in range(start_new, MAX_L + 1, p):
                        counts[x] -= 1
                        
                    if old_r == 0:
                        start_old = p
                    else:
                        start_old = old_r
                    for x in range(start_old, MAX_L + 1, p):
                        counts[x] += 1
                        
                    residues[i] = old_r
            
            if improved:
                continue # Restart loop with new L
        
        # 3. If no targeted move worked, try random moves
        if not improved:
            for _ in range(20):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 1)
                p = primes[i]
                new_r = random.randint(0, p - 1)
                old_r = residues[i]
                
                if new_r == old_r:
                    continue
                
                # Apply move
                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                for x in range(start_old, MAX_L + 1, p):
                    counts[x] -= 1
                    
                if new_r == 0:
                    start_new = p
                else:
                    start_new = new_r
                for x in range(start_new, MAX_L + 1, p):
                    counts[x] += 1
                    
                residues[i] = new_r
                
                new_L = calculate_L()
                
                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_res = residues[:]
                        no_improve_count = 0
                    improved = True
                    break
                else:
                    # Revert
                    if new_r == 0:
                        start_new = p
                    else:
                        start_new = new_r
                    for x in range(start_new, MAX_L + 1, p):
                        counts[x] -= 1
                        
                    if old_r == 0:
                        start_old = p
                    else:
                        start_old = old_r
                    for x in range(start_old, MAX_L + 1, p):
                        counts[x] += 1
                        
                    residues[i] = old_r
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Random restart / perturbation
                    for _ in range(n // 10):
                        i = random.randint(0, n - 1)
                        p = primes[i]
                        new_r = random.randint(0, p - 1)
                        old_r = residues[i]
                        
                        if old_r == 0:
                            start_old = p
                        else:
                            start_old = old_r
                        for x in range(start_old, MAX_L + 1, p):
                            counts[x] -= 1
                            
                        if new_r == 0:
                            start_new = p
                        else:
                            start_new = new_r
                        for x in range(start_new, MAX_L + 1, p):
                            counts[x] += 1
                            
                        residues[i] = new_r
                        
                    current_L = calculate_L()
                    no_improve_count = 0
    
    # Final verification with full range up to 100000? 
    if time.time() < deadline - 5:
        def final_compute_L(res):
            max_check = 100000
            counts_final = [0] * (max_check + 1)
            for i, p in enumerate(primes):
                r = res[i]
                if r == 0:
                    start_x = p
                else:
                    start_x = r
                for x in range(start_x, max_check + 1, p):
                    counts_final[x] += 1
            
            L = 0
            for x in range(1, max_check + 1):
                if counts_final[x] >= 2:
                    L = x
                else:
                    break
            return L

        final_L = final_compute_L(best_res)
        if final_L > best_L:
            best_L = final_L

    return [int(r) for r in best_res]
