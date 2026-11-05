import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(n + 1) if sieve[i]]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes)
    
    # Seed solution: r_i = ((i+2)//2) % p_i
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_res = residues[:]
    
    def evaluate(res):
        """Return the largest L such that every integer 1..L has count >= 2."""
        # Start with a reasonable check range
        max_check = 500
        
        while True:
            if time.time() > deadline:
                # Return current best estimate
                break
            
            # Build count array
            counts = np.zeros(max_check + 1, dtype=np.int32)
            for i in range(n_primes):
                p = primes[i]
                r = res[i]
                if r == 0:
                    xs = np.arange(p, max_check + 1, p)
                else:
                    xs = np.arange(r, max_check + 1, p)
                    xs = xs[xs >= 1]
                if len(xs) > 0:
                    counts[xs] += 1
            
            # Find first x >= 1 where counts[x] < 2
            bad = np.where(counts[1:] < 2)[0]
            if len(bad) == 0:
                # All good up to max_check
                if max_check >= 100000:
                    L = max_check
                    break
                else:
                    max_check = min(max_check * 2, 100000)
                    continue
            else:
                # First bad position is at index bad[0] in counts[1:]
                # So x = bad[0] + 1 fails, meaning L = bad[0]
                L = bad[0]
                break
        
        return L
    
    best_L = evaluate(best_res)
    
    # Local search with hill climbing
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        L = evaluate(best_res)
        
        # Compute counts to find bad positions
        check_range = max(L + 200, 1000)
        counts = np.zeros(check_range + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = best_res[i]
            if r == 0:
                xs = np.arange(p, check_range + 1, p)
            else:
                xs = np.arange(r, check_range + 1, p)
                xs = xs[xs >= 1]
            if len(xs) > 0:
                counts[xs] += 1
        
        bad_positions = np.where(counts[1:] < 2)[0]
        if len(bad_positions) == 0:
            # No bad positions in range, extend
            continue
        
        # Target the first few bad positions
        targets = bad_positions[:3]
        
        # Try changing residues for each prime
        # Prioritize primes that could cover bad positions
        for i in range(n_primes):
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = best_res[i]
            
            # Candidate new residues: values that would cover bad positions
            candidates = set()
            for bp in targets:
                x_val = bp + 1
                candidates.add(x_val % p)
            
            # Add a couple random candidates
            for _ in range(2):
                candidates.add(random.randint(0, p - 1))
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                if time.time() > deadline:
                    break
                
                best_res[i] = new_r
                new_L = evaluate(best_res)
                
                if new_L > L:
                    L = new_L
                    best_L = new_L
                    improved = True
                    break
                else:
                    best_res[i] = old_r
            if improved:
                break
        
        # If no improvement via targeted search, try random single moves
        if not improved and time.time() < deadline:
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = random.randint(0, n_primes - 1)
                old_r = best_res[i]
                new_r = random.randint(0, primes[i] - 1)
                if new_r == old_r:
                    continue
                best_res[i] = new_r
                new_L = evaluate(best_res)
                if new_L > L:
                    L = new_L
                    best_L = new_L
                    improved = True
                    break
                else:
                    best_res[i] = old_r
    
    # Final evaluation
    final_L = evaluate(best_res)
    if final_L > best_L:
        best_L = final_L
    
    return best_res
