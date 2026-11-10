import random
import math
import time

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We will search for a solution that covers up to M
    M = 10000
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in primes]
    
    def compute_counts(res):
        counts = [0] * (M + 1)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            # Determine start index
            if r == 0:
                start = p
            else:
                start = r
            # Add counts
            for x in range(start, M + 1, p):
                counts[x] += 1
        return counts

    def get_score(counts):
        for x in range(1, M + 1):
            if counts[x] < 2:
                return x - 1
        return M

    # Initial counts and score
    counts = compute_counts(residues)
    current_score = get_score(counts)
    best_score = current_score
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 110.0 # Leave some buffer for 120s limit
    
    # Phase 1: Greedy Repair
    # Try to fix the first failure repeatedly
    while current_score < M and (time.time() - start_time) < time_limit * 0.5:
        x_fail = current_score + 1
        if x_fail > M:
            break
        
        # Find primes that do NOT cover x_fail
        candidates = [i for i in range(n_primes) if x_fail % primes[i] != residues[i]]
        
        if not candidates:
            break
            
        # Pick a random candidate
        i = random.choice(candidates)
        p = primes[i]
        old_r = residues[i]
        new_r = x_fail % p
        
        if new_r == old_r:
            continue
        
        # Apply move
        residues[i] = new_r
        
        # Update counts
        start_old = old_r if old_r > 0 else p
        start_new = new_r if new_r > 0 else new_r # Wait, if new_r is 0, start is p
        
        # Correct start calculation
        start_old = old_r if old_r > 0 else p
        start_new = new_r if new_r > 0 else p
        
        for x in range(start_old, M + 1, p):
            counts[x] -= 1
        for x in range(start_new, M + 1, p):
            counts[x] += 1
            
        new_score = get_score(counts)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues[:]
        else:
            # Revert
            residues[i] = old_r
            for x in range(start_new, M + 1, p):
                counts[x] -= 1
            for x in range(start_old, M + 1, p):
                counts[x] += 1
            
            # If greedy move failed, try a random kick
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            residues[i] = new_r
            start_old = old_r if old_r > 0 else p
            start_new = new_r if new_r > 0 else p
            for x in range(start_old, M + 1, p):
                counts[x] -= 1
            for x in range(start_new, M + 1, p):
                counts[x] += 1
            new_score = get_score(counts)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues[:]
            else:
                # Revert
                residues[i] = old_r
                for x in range(start_new, M + 1, p):
                    counts[x] -= 1
                for x in range(start_old, M + 1, p):
                    counts[x] += 1
                # If we fail to improve even with a kick, we might be stuck.
                # But we continue the loop to try other candidates?
                # The loop structure above only tries ONE candidate per iteration.
                # This is slow. Let's try all candidates in a loop.
                break # Break to restart the while loop with a new random state?
                # Actually, if we break, we lose the chance to try other candidates for this x_fail.
                # Let's modify the inner logic to try all candidates if time permits.

    # Phase 2: Local Search (Hill Climbing)
    # Try to extend the score
    while (time.time() - start_time) < time_limit:
        x_fail = current_score + 1
        if x_fail > M:
            break
        
        improved = False
        candidates = [i for i in range(n_primes) if x_fail % primes[i] != residues[i]]
        random.shuffle(candidates)
        
        for i in candidates:
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            if new_r == old_r:
                continue
                
            residues[i] = new_r
            start_old = old_r if old_r > 0 else p
            start_new = new_r if new_r > 0 else p
            
            for x in range(start_old, M + 1, p):
                counts[x] -= 1
            for x in range(start_new, M + 1, p):
                counts[x] += 1
                
            new_score = get_score(counts)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues[:]
                improved = True
                break
            else:
                # Revert
                residues[i] = old_r
                for x in range(start_new, M + 1, p):
                    counts[x] -= 1
                for x in range(start_old, M + 1, p):
                    counts[x] += 1
        
        if not improved:
            # Random kick to escape local optimum
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            residues[i] = new_r
            start_old = old_r if old_r > 0 else p
            start_new = new_r if new_r > 0 else p
            for x in range(start_old, M + 1, p):
                counts[x] -= 1
            for x in range(start_new, M + 1, p):
                counts[x] += 1
            new_score = get_score(counts)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues[:]
            else:
                # Revert
                residues[i] = old_r
                for x in range(start_new, M + 1, p):
                    counts[x] -= 1
                for x in range(start_old, M + 1, p):
                    counts[x] += 1
                # If we are stuck and time is running out, break
                if time.time() - start_time > time_limit * 0.95:
                    break

    return best_residues
