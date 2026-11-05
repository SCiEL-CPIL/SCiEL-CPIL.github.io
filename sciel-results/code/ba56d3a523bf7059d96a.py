import time
import random
import math

def _get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Window for local search optimization
    # We want to push the score as high as possible.
    # Let's start with a window of 2000.
    W = 2000
    
    # Initialize residues
    # A simple random start
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Initialize counts
    counts = [0] * (W + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        # x in 1..W such that x % p == r
        # First x >= 1:
        # x = r if r >= 1 else r + p
        # Actually, r is 0..p-1.
        # If r == 0, x = p, 2p...
        # If r > 0, x = r, r+p...
        start = r
        if start == 0:
            start = p
        elif start < 1:
            start = r + p
        # Wait, x must be >= 1.
        # If r == 0, x can be p, 2p...
        # If r > 0, x can be r, r+p...
        # But if r < 1, i.e., r=0, handled.
        # What if r > W? Then no x.
        
        # Let's just iterate
        # x = r if r >= 1 else r + p
        # But we need x >= 1.
        # If r == 0, x starts at p.
        # If r > 0, x starts at r.
        
        x_start = r
        if x_start == 0:
            x_start = p
        if x_start > W:
            continue
        for x in range(x_start, W + 1, p):
            counts[x] += 1
            
    # Calculate initial BadCount (number of x in 1..W with counts[x] < 2)
    bad_count = sum(1 for x in range(1, W + 1) if counts[x] < 2)
    
    # Best score tracking
    # Score is largest L such that all 1..L have count >= 2.
    def get_score(counts_arr):
        for x in range(1, len(counts_arr)):
            if counts_arr[x] < 2:
                return x - 1
        return len(counts_arr) - 1
        
    current_score = get_score(counts)
    best_score = current_score
    best_residues = residues[:]
    
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Simulated Annealing parameters
    T = 10.0 # Temperature
    cooling = 0.999
    
    step = 0
    while time.time() - start_time < deadline:
        # Pick a random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        # Calculate delta bad_count
        # Decrement old_r
        delta_bad = 0
        
        # Helper to update counts and delta_bad for a residue change
        # This is done in-place on a temporary copy? No, we need to be fast.
        # We can compute the delta without applying, then apply if accepted.
        
        # Let's compute the change in bad_count for the window [1, W]
        
        # 1. Effect of removing old_r
        # x in 1..W, x % p == old_r
        x_start = old_r
        if x_start == 0:
            x_start = p
        if x_start <= W:
            for x in range(x_start, W + 1, p):
                c = counts[x]
                if c < 2:
                    # c becomes c-1, still < 2 (or less)
                    # If c was 2, it becomes 1 (bad increases)
                    # If c was 1, it becomes 0 (bad unchanged, was already bad)
                    # If c was 0, it becomes -1 (impossible, counts >= 0)
                    pass
                elif c == 2:
                    # becomes 1, so bad increases
                    delta_bad += 1
                else:
                    # c > 2, becomes c-1 >= 2, no change in bad
                    pass
        
        # 2. Effect of adding new_r
        x_start = new_r
        if x_start == 0:
            x_start = p
        if x_start <= W:
            for x in range(x_start, W + 1, p):
                c = counts[x]
                if c < 2:
                    # c becomes c+1
                    # If c was 0, becomes 1 (still bad)
                    # If c was 1, becomes 2 (bad decreases)
                    if c == 1:
                        delta_bad -= 1
                else:
                    # c >= 2, becomes c+1 >= 3, no change in bad
                    pass
        
        # Acceptance probability
        if delta_bad < 0:
            accept = True
        elif delta_bad == 0:
            # Neutral move, accept sometimes to explore
            accept = random.random() < 0.5
        else:
            # Worse move
            # Probability = exp(-delta_bad / T)
            # If T is small, this is unlikely.
            accept = random.random() < math.exp(-delta_bad / T)
            
        if accept:
            # Apply the change
            residues[i] = new_r
            
            # Update counts
            x_start = old_r
            if x_start == 0:
                x_start = p
            if x_start <= W:
                for x in range(x_start, W + 1, p):
                    counts[x] -= 1
            
            x_start = new_r
            if x_start == 0:
                x_start = p
            if x_start <= W:
                for x in range(x_start, W + 1, p):
                    counts[x] += 1
                    
            bad_count += delta_bad
            
            # Cool down
            T *= cooling
            
            # Check score periodically
            if step % 1000 == 0:
                s = get_score(counts)
                if s > best_score:
                    best_score = s
                    best_residues = residues[:]
                    
            step += 1
            
    # Final check
    s = get_score(counts)
    if s > best_score:
        best_residues = residues[:]
        
    return best_residues
