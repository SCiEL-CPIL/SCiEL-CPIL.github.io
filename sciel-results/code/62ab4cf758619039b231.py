import numpy as np
import time
import random

def construct():
    # Get primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Max L to check for score
    M = 100000
    
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Initialize residues randomly
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    # Initialize counts
    counts = np.zeros(M + 1, dtype=np.int64)
    for i, p in enumerate(primes):
        r = residues[i]
        counts[r::p] += 1
        
    # Calculate current L
    def calc_L(counts):
        # Find first index >= 1 where count < 2
        # We check 1 to M
        if counts[1] < 2:
            return 0
        # np.argmax returns 0 if no condition met. We need first True.
        # Let's find indices where counts < 2
        # To be efficient, we can check if all are >= 2
        if np.all(counts[1:] >= 2):
            return M
        # Find first index
        # This might be slow if done naively in Python, but np.argmax is fast
        # We want the first index where counts < 2
        # Invert: counts >= 2 is True. We want first False.
        # ~ (counts >= 2) is True where counts < 2.
        mask = ~(counts[1:] >= 2)
        first_false_idx = np.argmax(mask)
        # If no False, argmax returns 0. But we checked all True above.
        # So first_false_idx is 0-based index in counts[1:]
        # The actual index is first_false_idx + 1
        # L is the number of integers before this, so (first_false_idx + 1) - 1 = first_false_idx
        return first_false_idx

    current_L = calc_L(counts)
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Main loop
    while time.time() - start_time < deadline - 5.0:
        # Pick random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Pick new random residue
        r_new = random.randint(0, p - 1)
        if r_new == r_old:
            continue
            
        # Update counts
        counts[r_old::p] -= 1
        counts[r_new::p] += 1
        
        new_L = calc_L(counts)
        
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = current_L - new_L
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_L = new_L
                residues[i] = r_new
            else:
                # Reject, revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                residues[i] = r_old # Just in case, though it wasn't changed if we revert logic is clear
                
        # Cool down
        T = max(T_min, T * cooling_rate)

    return [int(r) for r in best_residues]
