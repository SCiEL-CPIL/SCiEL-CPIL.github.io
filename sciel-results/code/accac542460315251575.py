import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Lower bound is 31.
    # We try to find a valid tournament for n starting from 31.
    
    def verify(n, orientation):
        """
        orientation is a list of bits for pairs (i, j), i < j.
        Returns True if valid.
        """
        # Build in_bits for speed
        in_bits = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    # i -> j, so i is in-neighbor of j
                    in_bits[j] |= (1 << i)
                else:
                    # j -> i, so j is in-neighbor of i
                    in_bits[i] |= (1 << j)
                idx += 1
        
        # Check all 4-combinations
        for a, b, c, d in itertools.combinations(range(n), 4):
            if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]) == 0:
                return False
        return True

    def generate_random_tournament(n):
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                orientation.append(random.randint(0, 1))
        return orientation

    # We will try n from 31 to 90
    # For each n, we try a number of random tournaments
    # The number of trials should decrease as n increases, or we can just use a time budget.
    
    best_n = None
    best_orientation = None
    
    # Fallback: Paley(67) is known to be valid.
    # We can construct it if we run out of time.
    
    def get_paley_67():
        p = 67
        squares = set(x * x % p for x in range(1, p))
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return p, orientation

    # Try random search
    for n in range(31, 91):
        if time.time() > deadline - 10: # Leave 10s for fallback
            break
            
        # Number of trials: more for smaller n, less for larger?
        # Or just a fixed budget of time per n?
        # Let's try 1000 trials for n=31, 500 for n=32, etc.
        # Or just loop until time runs out for this n.
        
        trials = 2000
        if n > 40:
            trials = 500
        if n > 50:
            trials = 100
            
        for _ in range(trials):
            if time.time() > deadline - 5:
                break
                
            orientation = generate_random_tournament(n)
            if verify(n, orientation):
                best_n = n
                best_orientation = orientation
                break # Found valid for this n, go to next n? No, we want smallest n.
                # So if we find valid for n, we return immediately?
                # Yes, because we are iterating n from 31 upwards.
                # So if we find one for n=31, we return 31.
                # If we don't find one for 31, we try 32.
                
    if best_n is not None:
        return {'n': best_n, 'orientation': best_orientation}
        
    # Fallback to Paley(67)
    p, orientation = get_paley_67()
    return {'n': p, 'orientation': orientation}
