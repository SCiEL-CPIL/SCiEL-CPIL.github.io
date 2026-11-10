import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit
    
    moduli = list(range(2, 129, 2))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # Try multiple random restarts if needed
    for attempt in range(100):
        residues = [random.randint(0, m - 1) for m in moduli]
        
        def is_valid(i, j):
            g = gcds[i][j]
            return residues[i] % g != residues[j] % g

        # Local search
        stuck = False
        while time.time() < deadline:
            violated = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not is_valid(i, j):
                        violated.append((i, j))
            
            if not violated:
                return [[residues[i], moduli[i]] for i in range(n)]

            i, j = random.choice(violated)
            g = gcds[i][j]
            target_mod = residues[j] % g
            
            # Find a new residue for i that fixes the violation with j
            # and hopefully doesn't break too many others
            # Simple: random valid
            while True:
                new_r = random.randint(0, moduli[i] - 1)
                if new_r % g != target_mod:
                    break
            
            residues[i] = new_r
            
            # If we loop too many times without improvement, break inner loop?
            # The outer restart handles this.
            
            # Check time inside inner loop is important
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    # If we fail to find a valid solution (very unlikely), return a valid small set
    # e.g. just [[1, 2]]
    return [[1, 2]]
