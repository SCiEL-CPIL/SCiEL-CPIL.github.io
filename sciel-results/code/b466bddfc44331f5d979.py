import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n_primes = len(primes) # 168
    N = 200  # Search window size. Best known is ~125, so 200 is safe.

    # 2. Precompute coverage indices as NumPy arrays for fast updates
    # cover_arrays[i][r] is an array of indices x in [1, N] such that x % primes[i] == r
    cover_arrays = []
    for p in primes:
        lists = [None] * p
        for r in range(p):
            # List comprehension is fast enough for N=200
            idxs = [x for x in range(1, N + 1) if x % p == r]
            lists[r] = np.array(idxs, dtype=np.int32) if idxs else np.array([], dtype=np.int32)
        cover_arrays.append(lists)

    # 3. Objective function weights
    # Linearly decaying weights to strongly favor covering early integers
    weights = np.arange(N, 0, -1, dtype=np.float64)

    def compute_score(C):
        # C is array of size N+1, C[0] is ignored
        # Returns weighted sum of indicators C[x] >= 2
        return np.sum(weights * (C[1:] >= 2))

    # 4. Initialization: Greedy Heuristic
    C = np.zeros(N + 1, dtype=np.int32)
    residues = [0] * n_primes
    
    # Process primes in increasing order
    for i, p in enumerate(primes):
        # Find smallest x in [1, N] with C[x] < 2
        target_x = -1
        for x in range(1, N + 1):
            if C[x] < 2:
                target_x = x
                break
        
        if target_x == -1:
            # All covered in window, pick random
            r = random.randint(0, p - 1)
        else:
            # Pick r such that p covers target_x
            r = target_x % p
        
        residues[i] = r
        idxs = cover_arrays[i][r]
        if idxs.size > 0:
            C[idxs] += 1

    current_score = compute_score(C)
    best_score = current_score
    best_res = residues[:]
    
    # 5. Simulated Annealing
    T = 20.0  # Initial temperature
    T_min = 0.01
    alpha = 0.9995
    
    # We can also add a restart mechanism if stuck, but simple SA should work.
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        old_idxs = cover_arrays[i][old_r]
        new_idxs = cover_arrays[i][new_r]
        
        # Apply change
        if old_idxs.size > 0:
            C[old_idxs] -= 1
        if new_idxs.size > 0:
            C[new_idxs] += 1
            
        new_score = compute_score(C)
        
        delta = new_score - current_score
        
        # Acceptance probability
        # Use np.exp for speed, though math.exp is also fine
        if delta > 0 or random.random() < np.exp(delta / T):
            residues[i] = new_r
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_res = residues[:]
        else:
            # Rollback
            if new_idxs.size > 0:
                C[new_idxs] -= 1
            if old_idxs.size > 0:
                C[old_idxs] += 1
        
        T *= alpha

    # 6. Final Evaluation
    # Use best_res to compute exact L
    C_final = np.zeros(N + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = best_res[i]
        idxs = cover_arrays[i][r]
        if idxs.size > 0:
            C_final[idxs] += 1
            
    # Find L: largest L such that C_final[x] >= 2 for all 1 <= x <= L
    L = 0
    for x in range(1, N + 1):
        if C_final[x] >= 2:
            L = x
        else:
            break
            
    return best_res

