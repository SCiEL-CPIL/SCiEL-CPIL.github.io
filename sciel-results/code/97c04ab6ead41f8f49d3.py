import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
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
    n = len(primes)  # 168
    
    # We will evaluate coverage up to M
    M = 100000
    
    # Initialize residues
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Build initial counts
    counts = np.zeros(M + 1, dtype=np.int32)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        r_eff = p if r == 0 else r
        if r_eff <= M:
            positions = np.arange(r_eff, M + 1, p)
            counts[positions] += 1
            
    def get_L(counts):
        # Find first index >= 1 where count < 2
        # counts[0] is unused for the problem statement (1..L)
        deficient = np.where(counts[1:] < 2)[0]
        if len(deficient) == 0:
            return M
        return int(deficient[0]) + 1

    current_L = get_L(counts)
    best_L = current_L
    best_residues = residues[:]
    
    # Simulated Annealing
    T_start = 100.0
    T_end = 0.1
    # We will run for a fixed number of iterations or until time runs out
    # Let's estimate iterations. Each iteration is fast.
    # Let's try 20000 iterations.
    
    def update_and_get_L(i, old_r, new_r):
        p = primes[i]
        
        # Remove old contribution
        old_eff = p if old_r == 0 else old_r
        if old_eff <= M:
            positions = np.arange(old_eff, M + 1, p)
            counts[positions] -= 1
            
        # Add new contribution
        new_eff = p if new_r == 0 else new_r
        if new_eff <= M:
            positions = np.arange(new_eff, M + 1, p)
            counts[positions] += 1
            
        return get_L(counts)

    random.seed(42)
    
    # Greedy Initialization to get a better start
    # For each prime, pick residue that maximizes coverage of [1, 1000]
    # This is expensive if done naively, but M is small for this part.
    # Let's just use the random start for SA, it's usually good enough with high T.
    
    # SA Loop
    for k in range(50000): # Max iterations, will break on time
        if time.time() > deadline:
            break
            
        # Determine temperature
        # Linear cooling
        T = T_start * (T_end / T_start) ** (k / 50000.0)
        
        # Pick random prime
        i = random.randint(0, n - 1)
        p = primes[i]
        old_r = residues[i]
        
        # Pick new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Calculate new L
        new_L = update_and_get_L(i, old_r, new_r)
        
        # Acceptance
        if new_L >= current_L:
            # Accept
            current_L = new_L
            residues[i] = new_r
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
        else:
            # Reject or accept with probability exp((new_L - current_L) / T)
            delta = new_L - current_L
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_L = new_L
                    residues[i] = new_r
                else:
                    # Revert change in counts
                    update_and_get_L(i, new_r, old_r)
            # If delta >= 0, it would have been accepted above, so this branch is only for delta < 0 rejected
            # Wait, the logic above: if new_L >= current_L, accept.
            # Else (new_L < current_L), check prob.
            # If rejected, revert.
            
    # Final Hill Climbing pass to squeeze out more
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            p = primes[i]
            old_r = residues[i]
            best_local_L = current_L
            best_local_r = old_r
            
            # Try all residues for this prime
            # For large p, this is expensive. 
            # But M is 50000. 
            # Number of positions affected is M/p.
            # Cost to try one residue: O(M/p).
            # Total cost for prime p: O(M).
            # Total cost for all primes: O(n_primes * M) = 168 * 50000 = 8.4M ops.
            # This is fast.
            
            for new_r in range(p):
                if new_r == old_r:
                    continue
                new_L = update_and_get_L(i, old_r, new_r)
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_local_r = new_r
            
            if best_local_r != old_r:
                residues[i] = best_local_r
                current_L = best_local_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                improved = True
                # Note: We modified counts in the loop. 
                # The last update_and_get_L left counts in the state of best_local_r.
                # So counts are consistent with residues.
                
    # Verify and return
    # Rebuild counts to be safe? No, counts are maintained.
    # But let's do a final check with a larger M just in case?
    # The problem says L is capped at 10^5.
    # If our M=50000 is the bottleneck, we might miss a larger L.
    # However, usually the "bottleneck" appears early.
    # If we find L=50000, we should check further.
    
    # Let's check if L reached M
    if best_L >= M:
        # Extend M
        M_new = 100000
        counts_new = np.zeros(M_new + 1, dtype=np.int32)
        for i in range(n):
            p = primes[i]
            r = best_residues[i]
            r_eff = p if r == 0 else r
            if r_eff <= M_new:
                positions = np.arange(r_eff, M_new + 1, p)
                counts_new[positions] += 1
        deficient = np.where(counts_new[1:] < 2)[0]
        if len(deficient) == 0:
            best_L = M_new
        else:
            best_L = int(deficient[0]) + 1

    return [int(r) for r in best_residues]
