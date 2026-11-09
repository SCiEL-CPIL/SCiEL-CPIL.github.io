import time
import numpy as np
from fractions import Fraction
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    N = 10000  # Cap for L evaluation

    # Generate a small, diverse set of candidate prime lists
    candidate_sets = []
    
    # 1. Greedy: Smallest primes first
    ps = []
    s = Fraction(0, 1)
    for p in primes:
        if s + Fraction(1, p) <= 1:
            ps.append(p)
            s += Fraction(1, p)
    candidate_sets.append(ps)

    # 2. Greedy: Skip 2
    ps = []
    s = Fraction(0, 1)
    for p in primes:
        if p == 2: continue
        if s + Fraction(1, p) <= 1:
            ps.append(p)
            s += Fraction(1, p)
    candidate_sets.append(ps)

    # 3. Greedy: Skip 2, 3
    ps = []
    s = Fraction(0, 1)
    for p in primes:
        if p in (2, 3): continue
        if s + Fraction(1, p) <= 1:
            ps.append(p)
            s += Fraction(1, p)
    candidate_sets.append(ps)

    # 4. Random subsets (fixed seed for reproducibility)
    random.seed(42)
    for _ in range(50):
        ps = []
        s = Fraction(0, 1)
        shuffled = primes[:]
        random.shuffle(shuffled)
        for p in shuffled:
            if s + Fraction(1, p) <= 1:
                ps.append(p)
                s += Fraction(1, p)
        if len(ps) > 0:
            candidate_sets.append(ps)

    # Deduplicate
    unique_sets = []
    seen = set()
    for ps in candidate_sets:
        key = tuple(sorted(ps))
        if key not in seen:
            seen.add(key)
            unique_sets.append(ps)

    best_L = 0
    best_classes = []

    def optimize_residues(prime_list, time_limit, rng_seed=0):
        if not prime_list:
            return 0, []

        k = len(prime_list)
        rng = random.Random(rng_seed)

        # Precompute indices for each prime and residue
        # To save memory and time, we only compute when needed or cache small primes
        # For p > 1000, we don't have them. Max p is 997.
        # N is 10000.
        
        # Initialize residues randomly
        residues = [rng.randint(0, p - 1) for p in prime_list]

        # Coverage array
        counts = np.zeros(N + 1, dtype=np.int32)
        
        def update_counts(p, r, delta):
            if r == 0:
                start = p
            else:
                start = r
            if start > N:
                return
            idxs = np.arange(start, N + 1, p)
            counts[idxs] += delta

        for i, p in enumerate(prime_list):
            update_counts(p, residues[i], 1)

        covered = (counts > 0).astype(np.int8)

        def find_L():
            # Find first 0 in covered[1:]
            # Using argmax on inverted boolean is fast
            if not np.any(covered[1:]):
                return N
            # First index where covered is 0
            zeros = np.where(covered[1:] == 0)[0]
            if len(zeros) == 0:
                return N
            return int(zeros[0])

        L = find_L()
        best_L_local = L
        best_residues = residues[:]

        # Simulated Annealing
        T = 5.0
        T_min = 0.05
        cooling = 0.995
        steps = 0
        max_steps = 100000 # Hard cap on steps to prevent infinite loop if time check is slow

        while time.time() < time_limit and T > T_min and steps < max_steps:
            steps += 1
            i = rng.randint(0, k - 1)
            p = prime_list[i]
            old_r = residues[i]
            
            # Bias towards covering the current gap
            if L < N and rng.random() < 0.9:
                new_r = (L + 1) % p
            else:
                new_r = rng.randint(0, p - 1)

            if new_r == old_r:
                continue

            # Update counts
            update_counts(p, old_r, -1)
            update_counts(p, new_r, 1)
            
            # Update covered status for affected indices
            # We need to update covered for indices touched by old_r or new_r
            # Instead of unique concatenation which is slow, we can just recompute covered for the union
            # Or simpler: just check if L improved.
            # Finding L is O(N) which is 10000. This is fast enough for ~10k steps.
            
            # Optimization: Only recompute L if the change might affect the prefix
            # A change at residue r for prime p affects indices r, r+p, ...
            # If L is small, we only care about indices <= L+1.
            
            # Let's just recompute covered and L. It's simple and robust.
            covered = (counts > 0).astype(np.int8)
            new_L = find_L()

            residues[i] = new_r

            if new_L > L or rng.random() < math.exp((new_L - L) / T):
                L = new_L
                if L > best_L_local:
                    best_L_local = L
                    best_residues = residues[:]
            else:
                # Revert
                residues[i] = old_r
                update_counts(p, new_r, -1)
                update_counts(p, old_r, 1)
                # Revert covered (not strictly necessary if we recompute every time, but keeps state consistent if we stop mid-way)
                covered = (counts > 0).astype(np.int8)

            T *= cooling

        classes = [[prime_list[i], best_residues[i]] for i in range(k)]
        return best_L_local, classes

    n_sets = len(unique_sets)
    if n_sets == 0:
        return [[2, 0]]

    # Allocate time per set
    # We want to spend most time on the most promising sets.
    # Let's give each set equal time, but cap it.
    time_per_set = (deadline - start_time - 5.0) / n_sets

    for idx, ps in enumerate(unique_sets):
        if time.time() > deadline - 5:
            break
        
        t_limit = time.time() + min(time_per_set, deadline - time.time() - 5)
        L, classes = optimize_residues(ps, t_limit, rng_seed=idx * 1000 + 42)
        
        if L > best_L:
            best_L = L
            best_classes = classes

    if best_classes:
        s = Fraction(0, 1)
        for p, r in best_classes:
            s += Fraction(1, p)
        if s <= 1:
            return best_classes
    
    # Fallback
    return [[2, 0]]
