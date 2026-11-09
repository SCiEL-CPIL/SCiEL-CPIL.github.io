import time
from fractions import Fraction
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    primes_set = set(primes)
    
    # Precompute reciprocal values for quick sum checks if needed, 
    # but Fraction is required for exactness.
    # We will maintain the sum as a Fraction.

    def check_budget(prime_list):
        s = Fraction(0, 1)
        for p in prime_list:
            s += Fraction(1, p)
        return s <= 1

    # --- Strategy: Gap-Filling CSP ---
    # State:
    #   covered: boolean array (or bitset) for 1..L_max
    #   used_primes: set of primes used
    #   budget_remaining: Fraction
    #   uncovered_gaps: list of integers not covered
    
    # We want to maximize L such that 1..L is covered.
    # We will grow L dynamically.
    
    # Initial L estimate: The density is 1. 
    # If we use primes up to P, sum(1/p) ~ log(log(P)). 
    # To get sum <= 1, we can't use too many small primes.
    # Let's start with a greedy construction to get a baseline L.
    
    L_max_search = 100000 # Cap
    # But we expect L to be around 1000-2000 based on the prompt hint (L>=999).
    # Let's set a working L for the coverage array. 
    # If we cover 1..L, we can try to extend to L+1, etc.
    # However, managing a dynamic array is complex. 
    # Instead, we fix a target L range and try to cover it.
    # Since the score is the *largest* L covered, we should try to cover as much as possible.
    # Let's assume L is in the range [1000, 5000].
    
    # Let's use a fixed window for the coverage array to keep memory and time bounded.
    # If we can cover 1..5000, we score 5000 (or higher if we extend, but let's start with a fixed size).
    # Actually, the score is capped at 10^5.
    # Let's allocate an array of size 10001 to be safe and fast.
    N = 10000
    covered = bytearray(N + 1) # 0 means not covered, 1 means covered. Index 0 unused.
    
    # We will use a list of active primes and their residues.
    # active_primes: list of p
    # residues: list of r
    
    # Greedy initialization:
    # Try to cover 1..N by adding primes one by one.
    # At each step, choose the prime p (not used) and residue r that covers the most new gaps.
    # This is expensive if done naively (O(N * #primes * #residues)).
    # Optimization: 
    #   Maintain a list of uncovered numbers.
    #   For a candidate prime p, the best residue r is the one that hits the most uncovered numbers in 1..N.
    #   The number of hits for residue r is count of x in uncovered such that x % p == r.
    #   We can precompute for each p, the counts for each residue r among the *currently uncovered* numbers.
    
    # Let's implement this gap-filling heuristic.
    
    # Initialize all as uncovered
    # covered is all 0s initially.
    uncovered_count = N
    
    # We need to track which primes are used.
    used_primes = set()
    active_primes = []
    residues = []
    
    # To speed up, we can pre-sort primes. Smaller primes cover more numbers on average?
    # No, smaller p has more residues but each residue covers N/p numbers.
    # Larger p has fewer residues but each covers N/p (smaller) numbers.
    # Actually, the total coverage of a prime p is N. 
    # The "efficiency" in terms of budget 1/p is N * (1/p) / (1/p) = N? No.
    # Each prime covers exactly N/p numbers on average.
    # Cost is 1/p.
    # Ratio: (N/p) / (1/p) = N. It's constant!
    # So any prime covers the same *proportion* of the range per unit budget.
    # The difference is in the *pattern* of coverage (clumping).
    # Smaller primes have more frequent hits, potentially filling gaps better.
    # Let's try a mix.
    
    # Heuristic:
    # 1. Start with no primes.
    # 2. While budget allows and gaps exist:
    #    a. Find the prime p (from available) that maximizes the maximum coverage gain for any residue r.
    #    b. Select that p and the best r.
    #    c. Update covered array.
    
    # To make this efficient:
    # We maintain a list of uncovered indices.
    # For each candidate prime p, we want to know: for each r in 0..p-1, how many uncovered x have x%p==r?
    # Updating this for every step is O(N * #primes). Too slow.
    
    # Alternative:
    # Use a smaller set of candidate primes. 
    # Since sum(1/p) <= 1, we can't use many primes.
    # 1/2 + 1/3 + 1/5 + ... 
    # Let's estimate max number of primes.
    # If we use smallest primes:
    # 2: 0.5
    # 3: 0.833
    # 5: 1.033 (Exceeds 1 if we add 5 to 2,3? No, 1/2+1/3+1/5 = 0.5+0.33+0.2 = 1.033 > 1)
    # So {2,3} is valid. {2,5} is valid (0.7). {3,5} is valid (0.533).
    # {2,3,5} is invalid.
    # {2,5,7} = 0.5+0.2+0.142 = 0.842. Valid.
    # {2,5,7,11} = 0.842+0.09 = 0.932. Valid.
    # {2,5,7,11,13} = 0.932+0.077 = 1.009. Invalid.
    # So we can have about 4-5 small primes.
    # Or more larger primes.
    
    # Let's generate a few candidate prime sets manually or via greedy selection of primes.
    # Strategy: Generate several candidate sets of primes.
    # For each set, solve for residues using a local search (SA or Hill Climbing) to maximize coverage of 1..N.
    
    # Candidate Set Generation:
    # 1. Greedy by smallest primes: Pick smallest p if budget allows.
    # 2. Greedy by largest primes: Pick largest p? (Covers sparser patterns, might leave gaps).
    # 3. Randomized sets.
    
    candidate_sets = []
    
    # Set 1: Smallest primes
    ps = []
    s = Fraction(0,1)
    for p in primes:
        if s + Fraction(1,p) <= 1:
            ps.append(p)
            s += Fraction(1,p)
    candidate_sets.append(ps)
    
    # Set 2: Skip 2, take next smallest
    ps = []
    s = Fraction(0,1)
    for p in primes:
        if p == 2: continue
        if s + Fraction(1,p) <= 1:
            ps.append(p)
            s += Fraction(1,p)
    candidate_sets.append(ps)
    
    # Set 3: Skip 2 and 3
    ps = []
    s = Fraction(0,1)
    for p in primes:
        if p in (2,3): continue
        if s + Fraction(1,p) <= 1:
            ps.append(p)
            s += Fraction(1,p)
    candidate_sets.append(ps)
    
    # Set 4: Randomized
    random.seed(42)
    for _ in range(10):
        ps = []
        s = Fraction(0,1)
        shuffled = primes[:]
        random.shuffle(shuffled)
        for p in shuffled:
            if s + Fraction(1,p) <= 1:
                ps.append(p)
                s += Fraction(1,p)
        if ps:
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
    
    # For each candidate set, run a local search to optimize residues.
    # Objective: Maximize L such that 1..L is covered.
    # We will use the coverage array of size N=10000.
    # Score for a residue assignment: find the first gap in 1..N. That is L.
    
    def optimize_residues(prime_list, time_limit):
        if not prime_list:
            return 0, []
            
        k = len(prime_list)
        # Initialize residues randomly
        residues = [random.randint(0, p-1) for p in prime_list]
        
        # Build coverage
        covered = bytearray(N + 1)
        # We need to efficiently update coverage when a residue changes.
        # Let's store for each prime i, the list of indices it covers?
        # No, that's memory heavy if we store lists.
        # Instead, when we change residue of prime i from r_old to r_new:
        # 1. Decrement count for all x in 1..N where x % p_i == r_old.
        # 2. Increment count for all x in 1..N where x % p_i == r_new.
        # 3. Update covered[x] if count goes 0->1 or 1->0.
        
        # We need a count array.
        counts = [0] * (N + 1)
        
        # Init counts
        for i, p in enumerate(prime_list):
            r = residues[i]
            start = r if r > 0 else p
            for x in range(start, N + 1, p):
                counts[x] += 1
        
        # Init covered
        for x in range(1, N + 1):
            if counts[x] > 0:
                covered[x] = 1
        
        # Find current L
        L = 0
        for x in range(1, N + 1):
            if covered[x]:
                L = x
            else:
                break
        
        best_L_local = L
        best_residues = residues[:]
        
        # Simulated Annealing / Hill Climbing
        # Move: Change residue of one prime.
        # To speed up, we can precompute the list of x for each (p, r)?
        # That would be sum(p) lists. Total size N * k? No.
        # For each prime p, there are p residues. Each residue covers ~N/p numbers.
        # Total numbers stored: sum(p * N/p) = sum(N) = N * k.
        # N=10000, k~5. 50000 integers. This is fine.
        
        # Precompute indices for each prime and residue
        prime_indices = []
        for p in prime_list:
            indices_for_p = []
            for r in range(p):
                idxs = []
                start = r if r > 0 else p
                for x in range(start, N + 1, p):
                    idxs.append(x)
                indices_for_p.append(idxs)
            prime_indices.append(indices_for_p)
        
        rng = random.Random(123)
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        current_L = L
        
        while time.time() < time_limit and T > T_min:
            # Pick a prime to modify
            i = rng.randint(0, k - 1)
            p = prime_list[i]
            old_r = residues[i]
            
            # Pick new residue
            # Bias towards residues that cover the first gap?
            # First gap is current_L + 1
            first_gap = current_L + 1
            if first_gap <= N and rng.random() < 0.7:
                # Greedy: choose residue that covers first_gap
                new_r = first_gap % p
            else:
                new_r = rng.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Apply move
            # Remove old coverage
            for x in prime_indices[i][old_r]:
                counts[x] -= 1
                if counts[x] == 0:
                    covered[x] = 0
            
            # Add new coverage
            for x in prime_indices[i][new_r]:
                counts[x] += 1
                if counts[x] == 1:
                    covered[x] = 1
            
            residues[i] = new_r
            
            # Recalculate L
            # Optimization: If we broke coverage at some point, L might drop.
            # If we fixed the first gap, L might increase.
            # We can just scan from 1. N=10000 is fast enough in Python?
            # 10000 iterations * (1/cooling) steps. 
            # Steps: T goes 5 -> 0.01. ln(500)/ln(1/0.9995) ~ 6.2 / 0.0005 ~ 12400 steps.
            # 12400 * 10000 = 1.24e8 operations. Too slow in Python.
            
            # We need a faster way to find L.
            # Maintain a set of gaps?
            # Or, note that L only changes if the status of numbers near the current L changes.
            # If we cover a number < L, L doesn't change (unless we uncover it, then L drops).
            # If we cover a number > L, L might increase if all numbers between L and that number are covered.
            
            # Let's maintain a list of gaps (uncovered numbers).
            # If we add coverage, remove x from gaps if present.
            # If we remove coverage, add x to gaps if not present.
            # Then L is the smallest gap - 1.
            
            # Let's re-implement with a gap set.
            # This requires re-initializing the state for this specific optimization run.
            
            break # Break out of SA loop in this draft, let's restart with better structure.
            
        # This inner loop is getting complex. Let's simplify.
        # We will just use a simple hill climbing with a fixed number of steps or time limit,
        # and accept the speed cost of scanning L if necessary, or optimize L finding.
        
        # Let's try a different approach: 
        # Since N=10000 is small, scanning is O(N).
        # If we do 1000 steps, that's 10^7 ops. Doable.
        # Let's limit steps to 5000 per prime set.
        
        # Reset state for a cleaner SA
        covered = bytearray(N + 1)
        counts = [0] * (N + 1)
        residues = [random.randint(0, p-1) for p in prime_list]
        for i, p in enumerate(prime_list):
            r = residues[i]
            for x in prime_indices[i][r]:
                counts[x] += 1
        for x in range(1, N + 1):
            if counts[x] > 0:
                covered[x] = 1
        
        L = 0
        for x in range(1, N + 1):
            if covered[x]:
                L = x
            else:
                break
        
        best_L_local = L
        best_residues = residues[:]
        
        steps = 0
        max_steps = 2000 # Adjust based on time
        T = 10.0
        cooling = 0.995
        
        while steps < max_steps and time.time() < time_limit:
            i = rng.randint(0, k - 1)
            p = prime_list[i]
            old_r = residues[i]
            
            # Targeted move
            if L < N and rng.random() < 0.8:
                new_r = (L + 1) % p
            else:
                new_r = rng.randint(0, p - 1)
            
            if new_r == old_r:
                continue
                
            # Apply
            for x in prime_indices[i][old_r]:
                counts[x] -= 1
                if counts[x] == 0:
                    covered[x] = 0
            for x in prime_indices[i][new_r]:
                counts[x] += 1
                if counts[x] == 1:
                    covered[x] = 1
            residues[i] = new_r
            
            # Update L
            # If we uncovered something <= L, L might drop.
            # If we covered L+1, L might increase.
            # Simplest: rescan.
            # To optimize: 
            # If we only changed coverage of numbers > L, L stays same or increases.
            # If we changed coverage of numbers <= L, L might drop.
            
            # Let's just rescan. 2000 steps * 10000 = 2e7. Should be < 1s in PyPy, maybe 2-3s in CPython.
            # We have 145s total. 
            # We have ~15 candidate sets. 15 * 3s = 45s. Safe.
            
            new_L = 0
            for x in range(1, N + 1):
                if covered[x]:
                    new_L = x
                else:
                    break
            
            if new_L > L or rng.random() < math.exp((new_L - L) / T):
                L = new_L
                if L > best_L_local:
                    best_L_local = L
                    best_residues = residues[:]
            else:
                # Revert
                for x in prime_indices[i][new_r]:
                    counts[x] -= 1
                    if counts[x] == 0:
                        covered[x] = 0
                for x in prime_indices[i][old_r]:
                    counts[x] += 1
                    if counts[x] == 1:
                        covered[x] = 1
                residues[i] = old_r
                
            T *= cooling
            steps += 1
            
        classes = [[prime_list[i], best_residues[i]] for i in range(k)]
        return best_L_local, classes

    # Run optimization for each set
    time_per_set = (deadline - time.time()) / len(unique_sets)
    
    for ps in unique_sets:
        if time.time() > deadline - 5:
            break
        t_limit = time.time() + min(time_per_set, deadline - time.time() - 5)
        L, classes = optimize_residues(ps, t_limit)
        if L > best_L:
            best_L = L
            best_classes = classes

    # Final check
    if best_classes:
        s = Fraction(0,1)
        for p, r in best_classes:
            s += Fraction(1, p)
        if s <= 1:
            return best_classes
    return [[2, 0]]
