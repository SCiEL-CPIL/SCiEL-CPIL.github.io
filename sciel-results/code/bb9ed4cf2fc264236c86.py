import time
import random
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # 1. Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    
    # We need to select a subset of primes S such that sum(1/p) <= 1.
    # And assign a residue r_p to each p in S to maximize L.

    # Strategy:
    # 1. Heuristically select a "good" set of primes.
    #    Small primes are "expensive" but cover broadly.
    #    A common heuristic is to include small primes up to a certain limit,
    #    then fill the remaining budget with larger primes.
    #    Let's try a few candidate sets of primes.
    
    # Candidate 1: Greedy selection by "coverage efficiency" per cost?
    # Actually, let's look at the budget.
    # Sum(1/p) for p in {2, 3, 5, 7, 11, 13, 17, 19, 23}
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23
    # ~ 0.5 + 0.33 + 0.2 + 0.14 + 0.09 + 0.077 + 0.058 + 0.052 + 0.043
    # Sum ~ 1.49. Too much.
    
    # Let's try {2, 3, 5, 7}. Sum ~ 0.5 + 0.33 + 0.2 + 0.14 = 1.17. Too much.
    # {2, 3, 5}. Sum ~ 0.5 + 0.33 + 0.2 = 1.03. Too much.
    # {2, 3}. Sum ~ 0.83. Remaining 0.17.
    # Can we add 7? 0.83 + 0.14 = 0.97. Yes.
    # So {2, 3, 7} is a valid set.
    # Remaining 0.03. Can we add 11? 1/11 ~ 0.09. No.
    # Can we add a large prime? 1/p <= 0.03 => p >= 34.
    # So we could add {2, 3, 7, 37}.
    
    # Let's try {2, 5, 7, 11, 13}.
    # 0.5 + 0.2 + 0.14 + 0.09 + 0.077 = 0.907.
    # Remaining 0.093. Can add 17 (0.058). Sum = 0.965.
    # Can add 19 (0.052)? 0.965+0.052 > 1.
    # Can add 23 (0.043)? 0.965+0.043 < 1.
    # So {2, 5, 7, 11, 13, 17, 23} is a valid set?
    # Sum: 0.5+0.2+0.1428+0.0909+0.0769+0.0588+0.0435 = 1.1129. No, my previous sum was wrong.
    # Let's calculate exactly.
    
    def sum_recip(prime_list):
        s = Fraction(0)
        for p in prime_list:
            s += Fraction(1, p)
        return s

    # Let's generate candidate prime sets using a greedy algorithm that maximizes "coverage potential" per cost,
    # but since coverage is hard to estimate without assigning residues, we can use a proxy:
    # Smaller primes are better for small L?
    # Actually, for covering 1..L, we want primes that are coprime to each other as much as possible?
    # No, distinct primes are always coprime.
    
    # Let's use a local search on the prime set itself first?
    # Or just try a few fixed sets that are known to be "dense".
    
    # Set A: {2, 3, 7, 31}
    # 1/2 + 1/3 + 1/7 + 1/31 = 0.5 + 0.333 + 0.142 + 0.032 = 1.008 > 1.
    # {2, 3, 7, 37}
    # 0.5 + 0.333 + 0.142 + 0.027 = 0.999. Very close.
    
    # Set B: {3, 5, 7, 11, 13, 17, 19} (no 2)
    # 0.333 + 0.2 + 0.142 + 0.09 + 0.076 + 0.058 + 0.052 = 0.952.
    # Remaining 0.048. Add 23 (0.043). Sum = 0.995.
    # {3, 5, 7, 11, 13, 17, 19, 23}
    
    # Set C: {2, 5, 11, 13, 17, 19, 23}
    # 0.5 + 0.2 + 0.09 + 0.076 + 0.058 + 0.052 + 0.043 = 1.019 > 1.
    
    # Let's use a randomized search to find good prime sets.
    # Actually, let's just use the CSP approach on a promising set of primes.
    # The CSP approach (from inspiration 2) seems powerful.
    
    # Let's define a function to solve for residues given a set of primes.
    # We will use a randomized local search / simulated annealing on the residues.
    
    def solve_residues(prime_set, L_target, time_limit):
        # prime_set: list of primes
        # L_target: int
        # Returns: dict {p: r} or None
        
        # State: r = {p: r_p}
        # Objective: Maximize L such that 1..L is covered.
        # Constraint: sum(1/p) <= 1 (already satisfied by prime_set selection)
        
        # Precompute coverage masks for each prime and residue?
        # Or just check coverage on the fly?
        # On the fly: For each n in 1..L, check if any p | (n - r_p).
        # This is O(L * |prime_set|).
        # L up to 1000. |prime_set| ~ 10.
        # 1000 * 10 = 10,000 ops per evaluation.
        # Very fast.
        
        # Initial state: random residues
        r = {p: random.randint(0, p - 1) for p in prime_set}
        
        # Evaluate L
        def evaluate(residues):
            covered = [False] * (L_target + 1)
            for n in range(1, L_target + 1):
                for p in prime_set:
                    if (n - residues[p]) % p == 0:
                        covered[n] = True
                        break
            # Find largest L such that 1..L is covered
            L = 0
            for n in range(1, L_target + 1):
                if not covered[n]:
                    break
                L = n
            return L
        
        best_L = 0
        best_r = None
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9999
        min_T = 0.001
        
        t_start = time.time()
        
        while T > min_T and time.time() - t_start < time_limit:
            current_L = evaluate(r)
            
            if current_L > best_L:
                best_L = current_L
                best_r = dict(r)
            
            # Pick a random prime to change
            p = random.choice(prime_set)
            old_r = r[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            r[p] = new_r
            new_L = evaluate(r)
            
            # Accept?
            if new_L >= current_L:
                # Accept
                pass
            else:
                # Probabilistic acceptance
                # Delta = current_L - new_L
                # Prob = exp(-Delta / T)
                # Since L is discrete, Delta >= 1.
                delta = current_L - new_L
                if delta == 0:
                    prob = 1.0
                else:
                    prob = pow(2.71828, -delta / T)
                
                if random.random() > prob:
                    # Reject
                    r[p] = old_r
            
            T *= cooling
        
        return best_r, best_L

    # Let's try a few candidate prime sets.
    # We want sets with sum(1/p) close to 1 to maximize "flexibility".
    
    candidate_sets = []
    
    # Set 1: {2, 3, 7, 37}
    s1 = [2, 3, 7, 37]
    if sum_recip(s1) <= 1:
        candidate_sets.append(s1)
        
    # Set 2: {3, 5, 7, 11, 13, 17, 19, 23}
    s2 = [3, 5, 7, 11, 13, 17, 19, 23]
    if sum_recip(s2) <= 1:
        candidate_sets.append(s2)
        
    # Set 3: {2, 5, 13, 17, 19, 23, 29, 31}
    # 0.5 + 0.2 + 0.076 + 0.058 + 0.052 + 0.043 + 0.034 + 0.032 = 0.996
    s3 = [2, 5, 13, 17, 19, 23, 29, 31]
    if sum_recip(s3) <= 1:
        candidate_sets.append(s3)
        
    # Set 4: {2, 3, 5, 7, 17, 23, 31}
    # 0.5 + 0.333 + 0.2 + 0.142 + 0.058 + 0.043 + 0.032 = 1.31 > 1. No.
    
    # Set 5: {2, 3, 7, 43}
    # 0.5 + 0.333 + 0.142 + 0.023 = 0.999
    s5 = [2, 3, 7, 43]
    if sum_recip(s5) <= 1:
        candidate_sets.append(s5)
        
    # Set 6: {2, 3, 11, 13, 17, 23}
    # 0.5 + 0.333 + 0.09 + 0.076 + 0.058 + 0.043 = 1.09 > 1. No.
    
    # Set 7: {2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47} (Too many, sum > 1)
    
    # Let's add a generic greedy set builder.
    # Start with smallest primes, add if budget allows.
    # This gives {2, 3, 7} (cost 0.97).
    # Then add next smallest prime that fits.
    # 11 (0.09) -> 1.06. No.
    # 13 (0.077) -> 1.05. No.
    # 17 (0.058) -> 1.03. No.
    # 19 (0.052) -> 1.02. No.
    # 23 (0.043) -> 1.01. No.
    # 29 (0.034) -> 1.00. Yes (if exactly 1, or close).
    # Let's check 1/2 + 1/3 + 1/7 + 1/29.
    # 1/2 = 0.5
    # 1/3 = 0.3333
    # 1/7 = 0.1428
    # 1/29 = 0.0344
    # Sum = 1.0106. No.
    # 1/31 = 0.0322. Sum = 1.0085. No.
    # 1/37 = 0.0270. Sum = 1.0033. No.
    # 1/41 = 0.0243. Sum = 1.0005. No.
    # 1/43 = 0.0232. Sum = 0.9995. Yes.
    # So {2, 3, 7, 43} is good.
    
    # Let's also try {2, 3, 5, 13, 17, 23}
    # 0.5 + 0.333 + 0.2 + 0.076 + 0.058 + 0.043 = 1.21. No.
    
    # {2, 3, 5, 7, 17, 23, 31}
    # 0.5 + 0.333 + 0.2 + 0.142 + 0.058 + 0.043 + 0.032 = 1.31. No.
    
    # {2, 3, 5, 11, 13, 17, 19, 23}
    # 0.5 + 0.333 + 0.2 + 0.09 + 0.076 + 0.058 + 0.052 + 0.043 = 1.35. No.
    
    # {2, 3, 7, 37} and {2, 3, 7, 43} are good.
    # {3, 5, 7, 11, 13, 17, 19, 23} is good.
    # {2, 5, 13, 17, 19, 23, 29, 31} is good.
    
    # Let's run the solver on each candidate set, allocating time based on remaining budget.
    
    best_overall_L = 0
    best_overall_classes = []
    
    # Estimate time per set.
    # Total time left: deadline - time.time()
    # Number of sets: len(candidate_sets)
    # Allocate time evenly, with a safety margin.
    
    num_sets = len(candidate_sets)
    if num_sets == 0:
        # Fallback: just return empty or a single prime
        return []
        
    time_per_set = (deadline - time.time() - 2.0) / num_sets
    if time_per_set < 0:
        time_per_set = 1.0
        
    for i, prime_set in enumerate(candidate_sets):
        remaining = deadline - time.time() - 1.0
        if remaining < 1.0:
            break
            
        # Use min(time_per_set, remaining)
        t_alloc = min(time_per_set, remaining)
        
        # Run solver
        try:
            residues, L = solve_residues(prime_set, 1000, t_alloc)
            
            if residues is not None and L > best_overall_L:
                best_overall_L = L
                # Construct classes
                classes = [[p, residues[p]] for p in prime_set]
                best_overall_classes = classes
                
        except Exception:
            pass

    # If no solution found, or L is 0, try a simple greedy set cover as fallback?
    # The current program's greedy is slow but reliable for small L.
    # Let's just return the best found.
    
    return best_overall_classes

# EVOLVE-BLOCK-END
