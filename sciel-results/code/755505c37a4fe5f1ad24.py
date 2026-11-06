"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_up_to(limit):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

def _coverage_length(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    # Use a boolean array up to cap
    covered = [False] * (cap + 1)
    for p, r in classes:
        # Mark all n in [1, cap] with n ≡ r (mod p)
        start = r
        if start < 1:
            start += p * ((1 - r + p - 1) // p)
        for n in range(start, cap + 1, p):
            covered[n] = True
    # Find the largest L such that 1..L all covered
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _sum_reciprocals(classes):
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    all_primes = _primes_up_to(1000)
    # We need sum of 1/p <= 1. Let's figure out which primes we can use.
    # Greedily pick smallest primes first to maximize coverage per unit cost.
    
    # Strategy: Use a subset of primes with sum 1/p <= 1, then optimize residues.
    # The seed uses primes from 13 upward. Let's try to include more small primes.
    
    # First, let's try to find a good set of primes.
    # Sort by 1/p descending (i.e., p ascending) to get max coverage per cost.
    primes_sorted = sorted(all_primes, key=lambda p: 1.0/p, reverse=True)
    
    # Try to include as many small primes as possible within budget 1.
    # But we need to be careful: including too many small primes might waste budget.
    # Let's try different strategies.
    
    best_L = 0
    best_classes = []
    
    # Strategy 1: Use primes in increasing order, greedily adding if budget allows.
    # This gives us the "densest" covering with small moduli.
    budget = Fraction(1, 1)
    selected = []
    for p in primes_sorted:
        cost = Fraction(1, p)
        if budget - cost >= 0:
            selected.append(p)
            budget -= cost
    
    # Now optimize residues for these primes.
    # Start with a heuristic: assign residues to cover as much as possible.
    
    def optimize_residues(prime_set, timeout=None):
        """Use local search to find good residues."""
        n_primes = len(prime_set)
        # Initial assignment: try to cover 1..1000
        residues = [0] * n_primes
        # Heuristic: for each prime, try to cover numbers not yet covered
        # Start with r=0 for all, then improve
        
        # Compute coverage
        def compute_coverage(residues):
            covered = [False] * 1001
            for i, p in enumerate(prime_set):
                r = residues[i]
                start = r
                if start < 1:
                    start += p * ((1 - r + p - 1) // p)
                for n in range(start, 1001, p):
                    covered[n] = True
            # Find largest L
            L = 0
            for n in range(1, 1001):
                if covered[n]:
                    L = n
                else:
                    break
            return L, covered
        
        # Initial: try to maximize coverage of 1..1000
        # Greedy: for each prime in order, pick residue that covers most uncovered numbers
        covered = [False] * 1001
        for i, p in enumerate(prime_set):
            best_r = 0
            best_count = -1
            for r in range(p):
                start = r
                if start < 1:
                    start += p * ((1 - r + p - 1) // p)
                count = 0
                for n in range(start, 1001, p):
                    if not covered[n]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            residues[i] = best_r
            # Update covered
            start = best_r
            if start < 1:
                start += p * ((1 - best_r + p - 1) // p)
            for n in range(start, 1001, p):
                covered[n] = True
        
        L, covered = compute_coverage(residues)
        
        # Local search: try changing one residue at a time
        improved = True
        while improved and (timeout is None or time.time() < timeout):
            improved = False
            for i in range(n_primes):
                p = prime_set[i]
                old_r = residues[i]
                # Try all residues
                best_r = old_r
                best_L = L
                for r in range(p):
                    if r == old_r:
                        continue
                    residues[i] = r
                    new_L, _ = compute_coverage(residues)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                    elif new_L == best_L:
                        # Tie: prefer smaller residue? Or keep old
                        pass
                if best_r != old_r:
                    residues[i] = best_r
                    L = best_L
                    improved = True
        
        return residues, L
    
    # Optimize for the greedy-selected primes
    timeout1 = start_time + 30
    residues1, L1 = optimize_residues(selected, timeout1)
    if L1 > best_L:
        best_L = L1
        best_classes = [[selected[i], residues1[i]] for i in range(len(selected))]
    
    # Strategy 2: Try different prime sets
    # Maybe skip some primes to allow better residue assignment?
    # Or try including/excluding specific primes.
    
    # Let's also try the original seed approach as a baseline
    seed_primes = [p for p in range(13, 1001) if _is_prime(p)]
    seed_budget = sum(Fraction(1, p) for p in seed_primes)
    if seed_budget <= 1:
        seed_residues = [(i + 1) % p for i, p in enumerate(seed_primes)]
        L_seed = _coverage_length([[p, r] for p, r in zip(seed_primes, seed_residues)])
        if L_seed > best_L:
            best_L = L_seed
            best_classes = [[p, r] for p, r in zip(seed_primes, seed_residues)]
    
    # Try a few random prime subsets
    random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        # Random subset with budget constraint
        random_primes = random.sample(all_primes, min(20, len(all_primes)))
        random_primes.sort()
        random_budget = Fraction(0, 1)
        random_selected = []
        for p in random_primes:
            cost = Fraction(1, p)
            if random_budget + cost <= 1:
                random_selected.append(p)
                random_budget += cost
        if len(random_selected) < 3:
            continue
        timeout_t = min(start_time + 20, deadline - 5)
        residues_t, L_t = optimize_residues(random_selected, timeout_t)
        if L_t > best_L:
            best_L = L_t
            best_classes = [[random_selected[i], residues_t[i]] for i in range(len(random_selected))]
    
    # Verify the best solution
    if best_classes:
        total = _sum_reciprocals(best_classes)
        if total > 1:
            # Fallback to seed
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
            best_L = _coverage_length(best_classes)
    
    return best_classes
# EVOLVE-BLOCK-END
