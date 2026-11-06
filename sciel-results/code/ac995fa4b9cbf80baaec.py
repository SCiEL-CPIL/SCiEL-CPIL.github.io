import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # We need to select a set of primes such that sum(1/p) <= 1.
    # The sum of reciprocals of all primes >= 13 is approx 0.94.
    # Let's verify this with Fractions.
    # Primes: 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, ...
    
    # Strategy:
    # 1. Identify a valid set of primes.
    # 2. Use local search to assign residues to maximize coverage length L.
    
    # Let's try to include as many small primes as possible while staying under 1.
    # Sum of 1/p for p in {2, 3, 5, 7, 11, 13, ...}
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 = 0.5 + 0.33 + 0.2 + 0.14 + 0.09 = 1.26 (Too high)
    # Drop 2: 1/3 + 1/5 + 1/7 + 1/11 = 0.76. Can we add 13? 0.76 + 0.076 = 0.836.
    # Can we add 17? 0.836 + 0.058 = 0.894.
    # Can we add 19? 0.894 + 0.052 = 0.946.
    # Can we add 23? 0.946 + 0.043 = 0.989.
    # Can we add 29? 0.989 + 0.034 = 1.023 (Too high).
    # So {3, 5, 7, 11, 13, 17, 19, 23} is a candidate set with cost ~0.989.
    # Let's check if we can swap to get better coverage.
    # Actually, the seed program used primes >= 13.
    # Sum of 1/p for p >= 13:
    # 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31 + 1/37 + 1/41 + 1/43 + 1/47 + 1/53 + 1/59 + 1/61 + 1/67 + 1/71 + 1/73 + 1/79 + 1/83 + 1/89 + 1/97 + ...
    # This sum is known to be less than 1.
    
    # Let's construct the set of primes >= 13 first.
    candidate_primes = [p for p in primes if p >= 13]
    
    # Verify cost
    total_cost = Fraction(0, 1)
    for p in candidate_primes:
        total_cost += Fraction(1, p)
        
    if total_cost > 1:
        # If it's over, remove the largest primes (smallest cost) until it fits?
        # Or just use a smaller set. But primes >= 13 should fit.
        # Let's assume it fits for now, as 1/13+... is approx 0.94.
        pass

    # We will perform local search on the residues of these primes.
    # Initial assignment: random or greedy.
    # Objective: Maximize L such that 1..L is covered.
    
    # To speed up coverage check, we can use a bitset or boolean array.
    # Max L to check: 10000 is a good upper bound for search, final answer capped at 10^5 but we won't reach that with this set.
    # Actually, the task says L >= 999 is the goal. Let's aim for that.
    MAX_CHECK = 20000
    
    def evaluate_coverage(residues):
        # residues: dict p -> r
        covered = [False] * (MAX_CHECK + 1)
        for p, r in residues.items():
            # Mark n where n % p == r
            # Start from the first n >= 1 such that n % p == r
            first = r if r >= 1 else p
            # If r == 0, first is p (since n>=1, 0 is not in 1..L)
            # Wait, if r=0, n=p, 2p...
            # If r>0, n=r, r+p...
            if r == 0:
                first = p
            else:
                first = r
            
            for n in range(first, MAX_CHECK + 1, p):
                covered[n] = True
        
        # Find L
        L = 0
        for i in range(1, MAX_CHECK + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    # Initialize residues randomly
    residues = {p: random.randint(0, p-1) for p in candidate_primes}
    current_L = evaluate_coverage(residues)
    best_L = current_L
    best_residues = residues.copy()
    
    # Local Search
    # Try changing one residue at a time to improve L.
    # If L doesn't improve, maybe try a few random swaps to escape local minima.
    
    num_iterations = 0
    max_iterations = 5000 # Limit iterations to ensure time safety
    
    while time.time() < deadline and num_iterations < max_iterations:
        num_iterations += 1
        
        # Pick a random prime to change
        p = random.choice(candidate_primes)
        old_r = residues[p]
        new_r = random.randint(0, p-1)
        if new_r == old_r:
            continue
            
        residues[p] = new_r
        new_L = evaluate_coverage(residues)
        
        if new_L > current_L:
            current_L = new_L
            best_L = new_L
            best_residues = residues.copy()
            # Reset iteration counter? No, just continue.
        else:
            # Revert if no improvement (or accept with small probability for simulated annealing)
            # For now, simple hill climbing
            residues[p] = old_r
            
        # Check time
        if time.time() > deadline - 5:
            break

    # Convert best_residues to list of lists
    best_classes = [[p, r] for p, r in best_residues.items()]
    
    return best_classes
