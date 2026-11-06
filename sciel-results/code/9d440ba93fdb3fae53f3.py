"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    p_set = set(primes)
    
    # We want to cover 1..L. Let's aim for L=1000 initially, but we can extend if budget allows.
    # The reciprocal budget is 1.
    # Greedy Set Cover:
    # 1. Define the universe U = {1, 2, ..., L_max}. Let L_max = 10000 for safety, but we really care about 1..1000.
    #    Actually, the score is the largest L such that 1..L is covered.
    #    If we cover 1..1000, score is at least 1000.
    #    We can try to cover beyond 1000 if we have budget left, but the primes are <= 1000.
    #    With primes <= 1000, the "density" of coverage is limited.
    #    Let's focus on maximizing the contiguous prefix 1..L.
    
    # To make greedy efficient, we need to track which numbers are covered.
    # We will iterate until we run out of budget or time.
    # State: 
    #   covered: a boolean array or set of covered numbers.
    #   remaining_budget: float (sum of 1/p used)
    #   used_primes: set of primes already used.
    
    # Since we need exact rationals for the budget, we can track the sum of 1/p.
    # For the greedy selection, we need to find the prime p (not used) and residue r 
    # that maximizes the number of new integers in 1..L_max covered.
    
    # L_max for evaluation: The problem says capped at 10^5. 
    # But with primes <= 1000, can we cover 10^5?
    # Sum of reciprocals of primes up to 1000 is approx log(log(1000)) + M ~ 1.93.
    # Wait, sum_{p<=1000} 1/p is about 2.198. 
    # The budget is 1. So we can only use a subset of primes whose reciprocals sum to <= 1.
    # This is a significant constraint.
    
    # Let's determine the maximum L we can realistically cover.
    # If we pick primes to cover 1..L, each prime covers ~L/p numbers in 1..L.
    # Total coverage ~ sum (L/p) = L * sum(1/p).
    # To cover L numbers, we need L * sum(1/p) >= L, so sum(1/p) >= 1.
    # This suggests that with budget 1, we can barely cover a density of 1, i.e., all numbers, 
    # IF the classes were perfectly disjoint and efficient. But they are residues, so they overlap.
    # Actually, the expected number of covered integers in 1..L is L * sum(1/p).
    # If sum(1/p) = 1, we expect L integers to be covered.
    # To cover ALL integers 1..L, we need much more than sum(1/p)=1 on average due to overlaps (Poisson approximation: P(uncovered) = exp(-sum(1/p))).
    # If sum(1/p)=1, P(uncovered) = e^-1 ~ 0.37. So 37% of numbers would be uncovered on average.
    # This implies we CANNOT cover 1..1000 with sum(1/p)=1 using random residues.
    # However, we are optimizing residues. We can pack them to avoid overlaps.
    # This is a packing problem.
    
    # Let's try a greedy approach that explicitly tracks coverage of 1..L_target.
    # Let L_target = 1000.
    # We will maintain an array `covered` of size L_target+1 (1-indexed).
    # Initially all False.
    # At each step, for each unused prime p, and for each residue r in 0..p-1:
    #   Calculate how many new numbers in 1..L_target are covered by class (p, r).
    #   Choose the (p, r) that maximizes this count, subject to budget constraint.
    #   If max count is 0, stop.
    #   Add to solution, update covered array, subtract 1/p from budget.
    
    # Optimization:
    # Iterating all p (168 primes) and all r (up to 1000) is 168,000 candidates.
    # For each candidate, checking coverage in 1..L_target (1000 numbers) takes L_target/p steps.
    # Total work per iteration: sum_{p} (p * L_target/p) = sum_{p} L_target = 168 * 1000 = 168,000 operations.
    # Number of iterations: at most 168 primes.
    # Total operations: 168 * 168,000 ~ 28 million. This is very fast in Python if optimized, or even in pure Python.
    # 28 million simple operations might take a few seconds.
    
    L_target = 1000
    covered = [False] * (L_target + 1)
    used_primes = set()
    budget = 1.0
    selected_classes = []
    
    # Precompute reciprocals
    reciprocals = {p: 1.0/p for p in primes}
    
    # To speed up coverage check, we can just iterate through the numbers covered by (p,r) in 1..L_target.
    # Numbers are r, r+p, r+2p, ... <= L_target.
    # Note: r is 0 <= r < p. The numbers are r + k*p.
    # If r=0, numbers are p, 2p, ... (since 0 is not in 1..L_target).
    # If r>0, numbers are r, r+p, ...
    
    while True:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_p = None
        best_r = None
        
        for p in primes:
            if p in used_primes:
                continue
            if budget < reciprocals[p] - 1e-12: # Check budget
                continue
                
            # Try all residues
            # We can prune: if max possible gain (which is floor(L_target/p) or similar) is <= best_gain, skip?
            # Max gain for prime p is roughly L_target/p.
            # If L_target/p <= best_gain, we can't beat best_gain with this p.
            max_possible_gain = L_target // p + 1
            if max_possible_gain <= best_gain:
                continue
                
            for r in range(p):
                # Count new covers
                gain = 0
                # Start from r if r>0 else p
                start = r if r > 0 else p
                # Iterate
                # Optimization: if gain already exceeds best_gain, we can break early? No, we need exact gain to compare?
                # Actually, if gain > best_gain, we can update best and continue checking? 
                # No, we want the MAX gain. We must count all.
                # But we can break if gain + (remaining numbers) <= best_gain?
                
                # Let's just count.
                for x in range(start, L_target + 1, p):
                    if not covered[x]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
        
        if best_gain == 0:
            break
            
        # Select best_p, best_r
        used_primes.add(best_p)
        budget -= reciprocals[best_p]
        selected_classes.append([best_p, best_r])
        
        # Update covered array
        start = best_r if best_r > 0 else best_p
        for x in range(start, L_target + 1, best_p):
            covered[x] = True
            
        # If we have covered 1..L_target, we might try to extend?
        # The problem caps score at 10^5, but primes are <= 1000.
        # If 1..1000 is fully covered, L=1000 is achieved.
        # Can we go higher?
        # If we have budget left, we can add more primes.
        # But the greedy loop continues as long as best_gain > 0.
        # However, `covered` array only tracks 1..L_target.
        # If we want to go beyond L_target, we need to expand the array.
        # Given the budget constraint (sum 1/p <= 1), it is unlikely we can cover much beyond 1000 
        # because the density of coverage drops or overlaps increase.
        # Let's check if 1..1000 is fully covered.
        
        # Check if all 1..L_target are covered
        if all(covered[1:L_target+1]):
            # We achieved 1000.
            # Can we extend?
            # Let's try to extend L_target dynamically if we have budget and time.
            # But for now, let's just return the result.
            # The evaluator calculates L from the classes.
            # If we stop here, we might have budget left.
            # Let's continue the loop but expand the coverage array if needed?
            # It's complex. Let's stick to 1..1000 for the greedy target, 
            # but if we finish covering 1000, we can keep adding primes that cover new numbers 
            # in 1001..2000 etc?
            # The problem says "largest L such that every integer 1..L lies in some class".
            # If we cover 1..1000, L is at least 1000.
            # If we add a prime that covers 1001, does it help?
            # Only if 1001 is covered AND 1..1000 are covered.
            # The greedy algorithm optimizes for 1..1000.
            # Once 1..1000 is covered, the gain for any new prime in 1..1000 is 0.
            # So the loop will terminate (best_gain becomes 0).
            # To go beyond, we would need to change the objective function to include numbers > 1000.
            # Given the time limit and complexity, covering 1..1000 is the primary goal.
            # If we can't cover 1..1000, the loop stops when no new number in 1..1000 is covered.
            
            break
            
    # Calculate the actual L covered by the selected classes
    # We need to check 1..100000 potentially, but let's start with 1..10000
    # and extend if necessary.
    
    # Efficient check:
    # We have a list of classes [[p, r], ...].
    # An integer n is covered if exists (p, r) in classes such that n % p == r.
    # We want the largest L such that all 1..L are covered.
    
    # We can check incrementally.
    # Let's check up to 10000 first.
    L_max_check = 10000
    is_covered = [False] * (L_max_check + 1)
    
    for p, r in selected_classes:
        # Mark all numbers n in 1..L_max_check such that n % p == r
        start = r if r > 0 else p
        for n in range(start, L_max_check + 1, p):
            is_covered[n] = True
            
    # Find largest L
    L = 0
    for i in range(1, L_max_check + 1):
        if is_covered[i]:
            L = i
        else:
            break
            
    return selected_classes
# EVOLVE-BLOCK-END
