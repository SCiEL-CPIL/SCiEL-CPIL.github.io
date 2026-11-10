"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def _sieve_primes(limit):
    """Generate all primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Primes up to 1000
    primes = _sieve_primes(1000)
    
    # We want to maximize L. 
    # The seed program achieves ~999.
    # Let's try to optimize coverage for a target range.
    # Since the budget is sum(1/p) <= 1, and sum_{p<=1000} 1/p is large (~3), 
    # we are constrained.
    # However, we don't need to cover 1..1000 necessarily, just maximize L.
    # The seed covers 1..~999? 
    # Seed: primes starting from 13. 
    # p_0=13 covers 1. p_1=17 covers 2. ...
    # This is a very sparse set of primes.
    
    # Strategy: Greedy Set Cover on a window [1, N]
    # We need to pick classes (p, r) such that sum(1/p) <= 1.
    # To maximize L, we should focus on covering small integers densely.
    
    # Let's estimate the maximum possible L.
    # If we use small primes, 1/p is large, so we can't use many.
    # If we use large primes, 1/p is small, so we can use many, but they cover sparsely.
    
    # A better heuristic: 
    # 1. Determine a set of candidate primes.
    # 2. For each prime, the best residue r is the one that covers the most *uncovered* numbers in 1..N.
    
    # Let's try to cover 1..5000 first to see if we can beat 999.
    # Actually, the seed program's logic:
    # primes = [13, 17, 19, 23, ...]
    # i=0, p=13, r=1. Covers 1, 14, 27...
    # i=1, p=17, r=2. Covers 2, 19, 36...
    # This ensures 1, 2, 3... are covered.
    # The sum of 1/p for primes 13..~1000 is approx 0.92.
    # So there is budget left.
    
    # Can we do better?
    # The Erdos-Ruzsa conjecture states that for any C, there exists X such that
    # we can cover [1, X] with sum 1/p <= C.
    # For C=1, X=1000 is a specific instance.
    # The seed claims L >= 999.
    
    # Let's try a greedy approach to fill gaps or extend L.
    # Target N = 2000.
    N = 2000
    
    # Budget is 1.0
    budget = Fraction(1, 1)
    
    # Candidate primes: All primes <= 1000.
    # But small primes are expensive.
    # Let's sort primes by 1/p (descending) to prioritize "value" per cost? 
    # No, in set cover, we want max coverage.
    
    # Let's maintain an array `covered` of booleans for 1..N.
    covered = [False] * (N + 1)
    
    # We will select classes one by one.
    # At each step, for each unused prime p, find the best residue r.
    # Score of (p, r) = number of n in 1..N such that n%p == r and not covered[n].
    # Cost = 1/p.
    # We must ensure sum(cost) <= 1.
    
    # This is a knapsack-like set cover.
    # Greedy: Pick the class with max score / cost? Or just max score?
    # Since budget is tight, score/cost is a better heuristic (density).
    
    # However, we must be careful: using a small prime (high cost) might block
    # using many large primes.
    
    # Let's try a simple greedy:
    # 1. Start with empty set.
    # 2. While time remains:
    #    a. For each prime p not yet used, calculate best_r and best_score.
    #    b. Find the pair (p, r) with maximum best_score / (1/p) = p * best_score.
    #    c. If adding this pair keeps budget <= 1, add it.
    #    d. Update covered array.
    #    e. If no pair can be added, break.
    
    # Optimization: Precompute residues?
    # N is small (2000), primes ~168.
    # 168 * 2000 operations per step is fine.
    
    used_primes = set()
    classes = []
    
    # To speed up, we can precompute for each prime p, the counts of residues in 1..N.
    # But we need *uncovered* counts.
    
    while time.time() < deadline:
        best_p = None
        best_r = None
        best_val = -1
        best_cost = None
        
        # Iterate through available primes
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if budget < cost:
                continue
            
            # Find best residue for this p
            max_gain = 0
            best_res = 0
            # We can iterate r from 0 to p-1, but p can be up to 1000.
            # N is 2000. So most residues will have 0 or 1 number in 1..N.
            # Actually, for p > N, each residue covers at most 1 number.
            # For p <= N, residues cover N/p numbers.
            
            # To find max_gain, we need to check which residues have uncovered numbers.
            # Instead of iterating all r, we can just check the uncovered numbers.
            # But that's O(N).
            # Iterating r is O(p).
            # Since sum p is small, and we do this for each p, it's O(P * N) roughly.
            
            # Let's just iterate r for simplicity, N is small.
            # Actually, we can just check the first occurrence of each residue?
            # No, we need the *count* of uncovered numbers.
            
            # Efficient way:
            # Maintain a list of uncovered numbers?
            # Or just loop n in 1..N?
            
            # Let's try looping r.
            for r in range(p):
                gain = 0
                # Count uncovered n in 1..N with n % p == r
                # n = k*p + r
                # Start n = r if r > 0 else p (since n>=1)
                start = r if r > 0 else p
                for n in range(start, N + 1, p):
                    if not covered[n]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_res = r
            
            if max_gain == 0:
                continue
            
            # Value = gain / cost = gain * p
            val = max_gain * p
            
            if val > best_val:
                best_val = val
                best_p = p
                best_r = best_res
                best_cost = cost
        
        if best_p is None:
            break
        
        # Add the class
        used_primes.add(best_p)
        classes.append([best_p, best_r])
        budget -= best_cost
        
        # Update covered array
        for n in range(best_r if best_r > 0 else best_p, N + 1, best_p):
            covered[n] = True
        
        # Optional: Check if we've covered 1..N fully?
        # If yes, we can try to extend N? 
        # The task asks for max L. 
        # If we cover 1..2000, L >= 2000.
        # We can dynamically increase N if we have time and budget.
        
        # Let's check current max L
        current_L = 0
        for i in range(1, N + 1):
            if covered[i]:
                current_L = i
            else:
                break
        
        # If we covered the whole window, we could try to expand N.
        # But expanding N requires re-evaluating gains.
        # Given time constraints, let's stick to a fixed N for now, 
        # or increase N slowly.
        
        # Let's increase N if we are doing well.
        if current_L == N and N < 5000:
            N += 100
            covered.extend([False] * 100)
            # Note: This invalidates previous gain calculations, but greedy is adaptive.
            
    # Final score calculation
    # Find largest L such that 1..L are covered.
    # We used a window N. If we covered 1..N, L might be larger.
    # We need to check beyond N if we have time.
    
    # Let's verify coverage up to a limit, say 10000, to be safe.
    # But we only updated covered up to N.
    # We should re-check coverage for the final classes up to a higher limit.
    
    final_limit = 10000
    final_covered = [False] * (final_limit + 1)
    for p, r in classes:
        start = r if r > 0 else p
        for n in range(start, final_limit + 1, p):
            final_covered[n] = True
            
    L = 0
    for i in range(1, final_limit + 1):
        if final_covered[i]:
            L = i
        else:
            break
            
    return classes

# EVOLVE-BLOCK-END
