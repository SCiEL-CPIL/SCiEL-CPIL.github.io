"""
Strategy: Greedy Set Cover for Prime Residue Classes.

The current seed program uses a static, heuristic mapping (i-th prime covers i-th integer).
This approach treats the problem as a maximum coverage problem with a knapsack-like constraint
(sum of 1/p <= 1). We use a greedy heuristic:
1. Start with an empty set of classes and an uncovered set of integers (1..L_max).
2. While budget remains and uncovered set is not empty:
   a. For each available prime p (not yet used), find the residue r that covers the most
      currently uncovered integers.
   b. Select the (p, r) pair that gives the maximum marginal coverage.
   c. If the marginal coverage is 0, stop (no further improvement possible with greedy).
   d. Update uncovered set and budget.
3. Return the selected classes.

This is a structural change from static assignment to an adaptive, coverage-driven search.
"""

import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # We want to maximize L such that 1..L are covered.
    # Let's estimate a reasonable upper bound for L to search within.
    # The sum of reciprocals of primes up to 1000 is about 2.2.
    # With budget 1, we can use a subset.
    # The seed program covers up to roughly 1000.
    # Let's set L_max to 2000 to allow for potential extension, but keep it manageable.
    # Actually, the task says capped at 10^5, but 120s limit is tight.
    # Let's start with a conservative L_max = 2000. If we cover all, we can try higher.
    # But checking coverage for 10^5 integers every step is too slow.
    # We only care about the contiguous prefix starting from 1.
    
    # Represent uncovered numbers as a set for efficient updates?
    # Or a boolean array?
    # A boolean array of size L_max is faster for iteration.
    
    L_max = 2000
    covered = [False] * (L_max + 1)  # Index 0 is unused
    uncovered_count = L_max
    
    # Budget tracking
    current_budget = Fraction(1, 1)
    
    # List of selected classes
    selected_classes = []
    
    # Precompute residues for each prime? No, we need to find the best residue dynamically.
    # For each prime p, the best residue r is the one that maximizes the count of
    # uncovered numbers in {r, r+p, r+2p, ...} <= L_max.
    
    # To speed up, we can maintain a list of uncovered numbers.
    # However, finding the best r for a given p requires scanning residues.
    # There are p residues. For each residue, we check how many multiples are uncovered.
    # Total work per prime p is O(L_max/p * p) = O(L_max) if we do it naively?
    # No, for a fixed p, checking all p residues involves checking L_max/p elements per residue,
    # so total O(L_max) checks per prime.
    # With ~168 primes up to 1000, and L_max=2000, this is 168 * 2000 = 336,000 ops per iteration.
    # How many iterations? At most the number of primes we can afford.
    # Sum 1/p <= 1. The smallest primes are 2, 3, 5, ...
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31 ...
    # Approx 0.5+0.33+0.2 = 1.03. So we can pick about 8-10 small primes.
    # So ~10 iterations. 10 * 336,000 = 3.3M ops. This is very fast in Python.
    
    # Let's refine the "best residue" calculation.
    # For a given p, we want r in [0, p-1] that maximizes:
    # count = sum(1 for k in range(r, L_max+1, p) if not covered[k])
    
    # We can optimize this by precomputing counts for each residue class for each prime?
    # But the `covered` array changes every step.
    
    # Alternative: Maintain a list of uncovered numbers.
    # For a given p, we can iterate through the uncovered list and tally counts per residue.
    # This is O(U) where U is the number of uncovered numbers.
    # Initially U = L_max. As we cover more, U decreases.
    # This seems more efficient than scanning the full range for each residue.
    
    uncovered_set = set(range(1, L_max + 1))
    
    while current_budget > 0 and uncovered_set:
        best_p = None
        best_r = None
        best_gain = 0
        best_cost = None
        
        # Check time
        if time.time() > deadline - 5:
            break
            
        # Iterate through available primes
        # We should probably prioritize smaller primes as they have higher 1/p cost
        # but also potentially higher coverage density?
        # Actually, smaller p means fewer residues to check, but the numbers are denser.
        # Let's just iterate all available primes.
        
        available_primes = [p for p in primes if p not in {c[0] for c in selected_classes}]
        
        for p in available_primes:
            cost = Fraction(1, p)
            if current_budget < cost:
                continue
                
            # Find best residue r for this p
            # Count uncovered numbers for each residue r in 0..p-1
            counts = [0] * p
            # Iterate through uncovered numbers and increment count for n % p
            # This is O(U).
            for n in uncovered_set:
                counts[n % p] += 1
            
            # Find max count
            max_count = max(counts)
            if max_count > best_gain:
                best_gain = max_count
                best_p = p
                best_r = counts.index(max_count)
                best_cost = cost
        
        if best_p is None or best_gain == 0:
            break
            
        # Apply selection
        selected_classes.append([best_p, best_r])
        current_budget -= best_cost
        
        # Update uncovered set
        p = best_p
        r = best_r
        # Remove all n in uncovered_set such that n % p == r
        # We can do this by creating a new set
        uncovered_set = {n for n in uncovered_set if n % p != r}
        
    # Calculate the score: largest L such that 1..L are covered.
    # Since we tracked uncovered_set, we can find the smallest missing number.
    # However, we only tracked up to L_max.
    # If uncovered_set is empty, we covered 1..L_max.
    # We might be able to cover more?
    # The problem caps at 10^5, but our search space was limited to L_max.
    # If we covered all 1..L_max, we should check if we can extend.
    # But the greedy selection was based on coverage within 1..L_max.
    # If we extended L_max, we might have made different choices.
    # For now, let's assume L_max is large enough.
    # If uncovered_set is empty, we can try to extend the range.
    
    if not uncovered_set:
        # We covered 1..L_max. Can we go higher?
        # We have remaining budget?
        # Let's try to extend L_max by checking if we can cover L_max+1, etc.
        # But the residues were chosen for 1..L_max.
        # It's possible that L_max+1 is already covered by some class.
        # Let's just check how far the contiguous coverage goes.
        
        # We don't have the full coverage array for > L_max.
        # We can simulate coverage for n > L_max using the selected classes.
        # Let's extend the check up to, say, L_max * 2.
        
        extended_L_max = L_max * 2
        # Check coverage for 1..extended_L_max
        # We already know 1..L_max are covered.
        # Check L_max+1 to extended_L_max.
        
        for n in range(L_max + 1, extended_L_max + 1):
            is_covered = False
            for p, r in selected_classes:
                if n % p == r:
                    is_covered = True
                    break
            if not is_covered:
                L_final = n - 1
                break
        else:
            L_final = extended_L_max
            
        # If we hit the cap, return cap? No, return the actual L.
        # The task says capped at 10^5.
        if L_final >= 100000:
            L_final = 100000
    else:
        # Find the smallest number in uncovered_set
        L_final = min(uncovered_set) - 1
        
    return selected_classes

# Test locally
if __name__ == "__main__":
    classes = construct()
    # Verify coverage
    L = 0
    for i in range(1, 10000):
        covered = False
        for p, r in classes:
            if i % p == r:
                covered = True
                break
        if covered:
            L = i
        else:
            break
    print(f"Covered up to L={L}")
    print(f"Classes: {classes}")
    budget = sum(Fraction(1, p) for p, r in classes)
    print(f"Budget: {budget}")
