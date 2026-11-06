"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction

def construct():
    # Generate all primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    L_target = 1000
    
    # Precompute coverage bitmasks for all (p, r) pairs
    # covered_mask[p][r] = bitmask of integers in [1, L_target] congruent to r mod p
    covered_mask = {}
    for p in primes:
        covered_mask[p] = {}
        for r in range(p):
            bits = 0
            if r == 0:
                start = p
            else:
                start = r
            if start <= L_target:
                for x in range(start, L_target + 1, p):
                    bits |= (1 << (x - 1))
            covered_mask[p][r] = bits
    
    # Full mask for [1, L_target]
    full_mask = (1 << L_target) - 1
    
    # Strategy: Backtracking with propagation
    # We'll try to find a set of (p, r) pairs that cover [1, L_target]
    # with sum(1/p) <= 1
    
    # First, let's determine which primes are "affordable" given the budget
    # We need to select a subset of primes such that sum(1/p) <= 1
    
    # Sort primes by size (smaller primes cover more per residue)
    # But we need to be strategic about which primes to include
    
    # Key insight: We want to maximize coverage of the prefix [1, L]
    # Let's use a backtracking search where at each step we decide:
    # 1. Which prime to assign next (from remaining affordable primes)
    # 2. What residue to assign to that prime
    
    # To make this efficient, we'll use the following approach:
    # - Start with all primes sorted by 1/p (largest first, i.e., smallest p first)
    # - Use a recursive backtracking that tries to cover the uncovered integers
    # - Prune branches where the remaining budget can't cover the remaining gaps
    
    # Let's first find a good set of primes to use
    # We'll try different subsets and see which gives the best coverage
    
    # Approach: Use a greedy selection of primes, then backtrack on residues
    
    # Step 1: Select a set of primes with sum(1/p) <= 1
    # We'll try to include small primes (which cover more integers)
    
    # Let's try a systematic approach:
    # Start with the smallest primes and add them until budget is exceeded
    # Then try different combinations
    
    # Actually, let's use a different strategy:
    # 1. Start with L_target = 1000
    # 2. Use a backtracking search that:
    #    - Maintains the set of uncovered integers
    #    - At each step, selects a prime and residue that covers the most uncovered integers
    #    - Uses pruning based on budget feasibility
    
    # Let's implement a recursive backtracking with memoization
    
    import sys
    sys.setrecursionlimit(100000)
    
    # Precompute: for each prime p, the maximum number of integers it can cover
    # (this is floor(L_target / p) or ceil(L_target / p) depending on residue)
    
    # Let's use a iterative deepening approach:
    # Try to cover [1, L_target] using at most k primes, for increasing k
    
    # Actually, let's use a smarter approach:
    # 1. Determine a feasible set of primes (sum 1/p <= 1)
    # 2. For that set, find residues that cover [1, L_target]
    
    # Let's try several candidate prime sets and pick the best one
    
    def try_prime_set(candidate_primes):
        """Try to find residues for the given primes that cover [1, L_target]."""
        cost = sum(Fraction(1, p) for p in candidate_primes)
        if cost > 1:
            return None
        
        # Backtracking search for residues
        # Order primes by size (smaller first) for better pruning
        sorted_primes = sorted(candidate_primes)
        
        best_solution = None
        best_L = 0
        
        def backtrack(idx, uncovered, choices):
            nonlocal best_solution, best_L
            
            if idx == len(sorted_primes):
                # All primes assigned, check coverage
                covered = full_mask & ~uncovered
                # Find the largest L such that [1, L] is covered
                L = 0
                for i in range(L_target):
                    if covered & (1 << i):
                        L = i + 1
                    else:
                        break
                if L > best_L:
                    best_L = L
                    best_solution = choices[:]
                return
            
            p = sorted_primes[idx]
            
            # Try all residues, sorted by coverage of uncovered integers
            residues = list(range(p))
            residues.sort(key=lambda r: -bin(covered_mask[p][r] & uncovered).count('1'))
            
            for r in residues:
                bits = covered_mask[p][r]
                new_uncovered = uncovered & ~bits
                
                # Pruning: if the number of uncovered integers is too large
                # for the remaining primes to cover, skip
                remaining_primes = sorted_primes[idx + 1:]
                if remaining_primes:
                    max_coverage = sum(L_target // p for p in remaining_primes)
                    uncovered_count = bin(new_uncovered).count('1')
                    if uncovered_count > max_coverage:
                        continue
                
                backtrack(idx + 1, new_uncovered, choices + [[p, r]])
        
        backtrack(0, full_mask, [])
        
        return best_solution
    
    # Now let's try different prime sets
    # Start with small primes and gradually add more
    
    # First, let's try the set that was identified in the original code
    candidate_sets = [
        [3, 5, 7, 11, 13, 17, 19, 29],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [3, 5, 7, 11, 13, 17, 19, 23],
        [2, 5, 7, 11, 13, 17, 19, 23],
        [3, 5, 7, 11, 13, 17, 19, 23, 29],
        [2, 3, 5, 7, 11, 13, 17, 19],
    ]
    
    # Filter out sets that exceed budget
    valid_sets = []
    for s in candidate_sets:
        cost = sum(Fraction(1, p) for p in s)
        if cost <= 1:
            valid_sets.append(s)
    
    # Also try to generate more candidate sets
    # Start with small primes and add more until budget is exceeded
    small_primes = [p for p in primes if p <= 100]
    for i in range(1, len(small_primes)):
        for j in range(i, len(small_primes)):
            s = small_primes[i:j+1]
            cost = sum(Fraction(1, p) for p in s)
            if cost <= 1:
                if s not in valid_sets:
                    valid_sets.append(s)
    
    # Try each valid set and keep the best solution
    best_overall = None
    best_L_overall = 0
    
    for s in valid_sets:
        solution = try_prime_set(s)
        if solution:
            # Calculate the actual L covered
            covered = 0
            for p, r in solution:
                covered |= covered_mask[p][r]
            L = 0
            for i in range(L_target):
                if covered & (1 << i):
                    L = i + 1
                else:
                    break
            if L > best_L_overall:
                best_L_overall = L
                best_overall = solution
    
    if best_overall is None:
        # Fallback: use a simple greedy approach
        chosen = []
        uncovered = full_mask
        remaining_budget = Fraction(1, 1)
        
        for p in primes:
            if Fraction(1, p) > remaining_budget:
                continue
            
            best_r = None
            best_new = 0
            for r in range(p):
                new_cov = covered_mask[p][r] & uncovered
                count = bin(new_cov).count('1')
                if count > best_new:
                    best_new = count
                    best_r = r
            
            if best_r is not None and best_new > 0:
                chosen.append([p, best_r])
                uncovered &= ~covered_mask[p][best_r]
                remaining_budget -= Fraction(1, p)
        
        best_overall = chosen
        # Calculate L
        covered = 0
        for p, r in chosen:
            covered |= covered_mask[p][r]
        best_L_overall = 0
        for i in range(L_target):
            if covered & (1 << i):
                best_L_overall = i + 1
            else:
                break
    
    # Verify the result
    final_cost = sum(Fraction(1, p) for p, _ in best_overall)
    assert final_cost <= 1, f"Final cost {final_cost} exceeds 1"
    for p, r in best_overall:
        assert p <= 1000 and 0 <= r < p, f"Invalid class [{p}, {r}]"
    
    # Check that all primes are distinct
    prime_list = [p for p, _ in best_overall]
    assert len(prime_list) == len(set(prime_list)), "Duplicate primes"
    
    return best_overall
# EVOLVE-BLOCK-END
