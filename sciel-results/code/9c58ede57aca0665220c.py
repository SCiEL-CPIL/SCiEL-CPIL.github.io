"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import sys

def construct():
    sys.setrecursionlimit(100000)
    
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
    
    # Precompute coverage masks for all primes and residues
    # mask[p][r] = bitmask of integers in [1, L_target] congruent to r mod p
    # Use a dictionary since not all primes need all residues precomputed
    coverage_cache = {}
    
    def get_coverage(p, r):
        """Get or compute the coverage mask for prime p, residue r."""
        key = (p, r)
        if key in coverage_cache:
            return coverage_cache[key]
        
        bits = 0
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_target:
            for x in range(start, L_target + 1, p):
                bits |= (1 << (x - 1))
        coverage_cache[key] = bits
        return bits
    
    # Precompute all coverage masks for small primes
    for p in primes:
        if p > 100:  # Only precompute for smaller primes to save memory/time
            break
        for r in range(p):
            get_coverage(p, r)
    
    # Constraint satisfaction with backtracking and propagation
    # Strategy:
    # 1. Select a set of candidate primes whose reciprocal sum is <= 1
    # 2. Use backtracking to assign residues
    # 3. Strong pruning: at each step, check if remaining primes can cover all uncovered integers
    
    # First, find good candidate prime sets
    # We want primes that together can cover [1, 1000]
    
    # Heuristic: small primes are more valuable (cover more per unit budget)
    # But we need enough primes to cover all residues
    
    # Let's try multiple candidate sets and pick the best one
    
    def try_csp(candidate_primes, max_depth=500000):
        """
        Try to find residues for candidate_primes that cover [1, L_target].
        Returns list of [p, r] if successful, None otherwise.
        """
        total_cost = sum(Fraction(1, p) for p in candidate_primes)
        if total_cost > 1:
            return None
        
        n_primes = len(candidate_primes)
        uncovered = (1 << L_target) - 1  # All integers 1..L_target uncovered
        
        # Sort primes: process smaller primes first (they have more constraints)
        sorted_primes = sorted(candidate_primes)
        
        # Precompute for each position, which residues are feasible
        # and estimate coverage potential
        
        def max_coverable(uncovered_mask, remaining_primes):
            """Upper bound on how many uncovered integers remaining_primes can cover."""
            # Each prime p can cover at most L_target // p + 1 integers per residue
            # But they might overlap, so this is an upper bound
            total = 0
            for p in remaining_primes:
                # Max coverage per residue for prime p
                max_per_residue = (L_target + p - 1) // p
                total += max_per_residue
            return min(total, bin(uncovered_mask).count('1'))
        
        def backtrack(idx, uncovered_mask, choices, node_count):
            if node_count[0] >= max_depth:
                return None
            
            node_count[0] += 1
            
            if idx == n_primes:
                if uncovered_mask == 0:
                    return choices[:]
                return None
            
            p = sorted_primes[idx]
            remaining_primes = sorted_primes[idx + 1:]
            
            # Get all residues for this prime
            # Heuristic: try residues that cover the most uncovered integers first
            residue_coverages = []
            for r in range(p):
                bits = get_coverage(p, r)
                new_cov = bits & uncovered_mask
                count = bin(new_cov).count('1')
                residue_coverages.append((count, r, new_cov))
            
            # Sort by coverage (descending)
            residue_coverages.sort(key=lambda x: -x[0])
            
            # Pruning: if even the best residue can't help enough, skip
            # Also check if remaining primes can cover what's left after best choice
            
            for count, r, new_cov in residue_coverages:
                if count == 0:
                    # No new coverage from this residue
                    # Still might be needed if all residues give 0, but that's bad
                    pass
                
                new_uncovered = uncovered_mask & ~new_cov
                
                # Pruning: check if remaining primes can cover new_uncovered
                if remaining_primes:
                    # Upper bound check
                    unc_count = bin(new_uncovered).count('1')
                    # Each remaining prime p can cover at most (L_target + p - 1) // p integers
                    # But more precisely, the number of integers in [1, L_target] covered by
                    # any single residue of prime p is at most ceil(L_target / p)
                    max_possible = 0
                    for rp in remaining_primes:
                        max_possible += (L_target + rp - 1) // rp
                    
                    if max_possible < unc_count:
                        continue
                
                # Additional pruning: if new_uncovered has an integer that no remaining
                # prime can cover, prune
                # An integer x can be covered by prime p iff there exists r such that x ≡ r (mod p)
                # This is always true for any x and p, so this doesn't help directly
                # But we can check: for each uncovered x, is there a remaining prime p
                # such that x mod p is a valid residue? Yes, always.
                # So this pruning doesn't apply.
                
                # Try this assignment
                result = backtrack(idx + 1, new_uncovered, choices + [[p, r]], node_count)
                if result is not None:
                    return result
            
            return None
        
        node_count = [0]
        result = backtrack(0, uncovered, [], node_count)
        return result
    
    # Try different candidate prime sets
    # Start with sets that are known to have sum <= 1
    
    # Strategy: try to find a set of primes that can cover [1, 1000]
    # We'll try several candidate sets
    
    best_result = None
    best_L = 0
    
    # Candidate sets: we want primes with sum of reciprocals <= 1
    # that can potentially cover [1, 1000]
    
    # Set 1: [3, 5, 7, 11, 13, 17, 19, 29] - sum ≈ 0.9896
    candidate_sets = [
        [3, 5, 7, 11, 13, 17, 19, 29],
        [3, 5, 7, 11, 13, 17, 19, 31],
        [3, 5, 7, 11, 13, 17, 23, 29],
        [3, 5, 7, 11, 13, 19, 23, 29],
        [3, 5, 7, 11, 13, 17, 19, 37],
        [3, 5, 7, 11, 13, 17, 23, 31],
        [3, 5, 7, 11, 13, 19, 23, 31],
        [3, 5, 7, 11, 13, 17, 19, 41],
        [3, 5, 7, 11, 13, 17, 23, 37],
        [3, 5, 7, 11, 13, 19, 23, 37],
    ]
    
    # Also try larger sets with more primes (smaller individual primes)
    # 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 ≈ 1.033 > 1
    # So we need to drop one. Let's try dropping the largest contribution among the smaller ones.
    # Actually, let's just try various combinations.
    
    for cset in candidate_sets:
        cost = sum(Fraction(1, p) for p in cset)
        if cost > 1:
            continue
        
        result = try_csp(cset, max_depth=200000)
        if result is not None:
            # Check what L this covers
            covered = 0
            for p, r in result:
                covered |= get_coverage(p, r)
            
            # Find largest L such that 1..L all covered
            L = 0
            for i in range(L_target):
                if covered & (1 << i):
                    L = i + 1
                else:
                    break
            
            if L > best_L:
                best_L = L
                best_result = result
            
            if best_L >= 999:
                break
    
    # If CSP didn't find a good solution, try a greedy fallback
    if best_result is None or best_L < 100:
        # Greedy approach as fallback
        chosen = []
        used_primes = set()
        uncovered = (1 << L_target) - 1
        budget = Fraction(1, 1)
        
        for _ in range(50):
            best_p = None
            best_r = None
            best_new = 0
            best_bits = 0
            
            remaining = budget - sum(Fraction(1, p) for p, _ in chosen)
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if cost > remaining:
                    continue
                
                for r in range(p):
                    bits = get_coverage(p, r)
                    new_cov = bits & uncovered
                    count = bin(new_cov).count('1')
                    if count > best_new:
                        best_new = count
                        best_p = p
                        best_r = r
                        best_bits = new_cov
            
            if best_p is None or best_new == 0:
                break
            
            chosen.append([best_p, best_r])
            used_primes.add(best_p)
            uncovered &= ~best_bits
        
        # Check L
        covered = (1 << L_target) - 1 & ~uncovered
        L = 0
        for i in range(L_target):
            if covered & (1 << i):
                L = i + 1
            else:
                break
        
        if L > best_L:
            best_L = L
            best_result = chosen
    
    # Verify and return
    if best_result is None:
        # Absolute fallback: just use prime 3
        best_result = [[3, 0]]
    
    final_cost = sum(Fraction(1, p) for p, _ in best_result)
    if final_cost > 1:
        # Trim to fit budget
        best_result = []
        used = set()
        total = Fraction(0)
        for p, r in best_result:
            if p in used:
                continue
            if total + Fraction(1, p) <= 1:
                best_result.append([p, r])
                used.add(p)
                total += Fraction(1, p)
    
    # Ensure distinct primes
    seen = set()
    final = []
    for p, r in best_result:
        if p not in seen:
            final.append([p, r])
            seen.add(p)
    
    # Verify constraints
    final_cost = sum(Fraction(1, p) for p, _ in final)
    assert final_cost <= 1, f"Final cost {final_cost} exceeds 1"
    for p, r in final:
        assert p <= 1000 and 0 <= r < p, f"Invalid class [{p}, {r}]"
    
    return final
# EVOLVE-BLOCK-END
