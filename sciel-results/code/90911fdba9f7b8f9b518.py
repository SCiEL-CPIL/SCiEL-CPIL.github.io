"""Improved program: Greedy with local search for prime residue covering."""
import sys
from fractions import Fraction
from math import isqrt
import time

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES_1000 = _primes_up_to(1000)
# Map prime to index for quick lookup
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES_1000)}

def _calculate_coverage_length(solution, L_max=100000):
    """Calculate the length L such that 1..L is covered."""
    covered = bytearray(L_max + 1)
    for p, r in solution:
        start = r if r > 0 else p
        if start < 1:
            start = 1
        for m in range(start, L_max + 1, p):
            covered[m] = 1
    
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def construct():
    """
    Find a set of distinct primes p <= 1000 with residues r (0 <= r < p)
    such that sum(1/p) <= 1 and every integer 1..L is covered.
    Maximize L.
    """
    L_max = 100000
    budget = Fraction(1, 1)
    
    # 1. Greedy Construction
    # We prioritize primes that cover the most new numbers in a local window.
    # We iterate n from 1 to L_max. If n is uncovered, we pick a prime.
    
    covered = bytearray(L_max + 1)
    solution = []
    used_primes = set()
    remaining_budget = Fraction(1, 1)
    
    # Sort primes by ascending size to consider smaller primes first (higher density)
    # But we must check budget.
    # Let's try a strategy: for each uncovered n, try all unused primes p.
    # Calculate gain: number of newly covered integers in [n, n+50].
    # Pick p with max gain / (1/p) = gain * p, subject to budget.
    
    primes_asc = sorted(PRIMES_1000)
    
    for n in range(1, L_max + 1):
        if covered[n]:
            continue
            
        best_p = None
        best_score = -1
        best_r = None
        
        # Try all unused primes
        for p in primes_asc:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if remaining_budget < cost:
                continue
                
            r = n % p
            
            # Estimate gain in window [n, n+100]
            # This is expensive if done for every p. 
            # Optimization: Only check a subset or approximate.
            # For now, let's just pick the smallest p that fits budget.
            # Small p covers more numbers in the window.
            
            # Heuristic: Smaller p is better for coverage density.
            score = p # Higher score for smaller p? No, we want to maximize coverage.
            # Actually, let's just pick the smallest p that fits.
            # This uses budget faster but covers more ground.
            
            if best_p is None or p < best_p:
                best_p = p
                best_r = r
                best_score = p # Just tracking
                
        if best_p is None:
            break
            
        # Add best_p
        used_primes.add(best_p)
        solution.append([best_p, best_r])
        
        # Update covered
        p = best_p
        r = best_r
        start = r if r > 0 else p
        if start < 1:
            start = 1
        for m in range(start, L_max + 1, p):
            covered[m] = 1
            
        remaining_budget -= Fraction(1, best_p)
        
        # If budget is very low, we might stop early if we can't cover next gap?
        # No, just continue.
        
    # 2. Local Search Optimization
    # Try to improve L by changing residues of existing primes.
    # Iterate:
    #   For each prime in solution:
    #     Try all residues r' in 0..p-1
    #     If changing r to r' increases L, accept it.
    # Repeat until no improvement.
    
    current_L = _calculate_coverage_length(solution, L_max)
    improved = True
    iterations = 0
    max_iterations = 50 # Limit iterations to avoid timeout
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(len(solution)):
            p, r_old = solution[i]
            
            # Try all other residues
            # Optimization: Only try residues that might cover the first gap?
            # For now, try all if p is small, sample if p is large.
            # Since p <= 1000, trying all p residues is O(p).
            # Total work per pass: sum(p for p in solution).
            # With ~20-50 primes, this is manageable.
            
            best_new_r = r_old
            best_new_L = current_L
            
            # To speed up, we can precompute coverage without this prime, then add each r.
            # But that's O(L_max * num_primes) per residue change. Too slow.
            
            # Instead, we can just calculate L for each candidate r.
            # _calculate_coverage_length is O(L_max * num_primes).
            # If we do this for every residue of every prime, it's O(num_primes^2 * L_max).
            # 168^2 * 100000 = 2.8 * 10^9. Too slow.
            
            # Better: Incremental update.
            # Remove prime i, calculate base coverage.
            # Then for each candidate r, add prime i with r, calculate L.
            # This is O(L_max) per candidate r.
            # Total per prime i: O(p_i * L_max).
            # Total per pass: O(L_max * sum(p_i)).
            # sum(p_i) is roughly sum of selected primes. If we select ~30 primes, avg p=200, sum=6000.
            # 100000 * 6000 = 6 * 10^8. Still too slow for Python in 120s if repeated.
            
            # Let's limit local search to only try residues that cover the first gap.
            # Find first gap g.
            # Only try residues r such that g % p == r.
            # This reduces candidates to 1 per prime!
            
            # Find first gap in current coverage
            first_gap = -1
            # We need to recompute coverage to find first gap efficiently?
            # We have `covered` array from greedy. But it's stale after residue changes.
            # We must recompute coverage for each candidate.
            
            # Let's simplify: Only do 1-2 iterations of local search with full recomputation.
            # Or, just skip local search if it's too complex to implement efficiently.
            
            # Given the time constraint, let's try a simpler local search:
            # Only try to fix the first gap.
            # Find first gap g in current solution.
            # For each prime p in solution, if we can set r = g % p, does it help?
            # Changing r to g % p ensures g is covered.
            # Does it break coverage of previous numbers?
            # It might.
            
            # Let's just do a few iterations of "try to cover first gap by changing one prime's residue".
            
            # Recompute coverage to find first gap
            cov_check = bytearray(L_max + 1)
            for pp, rr in solution:
                st = rr if rr > 0 else pp
                if st < 1: st = 1
                for mm in range(st, L_max + 1, pp):
                    cov_check[mm] = 1
            
            for gg in range(1, L_max + 1):
                if not cov_check[gg]:
                    first_gap = gg
                    break
            
            if first_gap == -1:
                break # Fully covered up to L_max? Unlikely.
                
            # Try to cover first_gap by changing residue of one prime
            best_candidate = None
            best_candidate_L = current_L
            
            for i in range(len(solution)):
                p, r_old = solution[i]
                r_new = first_gap % p
                if r_new == r_old:
                    continue
                    
                # Test this change
                test_solution = solution.copy()
                test_solution[i] = [p, r_new]
                new_L = _calculate_coverage_length(test_solution, L_max)
                
                if new_L > best_candidate_L:
                    best_candidate_L = new_L
                    best_candidate = (i, p, r_new)
            
            if best_candidate:
                i, p, r_new = best_candidate
                solution[i] = [p, r_new]
                current_L = best_candidate_L
                improved = True
                break # Restart scan with new L
    
    return solution

# Test
if __name__ == "__main__":
    sol = construct()
    print(f"Number of classes: {len(sol)}")
    L = _calculate_coverage_length(sol)
    print(f"Covered length L: {L}")
    total = sum(Fraction(1, p) for p, r in sol)
    print(f"Total budget used: {total}")
