import math
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    MAX_M = 128
    candidates = list(range(2, MAX_M + 1))
    
    # Precompute gcd pairs
    def can_assign(moduli, residues, new_m):
        """Check if there exists a residue r in [0, new_m) such that for all existing (r_i, m_i),
        gcd(m_i, new_m) does NOT divide (r_i - r)."""
        m = new_m
        # For each existing modulus m_i, we need: gcd(m_i, m) ∤ (r_i - r)
        # i.e., r mod gcd(m_i, m) != r_i mod gcd(m_i, m)
        # This is a system of congruence-exclusion constraints.
        # We need r such that for each i: r != r_i (mod g_i) where g_i = gcd(m_i, m)
        
        # Collect constraints: r mod g_i != r_i mod g_i
        constraints = []  # (g, forbidden_residue)
        for r_i, m_i in zip(residues, moduli):
            g = math.gcd(m_i, m)
            if g == 1:
                continue  # no constraint
            constraints.append((g, r_i % g))
        
        if not constraints:
            return 0  # any residue works, pick 0
        
        # Try to find r in [0, m) satisfying all constraints
        # Brute force over r in [0, m) is at most 128 iterations
        for r in range(m):
            ok = True
            for g, forbidden in constraints:
                if r % g == forbidden:
                    ok = False
                    break
            if ok:
                return r
        return -1  # no valid residue
    
    # Greedy: pick moduli in order of "easiness" (fewer constraints imposed)
    # Score a modulus by how many other candidates it shares factors with
    # Prefer moduli with fewer/smaller gcds with remaining candidates
    
    def constraint_count(m, others):
        """Count total constraint weight: sum of gcd(m, m') for m' in others, m' != m"""
        total = 0
        for o in others:
            if o == m:
                continue
            g = math.gcd(m, o)
            if g > 1:
                total += g
        return total
    
    # Greedy selection with backtracking
    best_result = [[1, 2]]  # fallback: just one class
    
    def backtrack(selected_moduli, selected_residues, remaining):
        nonlocal best_result
        if time.time() > deadline:
            return
        
        if len(selected_moduli) > len(best_result):
            best_result = [[selected_residues[i], selected_moduli[i]] for i in range(len(selected_moduli))]
        
        if not remaining:
            return
        
        # Try each remaining modulus, ordered by constraint_count (ascending)
        scored = []
        for m in remaining:
            cc = constraint_count(m, selected_moduli)
            scored.append((cc, m))
        scored.sort()
        
        for cc, m in scored:
            if time.time() > deadline:
                return
            r = can_assign(selected_moduli, selected_residues, m)
            if r >= 0:
                new_remaining = [x for x in remaining if x != m]
                backtrack(selected_moduli + [m], selected_residues + [r], new_remaining)
                if time.time() > deadline:
                    return
    
    # Initial greedy pass to get a good starting set
    # Sort candidates by "isolation" - prefer primes and prime powers
    def prime_factor_count(m):
        """Return sum of prime factors (with multiplicity) - lower is better"""
        n = m
        total = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                total += d
                n //= d
            d += 1
        if n > 1:
            total += n
        return total
    
    initial_order = sorted(candidates, key=prime_factor_count)
    
    # Greedy initial selection
    greedy_moduli = []
    greedy_residues = []
    for m in initial_order:
        r = can_assign(greedy_moduli, greedy_residues, m)
        if r >= 0:
            greedy_moduli.append(m)
            greedy_residues.append(r)
    
    best_result = [[greedy_residues[i], greedy_moduli[i]] for i in range(len(greedy_moduli))]
    
    # Now try to improve with backtracking from a few different orderings
    # Try ordering by constraint_count dynamically
    if time.time() < deadline - 5:
        # Backtrack from the greedy solution, trying to add more
        remaining = [m for m in candidates if m not in greedy_moduli]
        backtrack(list(greedy_moduli), list(greedy_residues), remaining)
    
    # Try a few more initial orderings if time permits
    if time.time() < deadline - 10:
        # Order by number of distinct prime factors (ascending)
        def num_distinct_primes(m):
            n = m
            count = 0
            d = 2
            while d * d <= n:
                if n % d == 0:
                    count += 1
                    while n % d == 0:
                        n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        
        order2 = sorted(candidates, key=lambda m: (num_distinct_primes(m), prime_factor_count(m)))
        g2_moduli = []
        g2_residues = []
        for m in order2:
            r = can_assign(g2_moduli, g2_residues, m)
            if r >= 0:
                g2_moduli.append(m)
                g2_residues.append(r)
        
        if len(g2_moduli) > len(best_result):
            best_result = [[g2_residues[i], g2_moduli[i]] for i in range(len(g2_moduli))]
        
        if time.time() < deadline - 5:
            remaining2 = [m for m in candidates if m not in g2_moduli]
            backtrack(list(g2_moduli), list(g2_residues), remaining2)
    
    # Try ordering purely by modulus value
    if time.time() < deadline - 10:
        g3_moduli = []
        g3_residues = []
        for m in range(2, MAX_M + 1):
            r = can_assign(g3_moduli, g3_residues, m)
            if r >= 0:
                g3_moduli.append(m)
                g3_residues.append(r)
        
        if len(g3_moduli) > len(best_result):
            best_result = [[g3_residues[i], g3_moduli[i]] for i in range(len(g3_moduli))]
        
        if time.time() < deadline - 5:
            remaining3 = [m for m in candidates if m not in g3_moduli]
            backtrack(list(g3_moduli), list(g3_residues), remaining3)
    
    # Try descending modulus order
    if time.time() < deadline - 10:
        g4_moduli = []
        g4_residues = []
        for m in range(MAX_M, 1, -1):
            r = can_assign(g4_moduli, g4_residues, m)
            if r >= 0:
                g4_moduli.append(m)
                g4_residues.append(r)
        
        if len(g4_moduli) > len(best_result):
            best_result = [[g4_residues[i], g4_moduli[i]] for i in range(len(g4_moduli))]
        
        if time.time() < deadline - 5:
            remaining4 = [m for m in candidates if m not in g4_moduli]
            backtrack(list(g4_moduli), list(g4_residues), remaining4)
    
    if not best_result:
        best_result = [[0, 2]]
    
    return best_result
