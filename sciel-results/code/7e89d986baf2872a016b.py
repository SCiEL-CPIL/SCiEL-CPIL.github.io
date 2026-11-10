import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    max_m = 128
    all_moduli = list(range(2, max_m + 1))
    
    # Precompute gcd table for all pairs of moduli
    n_moduli = len(all_moduli)
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_disjoint(m1, m2, r1, r2):
        """Check if residue class (r1, m1) and (r2, m2) are disjoint.
        They are disjoint iff r1 % gcd(m1,m2) != r2 % gcd(m1,m2)"""
        g = gcd_table[all_moduli.index(m1)][all_moduli.index(m2)]
        return r1 % g != r2 % g
    
    # Greedy approach: try to add moduli in a smart order
    # Key insight: moduli with more prime factors create more constraints
    # Start with prime moduli (least restrictive), then add composite ones
    
    def score_selection(selected):
        """selected is a list of (modulus, residue) pairs"""
        return len(selected)
    
    def try_add_greedy(moduli_order):
        """Try to greedily add moduli in given order, finding valid residues"""
        selected = []  # list of (modulus, residue)
        
        for m in moduli_order:
            # Find a residue r in [0, m) that is disjoint from all selected
            # For each selected (m_s, r_s), we need r % gcd(m, m_s) != r_s % gcd(m, m_s)
            
            # Collect constraints: for each selected pair, forbidden residues modulo gcd
            # r must avoid specific residues modulo each gcd(m, m_s)
            
            # Use CRT-like approach: try all residues 0..m-1 and check
            found = False
            for r in range(m):
                valid = True
                for m_s, r_s in selected:
                    g = gcd(m, m_s)
                    if r % g == r_s % g:
                        valid = False
                        break
                if valid:
                    selected.append((m, r))
                    found = True
                    break
        
        return selected
    
    # Strategy: use multiple random orderings and pick the best
    best_selection = []
    
    # First, try a deterministic good ordering
    # Sort moduli by: fewer prime factors first (primes, then semiprimes, etc.)
    def count_prime_factors_with_repeats(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Sort by number of prime factors (ascending), then by value
    sorted_moduli = sorted(all_moduli, key=lambda m: (count_prime_factors_with_repeats(m), m))
    
    result = try_add_greedy(sorted_moduli)
    if len(result) > len(best_selection):
        best_selection = result
    
    # Try random orderings
    random.seed(42)
    num_tries = 200
    
    for trial in range(num_tries):
        if time.time() > deadline:
            break
        
        # Create a random permutation with bias toward primes
        order = all_moduli[:]
        random.shuffle(order)
        
        result = try_add_greedy(order)
        if len(result) > len(best_selection):
            best_selection = result
    
    # Local search: try to improve by swapping or re-adding moduli
    # Try removing some moduli and re-adding others
    for _ in range(50):
        if time.time() > deadline:
            break
        
        if not best_selection:
            break
        
        current_moduli = set(m for m, r in best_selection)
        unused_moduli = [m for m in all_moduli if m not in current_moduli]
        
        if not unused_moduli:
            break
        
        # Try swapping one selected modulus with one unused
        best_swap = None
        best_swap_score = len(best_selection)
        
        for m_remove in random.sample(list(current_moduli), min(5, len(current_moduli))):
            for m_add in random.sample(unused_moduli, min(10, len(unused_moduli))):
                # Try removing m_remove and adding m_add
                new_selected = [(m, r) for m, r in best_selection if m != m_remove]
                
                # Try to add m_add to new_selected
                for r in range(m_add):
                    valid = True
                    for m_s, r_s in new_selected:
                        g = gcd(m_add, m_s)
                        if r % g == r_s % g:
                            valid = False
                            break
                    if valid:
                        new_selected.append((m_add, r))
                        if len(new_selected) > best_swap_score:
                            best_swap_score = len(new_selected)
                            best_swap = new_selected[:]
                        break
        
        if best_swap:
            best_selection = best_swap
    
    # Convert to required format: list of [residue, modulus] pairs
    result = [[r, m] for m, r in best_selection]
    
    # Ensure nonempty
    if not result:
        result = [[1, 2]]
    
    return result
