"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _primes_upto(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(classes, cap=100000):
    """Compute the largest L such that 1..L are all covered."""
    if not classes:
        return 0
    pr_pairs = [(p, r) for p, r in classes]
    L = 0
    for n in range(1, cap + 1):
        covered = False
        for p, r in pr_pairs:
            if n % p == r:
                covered = True
                break
        if covered:
            L = n
        else:
            break
    return L

def _compute_L_fast(classes, cap=100000):
    """Faster L computation using a precomputed coverage array for small ranges."""
    if not classes:
        return 0
    pr_pairs = [(p, r) for p, r in classes]
    # Use a bitset-like approach: for each n, check coverage
    # Optimized by checking only the necessary numbers
    L = 0
    for n in range(1, cap + 1):
        covered = False
        for p, r in pr_pairs:
            if n % p == r:
                covered = True
                break
        if covered:
            L = n
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    all_primes = _primes_upto(1000)
    primes_desc = sorted(all_primes, reverse=True)
    
    # Start with the baseline solution: primes from 13 upward, residue (i+1) % p
    baseline_primes = [p for p in range(13, 1001) if p in set(all_primes)]
    baseline_classes = [[p, (i + 1) % p] for i, p in enumerate(baseline_primes)]
    baseline_L = _compute_L(baseline_classes)
    
    best_classes = baseline_classes
    best_L = baseline_L
    best_cost = sum(Fraction(1, p) for p, r in best_classes)
    
    used_primes = set(p for p, r in best_classes)
    
    # Greedy: try to extend coverage by adding classes
    # For each unused prime, try to find a residue that extends L the most
    for p in primes_desc:
        if time.time() > deadline:
            break
        if p in used_primes:
            continue
        
        current_L = best_L
        best_r = None
        best_new_L = current_L
        
        # Try residues that cover current_L + 1 first (most likely to help)
        # Then try other promising residues
        # To save time, we'll try a limited set of residues
        
        # The residue that covers current_L+1 is r = (current_L+1) % p
        r_target = (current_L + 1) % p
        
        # Try r_target first, then a few others
        residues_to_try = [r_target]
        # Add a few nearby residues for diversity
        for dr in range(1, min(5, p)):
            residues_to_try.append((r_target + dr) % p)
            residues_to_try.append((r_target - dr) % p)
        # Remove duplicates while preserving order
        seen = set()
        unique_residues = []
        for r in residues_to_try:
            if r not in seen:
                seen.add(r)
                unique_residues.append(r)
        
        for r in unique_residues:
            if time.time() > deadline:
                break
            
            # Simulate: with this class, how far can we go?
            test_L = current_L
            extended = False
            for n in range(current_L + 1, min(current_L + 500, 100001)):
                # Check if n is covered by old classes
                old_covered = False
                for op, orr in best_classes:
                    if n % op == orr:
                        old_covered = True
                        break
                if not old_covered:
                    # Check new class
                    if n % p == r:
                        test_L = n
                    else:
                        break
                else:
                    test_L = n
            
            if test_L > best_new_L:
                best_new_L = test_L
                best_r = r
        
        if best_r is not None and best_new_L > best_L:
            # Check budget
            new_cost = best_cost + Fraction(1, p)
            if new_cost <= Fraction(1, 1):
                best_classes = best_classes + [[p, best_r]]
                best_L = best_new_L
                best_cost = new_cost
                used_primes.add(p)
    
    # Try to improve residues of existing classes
    # For each class, try shifting its residue to see if it extends L
    for idx in range(len(best_classes)):
        if time.time() > deadline:
            break
        p, r = best_classes[idx]
        old_r = r
        best_r_for_p = r
        best_L_for_p = best_L
        
        # Try a few alternative residues
        for dr in range(-3, 4):
            if dr == 0:
                continue
            new_r = (r + dr) % p
            if new_r == r:
                continue
            # Test this alternative
            test_classes = best_classes.copy()
            test_classes[idx] = [p, new_r]
            test_L = _compute_L(test_classes, cap=min(best_L + 200, 100000))
            if test_L > best_L_for_p:
                best_L_for_p = test_L
                best_r_for_p = new_r
        
        if best_r_for_p != old_r:
            best_classes[idx] = [p, best_r_for_p]
            best_L = best_L_for_p
    
    # Final verification
    final_L = _compute_L(best_classes)
    if final_L > best_L:
        best_L = final_L
    
    return best_classes
# EVOLVE-BLOCK-END
