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
    # Build coverage check
    # For each n from 1 upward, check if any class covers n
    # Use a set of (p, r) pairs
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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    all_primes = _primes_upto(1000)
    # Sort primes in descending order (largest first = cheapest per unit coverage)
    primes_desc = sorted(all_primes, reverse=True)
    
    # Start with the baseline solution
    baseline_primes = [p for p in range(13, 1001) if p in set(all_primes)]
    baseline_classes = [[p, (i + 1) % p] for i, p in enumerate(baseline_primes)]
    baseline_L = _compute_L(baseline_classes)
    
    best_classes = baseline_classes
    best_L = baseline_L
    best_cost = sum(Fraction(1, p) for p, r in best_classes)
    
    # Greedy: try to extend coverage by adding classes
    used_primes = set(p for p, r in best_classes)
    
    # Try adding primes not yet used, choosing residue to maximize new coverage
    for p in primes_desc:
        if time.time() > deadline:
            break
        if p in used_primes:
            continue
        # Try all residues for this prime, pick the one that extends L the most
        # But trying all residues is expensive. Instead, try residues that cover the first uncovered number.
        
        # Find the first uncovered number
        current_L = best_L
        # Check if we can extend
        new_L = current_L
        best_r = None
        best_new_L = current_L
        
        # Try a subset of residues: those that cover numbers near current_L+1
        # The residue r covers r, r+p, r+2p, ...
        # To cover current_L+1, we need r = (current_L+1) % p
        # But other residues might cover more future numbers
        
        # Try all residues that are "useful": r such that r + kp <= some bound
        # Actually, let's just try all p residues (p <= 1000, so at most 1000 tries)
        for r in range(p):
            if time.time() > deadline:
                break
            # Check how far this class extends coverage from current_L
            # A number n is covered by (p, r) if n % p == r
            # We need to find the max L' such that 1..L' are all covered
            # with the new class added
            
            # Quick check: does this class cover current_L + 1?
            if (current_L + 1) % p != r:
                # It doesn't directly cover the next number, but might help later
                # Skip for now (optimization)
                continue
            
            # Simulate: with this class, how far can we go?
            # We know 1..current_L are covered. Check current_L+1, current_L+2, ...
            # A number n is covered if it's in the old classes OR in (p, r)
            # Old classes cover 1..current_L by definition.
            # For n > current_L, it's covered if n % p == r OR it's covered by old classes.
            # But old classes might cover some n > current_L too!
            
            # This is getting complex. Let's do a simpler check:
            # Just verify that the new class covers current_L+1, and hope it helps.
            # For a proper check, we'd need to simulate, which is expensive.
            
            # Let's do a limited simulation
            test_L = current_L
            for n in range(current_L + 1, min(current_L + 200, 100001)):
                # Check if n is covered by old classes
                old_covered = False
                for op, orr in best_classes:
                    if n % op == orr:
                        old_covered = True
                        break
                if not old_covered:
                    # Check new class
                    if n % p != r:
                        break
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
    
    # Final verification
    final_L = _compute_L(best_classes)
    if final_L > best_L:
        best_L = final_L
    
    return best_classes
# EVOLVE-BLOCK-END
