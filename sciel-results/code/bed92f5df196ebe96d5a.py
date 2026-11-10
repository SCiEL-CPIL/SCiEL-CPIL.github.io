"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(classes, cap=10000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    # For each number 1..cap, check if it's covered
    covered = bytearray(cap + 1)  # 0 = not covered, 1 = covered
    for p, r in classes:
        # Mark all numbers n in 1..cap where n ≡ r (mod p)
        start = r if r >= 1 else p  # smallest positive n ≡ r (mod p)
        # Actually, n ≡ r (mod p) means n = r + kp for k >= 0, and n >= 1
        # If r == 0, n = p, 2p, 3p, ...
        # If r > 0, n = r, r+p, r+2p, ...
        if r == 0:
            start = p
        else:
            start = r
        for n in range(start, cap + 1, p):
            covered[n] = 1
    
    # Find the largest L such that all 1..L are covered
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    primes = _primes_upto(1000)
    
    # Start with the seed program's approach as baseline
    # The i-th prime from 13 upward covers the integer i
    primes_from_13 = [p for p in primes if p >= 13]
    baseline_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_from_13)]
    baseline_L = _compute_L(baseline_classes, cap=10000)
    
    # Try greedy covering: for each uncovered number, pick the prime that covers it with minimum cost
    best_classes = baseline_classes[:]
    best_L = baseline_L
    
    # Greedy approach: iteratively add classes to cover gaps
    used_primes = set(p for p, r in best_classes)
    budget = Fraction(1) - sum(Fraction(1, p) for p, r in best_classes)
    
    current_classes = best_classes[:]
    current_L = best_L
    
    # Try to improve by reassigning residues
    # For each class, try different residues to maximize coverage of the front
    import random
    rng = random.Random(42)
    
    # Greedy: try to cover the next uncovered number
    for _ in range(200):  # Iteration limit
        if time.time() > deadline:
            break
        
        L_tmp = _compute_L(current_classes, cap=10000)
        if L_tmp > current_L:
            current_L = L_tmp
            best_classes = current_classes[:]
            best_L = current_L
        
        if current_L >= 1000:
            break
        
        gap = current_L + 1
        if gap > 10000:
            break
        
        # Try to cover 'gap' with a new or existing class
        # Option 1: Add a new prime class
        best_option = None
        best_cost = None
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget + cost > Fraction(1):
                continue
            # To cover gap, need gap ≡ r (mod p), so r = gap % p
            r = gap % p
            # This covers gap and gap+p, gap+2p, etc.
            # Cost is 1/p
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_option = (p, r)
        
        if best_option is not None:
            p_new, r_new = best_option
            current_classes.append([p_new, r_new])
            used_primes.add(p_new)
            budget -= best_cost
        
        # Also try to improve existing classes' residues
        # For each existing class, try adjusting residue to cover more of the gap region
        # This is complex, so we'll just do simple swaps
        
    # Final evaluation
    final_L = _compute_L(best_classes, cap=10000)
    return best_classes
# EVOLVE-BLOCK-END
