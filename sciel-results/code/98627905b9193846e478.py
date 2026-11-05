"""Greedy submodular maximization for doubly-covered prefix length."""
import time
import math

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(n+1) if sieve[i]]

PRIMES = _primes_up_to(1000)
NUM_PRIMES = len(PRIMES)  # 168

# We'll work with a working window. The score is the largest L such that
# every integer 1..L satisfies at least two congruences.
# We need to maximize this L.

def _count_coverage(residues, primes, limit):
    """Count how many primes cover each integer in 1..limit.
    Returns a list of counts for integers 1..limit."""
    counts = [0] * (limit + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        # x ≡ r (mod p) means x = r + k*p for integer k
        # We need x in [1, limit]
        # First x >= 1: r + k*p >= 1 => k >= (1-r)/p
        # If r == 0, first x is p (since x must be positive and x ≡ 0 mod p means x is multiple of p)
        # Actually x = r mod p, so x can be r, r+p, r+2p, ... but we need x >= 1
        # If r == 0, the values are p, 2p, 3p, ...
        # If r > 0, the values are r, r+p, r+2p, ... but we need x >= 1, so if r >= 1, start at r
        if r == 0:
            start = p
        else:
            start = r
        if start > limit:
            continue
        for x in range(start, limit + 1, p):
            counts[x] += 1
    return counts

def _doubly_covered_length(residues, primes, max_check):
    """Find the largest L such that every integer 1..L has count >= 2."""
    counts = _count_coverage(residues, primes, max_check)
    L = 0
    for i in range(1, max_check + 1):
        if counts[i] >= 2:
            L = i
        else:
            break
    return L

def _score(residues, primes, max_check=10000):
    """Score: largest L with full double coverage, capped at 10^5."""
    L = _doubly_covered_length(residues, primes, max_check)
    return min(L, 100000)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    primes = PRIMES
    n_primes = NUM_PRIMES
    
    # Initial residues: use a simple heuristic
    # For each prime p_i, set residue to cover some target
    # Start with residues that give decent initial coverage
    # Simple init: residue i = i % p_i
    residues = [i % p for i, p in enumerate(primes)]
    
    # Compute initial score
    current_score = _score(residues, primes, 2000)
    
    # Greedy hill-climbing with random restarts
    best_residues = residues[:]
    best_score = current_score
    
    import random
    random.seed(42)
    
    # Phase 1: Greedy local search
    # For each prime, try all possible residues and pick the best
    # Iterate until no improvement or time runs out
    
    max_check = 2000  # We'll check up to 2000 initially
    
    # Count how many times each integer is covered
    def _get_counts(residues, primes, limit):
        counts = [0] * (limit + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > limit:
                continue
            for x in range(start, limit + 1, p):
                counts[x] += 1
        return counts
    
    def _deficit_length(counts, limit):
        """Find the first position where count < 2."""
        for i in range(1, limit + 1):
            if counts[i] < 2:
                return i
        return limit + 1
    
    # Local search: for each prime, try changing its residue
    # Focus on primes that can help fix the current bottleneck
    
    while time.time() < deadline:
        counts = _get_counts(residues, primes, max_check)
        bottleneck = _deficit_length(counts, max_check)
        
        if bottleneck > max_check:
            # We've covered up to max_check, try extending
            max_check = min(max_check * 2, 10000)
            continue
        
        # Try to fix the bottleneck by changing residues
        improved = False
        for trial in range(3):  # Try a few times per iteration
            # Pick a random prime that covers the bottleneck or nearby
            # Or just pick a random prime
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            
            # Try a few random residues for this prime
            for _ in range(5):
                new_r = random.randint(0, p - 1)
                if new_r == residues[idx]:
                    continue
                old_r = residues[idx]
                residues[idx] = new_r
                
                new_counts = _get_counts(residues, primes, max_check)
                new_bottleneck = _deficit_length(new_counts, max_check)
                
                if new_bottleneck > bottleneck:
                    bottleneck = new_bottleneck
                    improved = True
                    break
                else:
                    residues[idx] = old_r
            
            if improved:
                break
        
        if not improved:
            # Try a more systematic approach: for each prime, find the best residue
            # that maximizes the bottleneck
            any_improved = False
            for idx in range(n_primes):
                if time.time() > deadline:
                    break
                p = primes[idx]
                old_r = residues[idx]
                best_new_r = old_r
                best_bottleneck = bottleneck
                
                # Try a subset of residues (can't try all if p is large)
                if p <= 50:
                    candidates = list(range(p))
                else:
                    # Try residues that cover the bottleneck
                    # x = bottleneck, we need bottleneck ≡ r (mod p)
                    # So r = bottleneck % p
                    r_target = bottleneck % p
                    candidates = [r_target]
                    # Also try nearby
                    for delta in range(1, 10):
                        candidates.append((r_target + delta) % p)
                        candidates.append((r_target - delta) % p)
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    residues[idx] = new_r
                    new_counts = _get_counts(residues, primes, max_check)
                    new_bottleneck = _deficit_length(new_counts, max_check)
                    if new_bottleneck > best_bottleneck:
                        best_bottleneck = new_bottleneck
                        best_new_r = new_r
                
                if best_new_r != old_r:
                    residues[idx] = best_new_r
                    bottleneck = best_bottleneck
                    any_improved = True
            
            if not any_improved:
                # Random restart
                residues = [random.randint(0, p - 1) for p in primes]
                counts = _get_counts(residues, primes, max_check)
                bottleneck = _deficit_length(counts, max_check)
        
        # Update best
        current_score = _score(residues, primes, max_check)
        if current_score > best_score:
            best_score = current_score
            best_residues = residues[:]
    
    # Final evaluation with larger check
    final_score = _score(best_residues, primes, 10000)
    if final_score > best_score:
        best_score = final_score
    
    return best_residues
