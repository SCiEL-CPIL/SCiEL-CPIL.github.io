"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import isqrt

# EVOLVE-BLOCK-START

def _primes_upto(n):
    """Return list of primes <= n via sieve."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _coverage_length(classes, cap=100000):
    """Return the largest L such that every integer 1..L is covered by some class."""
    if not classes:
        return 0
    covered = bytearray(cap + 1)  # index 0 unused
    for p, r in classes:
        # Mark all n in [1, cap] with n % p == r
        # First occurrence >= 1: start = r if r >= 1 else r + p
        start = r if r >= 1 else r + p
        if start > cap:
            continue
        covered[start::p] = bytearray(b'\x01') * len(covered[start::p])
    # Find largest L such that covered[1..L] are all 1
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin (evaluator kills at 150s)
    
    primes = _primes_upto(1000)
    
    # Greedy construction: iteratively pick the prime+residue that maximizes
    # coverage extension per unit of reciprocal cost.
    
    # State: which integers 1..MAXN are currently covered
    MAXN = 100000
    covered = bytearray(MAXN + 1)
    
    # Precompute for each prime p the list of residues and their coverage patterns
    # For efficiency, track for each prime p which residues have been used
    used_residues = {p: set() for p in primes}
    
    # Current budget: sum of 1/p for selected classes
    budget_used = Fraction(0)
    budget_limit = Fraction(1)
    
    selected = []  # list of [p, r]
    
    # For greedy: at each step, find the (p, r) that extends coverage the most
    # while keeping budget <= 1.
    
    # To make this efficient, we'll evaluate candidates in batches.
    # Score for a candidate (p, r): number of new integers covered in [1, current_L + window]
    # where window is some lookahead.
    
    # Actually, let's use a simpler approach: 
    # Score = (number of new covered integers in 1..MAXN) / (1/p)
    # i.e., coverage density per unit cost.
    
    # But computing full coverage for each candidate is expensive.
    # Instead, let's use a window-based approach:
    # Only consider integers up to current_L + LOOKAHEAD for scoring.
    
    LOOKAHEAD = 2000
    
    # Initial: no coverage
    current_L = 0
    
    # We'll iterate: at each step, evaluate all available (p, r) candidates
    # and pick the best one by a score function.
    
    # To speed up, precompute for each prime p the residues in order of
    # how many new integers they'd cover (initially, residue 0 covers p, 2p, 3p...;
    # residue r covers r, r+p, r+2p, ...)
    
    # Let's just do iterative greedy with a reasonable number of iterations.
    
    # For scoring, we compute new coverage in a window [1, min(current_L + LOOKAHEAD, MAXN)]
    
    def _score_candidate(p, r, covered_arr, window_end):
        """Count new integers covered by class (p, r) in [1, window_end]."""
        start = r if r >= 1 else r + p
        if start > window_end:
            return 0
        count = 0
        for n in range(start, window_end + 1, p):
            if not covered_arr[n]:
                count += 1
        return count
    
    # Main greedy loop
    max_iterations = 200  # at most 200 classes
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline - 20.0:
            break  # Safety margin for final evaluation
        
        window_end = min(current_L + LOOKAHEAD, MAXN)
        
        best_score = -1
        best_p = None
        best_r = None
        best_new_count = 0
        
        # Evaluate all available candidates
        for p in primes:
            cost = Fraction(1, p)
            if budget_used + cost > budget_limit:
                continue
            
            # Try each unused residue for this prime
            for r in range(p):
                if r in used_residues[p]:
                    continue
                
                # Quick estimate: count new coverages
                start = r if r >= 1 else r + p
                if start > window_end:
                    continue
                
                # Count new covered integers
                new_count = 0
                for n in range(start, window_end + 1, p):
                    if not covered[n]:
                        new_count += 1
                
                if new_count <= 0:
                    continue
                
                # Score: new_count / cost = new_count * p
                score = new_count * p
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
                    best_new_count = new_count
        
        if best_p is None:
            break  # No more candidates that improve coverage
        
        # Apply the best candidate
        p, r = best_p, best_r
        selected.append([p, r])
        used_residues[p].add(r)
        budget_used += Fraction(1, p)
        
        # Update coverage
        start = r if r >= 1 else r + p
        for n in range(start, MAXN + 1, p):
            covered[n] = 1
        
        # Update current_L
        # Find new current_L by scanning from old current_L
        new_L = current_L
        while new_L + 1 <= MAXN and covered[new_L + 1]:
            new_L += 1
        current_L = new_L
        
        iteration += 1
        
        if current_L >= 999:
            break  # We've reached the target
    
    # Final: compute exact coverage length
    L = _coverage_length(selected)
    
    # Verify budget constraint with exact rationals
    total = sum(Fraction(1, p) for p, r in selected)
    if total > 1:
        # This shouldn't happen, but just in case, remove classes until valid
        # (shouldn't be needed since we check before adding)
        pass
    
    return selected

# EVOLVE-BLOCK-END
