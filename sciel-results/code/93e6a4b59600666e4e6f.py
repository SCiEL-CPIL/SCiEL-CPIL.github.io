"""
Structural redesign: instead of a fixed heuristic assignment, use a greedy
set-cover / local-search approach to maximize the contiguous covered prefix
1..L under the constraint sum(1/p) <= 1 with distinct primes <= 1000.

Key structural changes from the seed:
1. The seed assigns residue (i+1) % p_i to the i-th prime >= 13, which is
   essentially arbitrary and wastes the budget. Instead, we treat this as a
   covering problem: choose a subset of primes (with sum 1/p <= 1) and assign
   residues so that 1..L is fully covered.
2. We use a greedy algorithm: repeatedly pick the prime/residue pair that
   covers the most currently-uncovered integers in the prefix, subject to
   the budget constraint.
3. We then try to extend coverage by swapping or adding classes.
4. We include a deadline guard to stay under 150s.

The insight: to cover 1..L, we need for each integer n in 1..L some prime p
and residue r such that n ≡ r (mod p). A single class (p, r) covers about
L/p integers in 1..L. The budget sum(1/p) <= 1 limits how many classes we
can have.

A better structural approach: 
- For a target L, the problem is: can we cover 1..L with classes (p_i, r_i),
  distinct primes <= 1000, sum 1/p_i <= 1?
- We binary search on L and use a greedy + local search to find a covering.

But since we just need to maximize L, let's do:
1. Start with L=1000 (the target).
2. Try to find a covering for 1..L using greedy set cover with budget.
3. If it fails, reduce L and retry, or use local search to improve.

Actually, a cleaner structural approach:
- Precompute all primes <= 1000.
- Use a greedy algorithm: at each step, among all (p, r) pairs where p is
  not yet used, pick the one that maximizes the number of newly covered
  integers in 1..L, subject to adding 1/p not exceeding the remaining budget.
- After greedy, do local search: try swapping a class for another to extend
  coverage.

Let me think about the budget: sum 1/p <= 1. The sum of reciprocals of all
primes up to 1000 is about log(log(1000)) + M ≈ 2.198 + 0.261 ≈ 2.46, so we
have plenty of room. The constraint is tight enough to limit choices.

For covering 1..L with L up to 1000, each class (p, r) covers floor((L-r)/p) + (1 if r>0 else 0) integers roughly L/p.

Let me implement a robust greedy + local search.
"""

import time
from fractions import Fraction

def _primes_upto(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _coverage_length(classes, cap=100000):
    """Return the largest L such that every integer 1..L is in some class."""
    if not classes:
        return 0
    covered = set()
    for p, r in classes:
        # integers n in 1..cap with n % p == r
        # first such n: if r == 0, then n = p, 2p, ...; else n = r, r+p, ...
        # but n must be >= 1
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start = 1 + (1 - start) % p  # ensure >= 1
            # Actually just: smallest n >= 1 with n % p == r
            # if r == 0: n = p, 2p, ...
            # if r > 0: n = r, r+p, ... (r >= 1 since 0 <= r < p, r>0 means r>=1)
            pass
        # Let's just compute properly
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p * ((1 - first + p - 1) // p)
        # Actually simpler: iterate
        # For efficiency with cap up to 100000 and small number of classes, this is fine
        n = first
        while n <= cap:
            covered.add(n)
            n += p
    
    # Find largest L such that 1..L all in covered
    L = 0
    for i in range(1, cap + 1):
        if i in covered:
            L = i
        else:
            break
    return L

def _is_valid(classes):
    """Check: distinct primes <= 1000, 0 <= r < p, sum 1/p <= 1."""
    seen = set()
    total = Fraction(0)
    for p, r in classes:
        if p in seen:
            return False
        if p > 1000 or p < 2:
            return False
        if not (0 <= r < p):
            return False
        seen.add(p)
        total += Fraction(1, p)
    return total <= 1

def _greedy_cover(target_L, primes, deadline):
    """
    Greedy set cover: pick (p, r) pairs to cover 1..target_L with budget sum(1/p) <= 1.
    Returns list of (p, r) or None if not possible.
    """
    budget = Fraction(1)
    used_primes = set()
    covered = set()
    classes = []
    
    # Precompute for each prime p and residue r, how many of 1..target_L it covers
    # and which ones
    # For efficiency, for each p, the best r is the one that covers the most uncovered
    
    # We'll do iterative greedy: at each step, find the (p, r) that maximizes
    # new coverage / (1/p) or just new coverage, subject to budget.
    
    uncovered = set(range(1, target_L + 1))
    
    while uncovered and time.time() < deadline:
        best_gain = 0
        best_pair = None
        best_cost = None
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget - cost < 0:
                continue
            
            # For this p, find r that maximizes coverage of uncovered
            # n ≡ r (mod p) for n in uncovered
            # Group uncovered by n % p
            counts = {}
            for n in uncovered:
                r_val = n % p
                counts[r_val] = counts.get(r_val, 0) + 1
            
            if not counts:
                continue
            
            best_r = max(counts, key=counts.get)
            gain = counts[best_r]
            
            # Prefer higher gain; tie-break by smaller p (cheaper)
            if gain > best_gain or (gain == best_gain and best_pair is not None and p < best_pair[0]):
                best_gain = gain
                best_pair = (p, best_r)
                best_cost = cost
        
        if best_pair is None or best_gain == 0:
            break
        
        p, r = best_pair
        classes.append((p, r))
        used_primes.add(p)
        budget -= best_cost
        
        # Update uncovered
        new_uncovered = set()
        for n in uncovered:
            if n % p != r:
                new_uncovered.add(n)
        uncovered = new_uncovered
    
    if not uncovered:
        return classes
    return None

def _local_search_improve(classes, target_L, primes, deadline):
    """
    Try to improve coverage by swapping classes. Returns improved classes or None.
    """
    # This is a simplified local search: try replacing one class with another
    # to cover more of the prefix.
    
    current_L = _coverage_length(classes, target_L + 100)
    
    for _ in range(50):  # limited iterations
        if time.time() > deadline:
            break
        if current_L >= target_L:
            break
        
        improved = False
        for i in range(len(classes)):
            p_old, r_old = classes[i]
            for p_new in primes:
                if p_new == p_old or p_new in [c[0] for c in classes]:
                    continue
                # Try all residues for p_new
                for r_new in range(p_new):
                    new_classes = classes[:i] + [(p_new, r_new)] + classes[i+1:]
                    new_L = _coverage_length(new_classes, target_L + 100)
                    if new_L > current_L:
                        classes = new_classes
                        current_L = new_L
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    primes = _primes_upto(1000)
    
    # Target: try to cover 1..999 (the Erdős-Ruzsa target)
    # First, try greedy for L=999
    best_classes = None
    best_L = 0
    
    # Try greedy for various target Ls, starting from 999 and going down
    for target_L in range(999, 500, -1):
        if time.time() > deadline:
            break
        result = _greedy_cover(target_L, primes, deadline)
        if result is not None:
            # Verify
            if _is_valid(result):
                L = _coverage_length(result, 100000)
                if L > best_L:
                    best_L = L
                    best_classes = result
                if L >= 999:
                    return [[p, r] for p, r in best_classes]
    
    # If greedy didn't reach 999, try local search on the best found
    if best_classes is not None:
        improved = _local_search_improve(best_classes, 999, primes, deadline)
        if improved is not None:
            L = _coverage_length(improved, 100000)
            if L > best_L:
                best_L = L
                best_classes = improved
    
    # Fallback: use the seed-like approach if nothing better
    if best_classes is None:
        # Seed approach
        seed_primes = [p for p in primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    
    # Final check
    if not _is_valid(best_classes):
        # Fallback to minimal valid
        best_classes = [[2, 0], [3, 0]]
    
    return [[p, r] for p, r in best_classes]
