"""
Strategy: Greedy Set Cover for Prime Residue Classes.

The task is to cover integers 1..L with prime residue classes {p, r}
such that sum(1/p) <= 1 and p <= 1000. We maximize L.

Approach:
1. Generate all primes up to 1000.
2. Use a greedy algorithm to select (p, r) pairs that maximize coverage
   of consecutive integers starting from 1, subject to the reciprocal budget.
3. The key insight: we want to cover 1, 2, 3, ... as far as possible.
   For each integer n not yet covered, we need some (p, r) with n ≡ r (mod p).
   We prefer small p (cheap) that cover many uncovered integers.

Better approach: 
- We have budget B = 1 (sum of 1/p <= 1).
- We want to maximize the consecutive prefix 1..L that is covered.
- Use greedy: at each step, among all available primes, pick the one whose
  best residue covers the most currently-uncovered integers in the prefix,
  weighted by cost.

Actually, let's think more carefully. The seed covers about 999 with a simple
heuristic. To do better, we need a smarter selection.

Key observation: The sum of reciprocals of all primes up to 1000 is about 2.2,
so we can only use a subset. The budget is 1.

Let me think about what gives L >= 999. The seed uses primes starting from 13
and maps the i-th such prime to cover integer i+1 (via residue (i+1) % p).
This gives sum of 1/p for primes 13,17,19,...,997 which is roughly 0.92.

To improve, we should use a proper greedy set cover:
1. Start with target L (try to cover 1..L_max where L_max is large, say 10000).
2. Maintain a set of uncovered integers in 1..L_max.
3. At each step, for each available prime p, find the residue r that covers
   the most uncovered integers. Pick the (p, r) with best coverage/cost ratio.
4. Continue until budget is exhausted or no improvement.
5. Then compute the actual consecutive prefix covered.

But this is O(num_primes * L_max) per step which is too slow for large L_max.

Alternative: Focus on covering the prefix 1..L. For each n in 1..L, it's covered
if there exists a selected (p, r) with n ≡ r (mod p).

Let me use a different strategy:
- We know we want to cover 1..L. 
- For small n, many primes can cover it. For large n, fewer options.
- The critical integers are those that are hard to cover.

Actually, the simplest improvement: use all primes up to some bound with their
optimal residues, selected greedily by coverage gain per unit cost.

Let me implement a time-bounded greedy:
- Target: cover as much of 1..L_max as possible.
- L_max = 20000 (to allow for potential improvement beyond 999).
- Greedy: repeatedly pick (p, r) that maximizes new coverage / (1/p).
- Stop when budget < 1 or no positive gain.
- Track elapsed time; return best result found within deadline.

For efficiency, represent uncovered as a boolean array of length L_max+1.
For each prime p and residue r, coverage = number of uncovered n in 1..L_max
with n ≡ r (mod p). This can be computed by stepping through r, r+p, r+2p, ...

Number of primes up to 1000: 168.
For each prime, 168 residues * (L_max/p) steps. Total per iteration: sum over p of (L_max) ≈ 168 * 20000 = 3.36M.
With ~50-100 iterations, that's ~300M operations. Might be tight in 150s in Python.

Let me optimize: only consider primes where the best residue covers at least 1 uncovered.
Also, use numpy for the coverage computation.

Actually, let me use a smarter approach. Since we care about the prefix 1..L,
once we have a set of classes, L is the largest integer such that 1..L are all covered.

Let me try a different angle: 
- The seed gets L around 999. 
- To exceed 999, we need to cover 1000, 1001, etc.
- Integer 1000: needs a prime p and residue r=1000%p. 
- The seed already covers up to ~999. The issue is that some integers in 1..999 might
  not be covered, or the coverage stops before 999.

Let me just implement a solid greedy and see what L we get.
"""

import time
from fractions import Fraction

def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _coverage_count(uncovered, p, r, L_max):
    """Count uncovered integers n in [1, L_max] with n ≡ r (mod p)."""
    # n = r, r+p, r+2p, ... but n >= 1
    # If r == 0, n = p, 2p, 3p, ...
    # If r > 0, n = r, r+p, r+2p, ... but need n >= 1
    start = r
    if start < 1:
        start += ((1 - start + p - 1) // p) * p
    if start > L_max:
        return 0
    count = 0
    n = start
    while n <= L_max:
        if uncovered[n]:
            count += 1
        n += p
    return count

def construct():
    deadline = time.time() + 120  # 120s budget, with margin for 150s limit
    
    primes = _sieve(1000)
    num_primes = len(primes)
    
    # We'll try to cover 1..L_max
    L_max = 20000
    
    # uncovered[i] = True if integer i is not yet covered, for i in 1..L_max
    uncovered = [True] * (L_max + 1)
    uncovered[0] = False  # we don't care about 0
    
    selected = []  # list of (p, r)
    budget_used = Fraction(0)
    budget_limit = Fraction(1)
    
    # Available primes: all of them initially
    available = list(range(num_primes))
    
    # Precompute: for each prime index, the prime value
    prime_vals = [primes[i] for i in range(num_primes)]
    
    while time.time() < deadline - 5:  # 5s safety margin
        best_gain = 0
        best_pi = -1
        best_r = -1
        best_cost = Fraction(1)
        
        for pi in available:
            p = prime_vals[pi]
            cost = Fraction(1, p)
            
            # Find the residue r that gives max coverage
            max_cov = 0
            best_r_for_p = 0
            for r in range(p):
                cov = _coverage_count(uncovered, p, r, L_max)
                if cov > max_cov:
                    max_cov = cov
                    best_r_for_p = r
            
            if max_cov == 0:
                continue
            
            # Gain per unit cost (use float for comparison, Fraction for exact budget)
            gain_ratio = max_cov / (1.0 / p)  # = max_cov * p
            
            if gain_ratio > best_gain or (gain_ratio == best_gain and cost < best_cost):
                best_gain = gain_ratio
                best_pi = pi
                best_r = best_r_for_p
                best_cost = cost
        
        if best_pi == -1:
            break  # No more coverage possible
        
        p = prime_vals[best_pi]
        r = best_r
        
        # Check budget
        if budget_used + best_cost > budget_limit:
            # Can't afford this prime; try to find the cheapest that still gives gain
            # For now, just break
            break
        
        # Apply the selection
        selected.append([p, r])
        budget_used += best_cost
        
        # Update uncovered
        n = r
        if n < 1:
            n += ((1 - n + p - 1) // p) * p
        while n <= L_max:
            if n >= 1 and n <= L_max:
                uncovered[n] = False
            n += p
        
        # Remove from available
        available.remove(best_pi)
        
        # Check if we've covered the entire prefix up to some point
        # (early termination if no more gains possible)
    
    # Compute the actual L: largest such that 1..L are all covered
    L = 0
    for i in range(1, L_max + 1):
        if uncovered[i]:
            L = i - 1
            break
    else:
        L = L_max
    
    # Cap at 10^5
    if L > 100000:
        L = 100000
    
    return selected

# Quick test
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s, Classes: {len(result)}")
    # Compute L
    L_max = 20000
    uncovered = [True] * (L_max + 1)
    for p, r in result:
        n = r
        if n < 1:
            n += ((1 - n + p - 1) // p) * p
        while n <= L_max:
            if 1 <= n <= L_max:
                uncovered[n] = False
            n += p
    L = 0
    for i in range(1, L_max + 1):
        if uncovered[i]:
            L = i - 1
            break
    else:
        L = L_max
    print(f"L = {L}")
