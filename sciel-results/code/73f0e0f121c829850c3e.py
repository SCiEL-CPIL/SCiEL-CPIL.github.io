"""Maximum Independent Set on conflict graph for admissible tuples in [0, 5000].

We model the problem as: vertices are integers 0..5000. Two vertices conflict
if they "complete" a full residue class modulo some prime p <= 5000 in the sense
that for every prime q <= current_size, at least one residue class mod q is missed.

Actually, the admissibility condition is: for every prime p <= len(tuple), some
residue class mod p is missed by the tuple.

We want the largest such subset of [0, 5000].

Approach: Greedy set construction with smart ordering.
- Start empty, add points that maintain admissibility.
- Use degree-based heuristic: prioritize points that "break" more potential conflicts.

But a better framing: we want to REMOVE as few points as possible from [0,5000]
so that the result is admissible. Equivalent: find the maximum admissible subset.

Greedy approach:
- For each prime p <= 5000, we need to ensure at least one residue class mod p is empty.
- We can think of this as: choose, for each prime p, a residue class r_p to "ban",
  then remove all x in [0,5000] where x ≡ r_p (mod p) for some p.
- But we don't need to ban a class for primes p where the remaining set already misses a class.

So the problem is: choose a set of (p, r_p) pairs (one per prime p we choose to enforce)
such that the total number of removed points is minimized, and for every prime q <= |remaining|,
q either has an enforced ban or naturally misses a residue class.

This is complex. Let's use a greedy heuristic:
1. Start with all points [0, 5000].
2. While there exists a prime p <= n such that all residue classes mod p are present:
   - For each such "full" prime p, compute for each residue r the number of points with x % p == r.
   - The cost of banning residue r mod p is the number of points removed.
   - But banning also helps for other primes? Not directly.
   - Choose the (p, r) with minimum removal count, but also consider "degree" = how many
     full primes this ban might help resolve (indirectly, by reducing n).

Actually, let's think differently. The key insight: when we remove points, n decreases,
so primes p > n no longer need to be satisfied. So banning a residue class mod p removes
~N/p points, and if n drops below some primes, those primes are no longer constrained.

Greedy: at each step, among all full primes p <= n, pick the one where banning the least
populated residue class removes the fewest points, with tie-breaking preferring larger p
(so n drops more, potentially freeing up larger primes).

Wait, that's what the current program does essentially. Let me try a different angle.

Alternative: Precompute for each prime p and residue r, the set of points in [0,5000] with x≡r mod p.
Then use a greedy set cover / maximum independent set heuristic.

Let me try: Build the tuple by adding points one at a time, in an order that maximizes
"efficiency". Start with an empty set S. For candidate point x, check if S ∪ {x} is admissible.
But admissibility depends on |S|, which changes as we add.

This is hard to do incrementally. Let's stick with removal-based.

Improved greedy:
- Maintain counts[p][r] = number of points x in current set with x % p == r, for all p <= 5000.
- A prime p is "full" if min_r counts[p][r] > 0 and p <= n.
- At each step, for each full prime p, consider banning the least-populated residue r*.
  Cost = counts[p][r*].
  Benefit: n decreases by cost, so primes in (n - cost, n] are freed.
  Score = cost - λ * (number of freed primes), where λ is a weight.
- Pick the (p, r*) with best score.

Let me implement this carefully with numpy for speed.
"""

import numpy as np

N = 5000


def _primes_up_to(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    primes = _primes_up_to(N)
    n_primes = len(primes)
    
    # Initial set: all points 0..N
    points = np.arange(N + 1, dtype=np.int64)
    n = len(points)
    
    # For each prime, maintain count of each residue class
    # counts[p_idx][r] = number of points x in current set with x % primes[p_idx] == r
    counts = {}
    for idx, p in enumerate(primes):
        c = np.zeros(p, dtype=np.int32)
        for x in points:
            c[x % p] += 1
        counts[idx] = c
    
    # is_full[p_idx] = True if min(counts[p_idx]) > 0
    is_full = np.zeros(n_primes, dtype=bool)
    for idx, p in enumerate(primes):
        if p <= n:
            is_full[idx] = (counts[idx].min() > 0)
    
    # Map prime value to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    while True:
        n = len(points)
        if n == 0:
            break
        
        # Find full primes p <= n
        full_indices = []
        for idx, p in enumerate(primes):
            if p <= n and is_full[idx]:
                full_indices.append(idx)
        
        if not full_indices:
            break
        
        best_score = float('inf')
        best_pidx = None
        best_r = None
        best_cost = None
        
        for pidx in full_indices:
            p = primes[pidx]
            c = counts[pidx]
            min_r = c.argmin()
            min_count = c[min_r]
            
            if min_count == 0:
                # This shouldn't happen since is_full is True
                continue
            
            new_n = n - min_count
            # Count how many full primes q are in (new_n, n]
            freed = 0
            for qidx in full_indices:
                q = primes[qidx]
                if new_n < q <= n:
                    freed += 1
            
            # Score: minimize cost, but reward freeing primes
            # Each freed prime means we no longer need to satisfy it
            score = min_count - freed * 2.0
            
            if score < best_score:
                best_score = score
                best_pidx = pidx
                best_r = min_r
                best_cost = min_count
        
        if best_pidx is None:
            break
        
        p = primes[best_pidx]
        r = best_r
        
        # Remove all points x where x % p == r
        mask = (points % p) != r
        to_remove = points[~mask]
        points = points[mask]
        
        new_n = len(points)
        
        # Update counts for all primes q <= new_n (primes > new_n don't matter anymore)
        for x in to_remove:
            for qidx, q in enumerate(primes):
                if q > new_n:
                    break
                idx_in_q = x % q
                counts[qidx][idx_in_q] -= 1
                if counts[qidx][idx_in_q] == 0 and is_full[qidx]:
                    is_full[qidx] = False
        
        # Also, primes that were not full but might become relevant if n changes?
        # n only decreases, so primes > new_n are irrelevant.
        # Primes <= new_n that were not full remain not full (we only removed points).
    
    return points.tolist()
