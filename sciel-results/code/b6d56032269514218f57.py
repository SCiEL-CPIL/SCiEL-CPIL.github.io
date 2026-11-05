import math
import heapq
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    LIMIT = 10**18
    
    def calc_score(n):
        if n < 2:
            return 0
        # Factorize n
        temp = n
        factors = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate divisors
        divs = [1]
        for p, exp in factors:
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    def estimate_coprime_transitions(factors):
        """
        Estimate the number of coprime adjacent pairs using a theoretical bound.
        For a number with prime factorization p1^e1 * p2^e2 * ... * pk^ek,
        the divisors form a k-dimensional grid. The number of coprime transitions
        is bounded by the structure of this grid.
        
        Key insight: two adjacent divisors in sorted order are coprime if and only if
        they share no prime factors. This happens when moving from one "layer" to another
        in the divisor lattice.
        
        A better estimate: use the fact that the number of coprime pairs is related to
        the number of "maximal" divisors in each prime direction.
        """
        tau = 1
        for p, e in factors:
            tau *= (e + 1)
        
        # Theoretical upper bound: in a k-dimensional grid with dimensions (e1+1, e2+1, ..., ek+1),
        # the maximum number of coprime adjacent pairs is bounded by the surface area of the grid.
        # For a 2D grid (a+1)*(b+1), the max coprime transitions is roughly 2*min(a,b) + max(a,b)
        # For higher dimensions, it's more complex.
        
        # Use a heuristic: the number of coprime transitions is at most the number of divisors
        # minus the number of divisors that share a common factor with their neighbor.
        # A simpler bound: tau - 1 (trivial upper bound)
        
        # Better: use the fact that coprime transitions occur when we cross "boundaries" 
        # in the prime exponent space. The number of such boundaries is related to the 
        # number of primes and their exponents.
        
        # For now, use tau as a proxy for search priority, but we'll refine this.
        return tau
    
    def calc_score_fast(factors):
        """
        Fast score calculation given pre-computed factors.
        """
        if not factors:
            return 0
        
        # Generate divisors
        divs = [1]
        for p, exp in factors:
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    def get_factors(n, primes_list):
        """Factorize n using given primes."""
        temp = n
        factors = []
        for p in primes_list:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        return factors
    
    # Search space: first 15 primes
    # We'll use a priority queue with a better scoring function.
    # The key insight from the directive: model as maximum independent set on complement graph.
    # This means we want factorizations where the divisor adjacency graph has sparse non-coprime edges.
    
    # Start with primorial of first 15 primes
    start_exps = [1]*15 + [0]*10
    start_n = 1
    for i, e in enumerate(start_exps):
        if e > 0:
            start_n *= primes[i] ** e
    
    if start_n > LIMIT:
        # Reduce to fit within limit
        start_exps = [1]*14 + [0]*11
        start_n = 1
        for i, e in enumerate(start_exps):
            if e > 0:
                start_n *= primes[i] ** e
    
    best_n = start_n
    best_score = calc_score(start_n)
    
    # Use a priority queue where the priority is based on a better estimate
    # of the actual coprime transition count, not just tau.
    
    def estimate_score_from_exps(exps):
        """
        Estimate the score from exponent vector without generating all divisors.
        Uses the theoretical bound: the number of coprime transitions is related to
        the structure of the divisor lattice.
        
        For a number with k distinct prime factors, the divisors form a k-dimensional grid.
        The number of coprime adjacent pairs is maximized when the grid is "balanced" 
        (exponents are similar) and when there are many distinct primes.
        
        Heuristic: score ≈ sum over primes of (e_i + 1) * (number of primes with e_j > 0, j != i)
        This counts the number of "directions" in which we can move while staying coprime.
        """
        active_primes = sum(1 for e in exps if e > 0)
        if active_primes <= 1:
            return 0
        
        # For each prime, the number of coprime transitions in that direction
        # is proportional to the number of other primes present.
        score_est = 0
        for i, e in enumerate(exps):
            if e > 0:
                # Number of other active primes
                other_active = sum(1 for j, ej in enumerate(exps) if j != i and ej > 0)
                # In the i-th prime direction, we can have at most (e_i) coprime transitions
                # for each combination of other exponents
                score_est += e * other_active
        
        # Normalize and return
        return score_est
    
    heap = []
    initial_est = estimate_score_from_exps(start_exps)
    heapq.heappush(heap, (-initial_est, -1, start_n, tuple(start_exps)))
    visited = set()
    visited.add(tuple(start_exps))
    
    # Limit search to first 15 primes
    NUM_PRIMES = 15
    
    while heap and time.time() < deadline:
        neg_est, _, n, exps = heapq.heappop(heap)
        
        # Evaluate the current node
        factors = [(primes[i], exps[i]) for i in range(NUM_PRIMES) if exps[i] > 0]
        if factors:
            s = calc_score_fast(factors)
            if s > best_score:
                best_score = s
                best_n = n
        
        # Check neighbors
        for i in range(NUM_PRIMES):
            # Try increasing exponent
            new_exps = list(exps)
            new_exps[i] += 1
            new_n = n * primes[i]
            if new_n <= LIMIT:
                key = tuple(new_exps)
                if key not in visited:
                    visited.add(key)
                    est = estimate_score_from_exps(new_exps)
                    heapq.heappush(heap, (-est, -len(visited), new_n, key))
                    
            # Try decreasing exponent
            if exps[i] > 0:
                new_exps = list(exps)
                new_exps[i] -= 1
                new_n = n // primes[i]
                if new_n >= 2:
                    key = tuple(new_exps)
                    if key not in visited:
                        visited.add(key)
                        est = estimate_score_from_exps(new_exps)
                        heapq.heappush(heap, (-est, -len(visited), new_n, key))

    return best_n
