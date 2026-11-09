"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    def prime_factors(n):
        """Return set of prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    def count_distinct_primes(arr):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= prime_factors(arr[i] + arr[j])
        return len(primes)

    def is_smooth(n, prime_set):
        """Check if n has only prime factors in prime_set."""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    # Strategy 1: Powers of 2 (structural)
    powers_of_2 = [2**i for i in range(12)]
    assert all(x <= 10000 for x in powers_of_2)
    assert len(set(powers_of_2)) == 12

    best_result = powers_of_2
    best_count = count_distinct_primes(powers_of_2)

    # Strategy 2: Constraint satisfaction - find 12 numbers where all pairwise sums are 7-smooth
    # This guarantees at most 4 distinct primes {2,3,5,7}
    target_primes = {2, 3, 5, 7}

    # Precompute which numbers are "compatible" with each other
    # Two numbers a, b are compatible if a+b is 7-smooth
    # We need to find a clique of size 12 in this compatibility graph

    # First, precompute for each number 1..10000 whether it's "useful"
    # A number x is useful if there exist at least 11 other numbers y such that x+y is 7-smooth
    # But that's expensive. Instead, let's build the graph incrementally.

    # Better approach: for each pair (a,b) with a<b, check if a+b is 7-smooth
    # Store adjacency as a set for each node

    # To make this feasible, let's first filter: which numbers can participate?
    # A number x can be in a valid set only if there are enough numbers y such that x+y is 7-smooth.
    # Let's just build the full compatibility structure and do backtracking.

    # Precompute: for each x in 1..10000, the set of y > x such that x+y is 7-smooth
    # This is O(10000^2) which is 100 million - too slow.

    # Alternative: generate all 7-smooth numbers up to 20000, then for each smooth s,
    # all pairs (a,b) with a+b=s and 1<=a<b<=10000 are compatible.

    # 7-smooth numbers up to 20000:
    smooth_numbers = set()
    def gen_smooth(max_val):
        """Generate all 7-smooth numbers up to max_val."""
        powers2 = [2**i for i in range(15) if 2**i <= max_val]
        powers3 = [3**i for i in range(10) if 3**i <= max_val]
        powers5 = [5**i for i in range(7) if 5**i <= max_val]
        powers7 = [7**i for i in range(6) if 7**i <= max_val]
        for a in powers2:
            for b in powers3:
                for c in powers5:
                    for d in powers7:
                        v = a * b * c * d
                        if v <= max_val and v >= 2:
                            smooth_numbers.add(v)
    gen_smooth(20000)

    # Now for each smooth sum s, generate compatible pairs
    # adj[x] = set of y such that x+y is 7-smooth
    adj = [set() for _ in range(10001)]

    for s in smooth_numbers:
        if time_exceeded():
            break
        # Pairs (a,b) with a+b=s, 1<=a<b<=10000
        # a ranges from max(1, s-10000) to min(10000, s//2)
        a_min = max(1, s - 10000)
        a_max = min(10000, s // 2)
        if a_min > a_max:
            continue
        for a in range(a_min, a_max + 1):
            b = s - a
            if b > a and b <= 10000:
                adj[a].add(b)
                adj[b].add(a)

    if not time_exceeded():
        # Backtracking to find clique of size 12
        # Use degree-based ordering: start with high-degree nodes
        # Forward checking: at each step, remaining candidates must all be connected to all selected

        # Sort numbers by degree (descending) to find cliques faster
        candidates = list(range(1, 10001))
        candidates.sort(key=lambda x: len(adj[x]), reverse=True)

        found_clique = None

        def backtrack(selected, remaining):
            """Find clique of size 12."""
            nonlocal found_clique
            if time_exceeded():
                return
            if len(selected) == 12:
                found_clique = selected[:]
                return
            need = 12 - len(selected)
            if len(remaining) < need:
                return
            for i in range(len(remaining)):
                node = remaining[i]
                # Check if node is connected to all selected
                if all(node in adj[s] for s in selected):
                    # Forward check: remaining after this node must have enough
                    # that are connected to all selected + node
                    new_remaining = []
                    for j in range(i + 1, len(remaining)):
                        rnode = remaining[j]
                        if rnode in adj[node] and all(rnode in adj[s] for s in selected):
                            new_remaining.append(rnode)
                    if len(new_remaining) >= need - 1:
                        backtrack(selected + [node], new_remaining)
                        if found_clique is not None:
                            return

        # Start backtracking with top-degree candidates
        # Limit initial candidates to top N to keep search manageable
        top_candidates = candidates[:500]  # Start with top 500 by degree
        backtrack([], top_candidates)

        if found_clique is not None:
            cnt = count_distinct_primes(found_clique)
            if cnt < best_count:
                best_count = cnt
                best_result = found_clique

    # Strategy 3: Try greedy for small prime sets as fallback
    def try_greedy_for_prime_set(prime_set, max_val=10000):
        """Greedy: build set of 12 numbers where all pairwise sums are smooth."""
        candidates = []
        for x in range(1, max_val + 1):
            if time_exceeded():
                return None
            valid = True
            for y in candidates:
                if not is_smooth(x + y, prime_set):
                    valid = False
                    break
            if valid:
                candidates.append(x)
            if len(candidates) >= 12:
                return candidates[:12]
        return None

    prime_set_candidates = [
        {2},
        {2, 3},
        {3, 5},
        {2, 3, 5},
        {5, 7},
        {2, 5},
        {3, 7},
        {2, 3, 5, 7},
    ]

    for ps in prime_set_candidates:
        if time_exceeded():
            break
        result = try_greedy_for_prime_set(ps)
        if result is not None and len(result) == 12:
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                cnt = count_distinct_primes(result)
                if cnt < best_count:
                    best_count = cnt
                    best_result = result
                if best_count <= len(ps):
                    break

    # Strategy 4: Other structured sets
    odds = list(range(1, 24, 2))[:12]
    cnt = count_distinct_primes(odds)
    if cnt < best_count:
        best_count = cnt
        best_result = odds

    mult6 = [6*(i+1) for i in range(12)]
    cnt = count_distinct_primes(mult6)
    if cnt < best_count:
        best_count = cnt
        best_result = mult6

    close = list(range(100, 112))
    cnt = count_distinct_primes(close)
    if cnt < best_count:
        best_count = cnt
        best_result = close

    small = list(range(1, 13))
    cnt = count_distinct_primes(small)
    if cnt < best_count:
        best_count = cnt
        best_result = small

    return best_result
# EVOLVE-BLOCK-END
