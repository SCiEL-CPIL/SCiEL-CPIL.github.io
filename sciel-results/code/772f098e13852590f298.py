"""Rewritten program using a maximal independent set approach on the conflict graph."""

def construct():
    N = 256
    # Build list of candidates: all integers in 2..256
    candidates = list(range(2, N + 1))
    
    # We need: for every three distinct a, b, c in S, a does not divide b*c.
    # Equivalently, for each a in S, for every distinct b, c in S (b != c, both != a),
    # a must NOT divide b*c.
    #
    # This is a constraint satisfaction / optimization problem.
    # Key insight: if we pick only numbers that are "large enough" relative to their divisors,
    # or numbers whose prime factors are not all present in pairs, we can avoid violations.
    #
    # A different approach: greedy maximal independent set on a graph where edges represent
    # "conflict" - but the conflict is a 3-way hyperedge, not a 2-way edge.
    #
    # Let me think about what constraints actually matter.
    # For a to divide b*c, the prime factorization of a must be "covered" by b*c.
    #
    # Strategy: Use a greedy approach. Start with all numbers, and iteratively remove
    # numbers that cause the most conflicts, or use a heuristic to find a large valid set.
    #
    # Alternative strategy: 
    # - Numbers > N/2 = 128: for any a > 128, b*c must be >= a for a|b*c. 
    #   Since b,c are at most 256, b*c can be up to 256*255. 
    #   But more importantly, if a > 128 and a is composite, say a = p*q with p,q >= 2,
    #   then for a | b*c, we need b*c to be a multiple of a. 
    #   The smallest multiple of a that can be formed as a product of two distinct numbers
    #   from the set... this is complex.
    #
    # Let me try a different tack: use the fact that if all numbers in S are square-free
    # and share no common prime factor with any other number's "complement", we're safe.
    #
    # Actually, let me try a direct greedy approach:
    # 1. Sort candidates by some heuristic (e.g., by number of "dangerous" pairs)
    # 2. Greedily add numbers if they don't violate the constraint with already-chosen numbers.
    
    # First, let's understand the constraint better with a helper function.
    # Check if adding 'new_elem' to existing set 'S' would violate the constraint.
    # For each pair (b, c) in S with b != c, check if new_elem divides b*c.
    # For each b in S, check if new_elem divides b * new_elem... no, b and c must be distinct from each other AND from a.
    # Wait, re-reading: "for every THREE DISTINCT members a,b,c" - so a, b, c are all distinct.
    # So when checking if adding x to S is safe:
    #   For every pair (b, c) in S where b != c: check x does not divide b*c.
    #   Also, for every b in S: check b does not divide x * (any other element in S).
    #   And for every pair (b,c) in S: check b does not divide c * x, and c does not divide b * x.
    
    # This is getting complex. Let me use a simpler but effective approach:
    # Start with an empty set, and greedily add numbers in order of "safety".
    
    # Heuristic: numbers with fewer small prime factors are safer.
    # Let's sort by number of divisors or by smallest prime factor.
    
    def num_divisors(n):
        count = 0
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 2
                while n % i == 0:
                    n //= i
            i += 1
        if n > 1:
            count += 1
        return count
    
    def smallest_prime_factor(n):
        if n % 2 == 0:
            return 2
        i = 3
        while i * i <= n:
            if n % i == 0:
                return i
            i += 2
        return n
    
    # Sort candidates: prefer numbers with larger smallest prime factor (less likely to divide products)
    # and fewer divisors
    sorted_candidates = sorted(candidates, key=lambda x: (-smallest_prime_factor(x), -num_divisors(x)))
    
    S = []
    
    def is_safe(candidate, S):
        """Check if candidate can be added to S without violating the constraint."""
        # For every pair (b, c) in S with b != c: candidate must not divide b*c
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                b, c = S[i], S[j]
                if (b * c) % candidate == 0:
                    return False
        
        # For every b in S: b must not divide candidate * c for any c in S, c != b
        # And candidate must not divide b * c for any pair (already checked above)
        # Also: for each b in S, check if b divides candidate * c for some c in S, c != b, c != candidate
        for i in range(len(S)):
            b = S[i]
            for j in range(len(S)):
                if i == j:
                    continue
                c = S[j]
                if (candidate * c) % b == 0:
                    return False
            # Also check: does b divide candidate * candidate? No, b and c must be distinct.
            # But we also need: for the new element as 'a', we already checked above.
            # For new element as 'b' or 'c': need to check if any existing a divides new_elem * other_elem
        return True
    
    # Greedy: add each candidate if safe
    for cand in sorted_candidates:
        if is_safe(cand, S):
            S.append(cand)
    
    # Return as list of Python ints
    return [int(x) for x in S]
