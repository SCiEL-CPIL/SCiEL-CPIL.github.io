"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    N = 256
    # Candidate pool: numbers 2..256
    candidates = list(range(2, N + 1))
    
    # Precompute prime factorization for each number
    # For each number, store the set of prime factors
    prime_factors = {}
    for x in range(2, N + 1):
        factors = set()
        n = x
        p = 2
        while p * p <= n:
            while n % p == 0:
                factors.add(p)
                n //= p
            p += 1
        if n > 1:
            factors.add(n)
        prime_factors[x] = factors
    
    # For each a, determine which pairs (b,c) with b,c in candidates, b != c, 
    # and a does not divide b*c.
    # But we want: for every triple (a,b,c) of distinct members, a does not divide b*c.
    # So if we include a, b, c, we need a ∤ b*c.
    
    # Strategy: Use a greedy approach with backtracking or simply try to include
    # as many as possible. Since N=256 is small, we can use a more direct approach.
    
    # Key insight: If a is composite and has a prime factor p, then if both b and c
    # are divisible by p, and a's prime power constraints are satisfied, a might divide b*c.
    # Specifically, a | b*c iff for every prime p^k || a, the sum of exponents of p in b and c >= k.
    
    # Let's use a different approach: iterate and maintain a set, checking constraints incrementally.
    # But this is O(n^3) which is 256^3 ~ 16 million, might be too slow.
    
    # Better: Precompute for each a, the set of "bad" pairs (b,c) such that a | b*c.
    # Then we need to select a subset S such that for all a in S, there is no pair (b,c) in S with b != c and a | b*c.
    
    # Let's build a graph: vertices are candidates. There's an edge between b and c if there exists
    # some a that we might include such that a | b*c. But this is tricky because a is also a vertex.
    
    # Alternative: For each a, define forbidden pairs. When we include a, we must exclude all pairs (b,c)
    # where a | b*c. But b and c must also be in the set.
    
    # Let's try a greedy heuristic: sort candidates and add them one by one if they don't violate constraints.
    # But greedy might not be optimal.
    
    # Given the small size (256 elements), let's try a more systematic approach.
    # We can model this as: find the largest subset S such that for all distinct a,b,c in S, a ∤ b*c.
    
    # Let's precompute: for each a, and for each pair (b,c) with b != c, whether a | b*c.
    # Then, we need to find the largest S such that for all a in S, and all distinct b,c in S, a ∤ b*c.
    
    # This is equivalent to: S is a valid set iff for all a in S, the set S \ {a} contains no pair (b,c) 
    # such that a | b*c.
    
    # Let's define: for each a, bad_pairs[a] = set of (b,c) with b != c, b,c in 2..256, and a | b*c.
    # Then S is valid iff for all a in S, (S \ {a}) contains no pair from bad_pairs[a].
    
    # This is still complex. Let's try a simple greedy first and see how it does.
    
    # Greedy: sort by some criterion (e.g., number of constraints it imposes), add if possible.
    # A number a imposes constraints: for every other pair (b,c) in the set, a must not divide b*c.
    # So numbers with fewer prime factors might be "safer" to include?
    
    # Actually, prime numbers are safe: if a is prime, a | b*c iff a|b or a|c.
    # So if we include a prime p, we just can't include two multiples of p.
    
    # Let's try: include all primes first, then add composites if possible.
    
    result = []
    for a in candidates:
        # Check if adding a to result violates any constraint
        valid = True
        # For each existing pair (b,c) in result, check if a | b*c
        # Also, for each existing element b in result, and a itself as one of the triple members:
        # We need to check triples where a is the divisor, and also where a is b or c.
        
        # Actually, the constraint is symmetric in a sense: for every triple of distinct members (x,y,z),
        # x does not divide y*z. So if we add a, we need to check:
        # 1. For every pair (b,c) in result (b != c), a does not divide b*c. (a is the divisor)
        # 2. For every pair (b,c) in result (b != c), b does not divide a*c. (b is the divisor, a is one of the product)
        # 3. For every pair (b,c) in result (b != c), c does not divide a*b. (c is the divisor)
        
        # But wait, cases 2 and 3 are already checked when b and c were added, IF a was not in the set.
        # No, when b was added, a was not in the set, so the triple (b, a, c) was not checked.
        # So we need to check all triples involving a.
        
        for i, b in enumerate(result):
            for c in result[i+1:]:
                # Triple (a, b, c): check a | b*c, b | a*c, c | a*b
                if a * b * c == 0:  # won't happen since all >= 2
                    valid = False
                    break
                if (b * c) % a == 0:
                    valid = False
                    break
                if (a * c) % b == 0:
                    valid = False
                    break
                if (a * b) % c == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            result.append(a)
        
        if time.time() > deadline:
            break
    
    # Ensure we return a list of Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
