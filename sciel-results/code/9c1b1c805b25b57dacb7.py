"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We want to find the largest subset S of {2, ..., 256}
    # such that for all distinct a, b, c in S, a does not divide b*c.
    
    # Strategy: Greedy selection based on "divisibility independence".
    # Numbers with fewer divisors (like primes) are less likely to divide products of others.
    # We sort candidates by the number of divisors (ascending) and then by value.
    # This prioritizes primes.
    
    N = 256
    candidates = list(range(2, N + 1))
    
    def num_divisors(n):
        count = 0
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
            i += 1
        return count + 1 # +1 for 1 and n (actually 1 and n are always divisors, loop finds pairs)
        # Correction: 1 and n are divisors.
        # If n is prime, loop finds none, count=0, return 2. Correct.
        # If n=4, i=2, 4%2==0, count=2, i*i==n so count=1. Return 1+1=2? No, divisors of 4 are 1,2,4 (3 divisors).
        # My function: i=2, 4%2==0, count=2. i*i==n, count=1. Return 1+1=2. Incorrect.
        # Let's use a simpler robust divisor count.
    
    def count_divs(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c

    # Sort by number of divisors, then by value
    candidates.sort(key=lambda x: (count_divs(x), x))
    
    S = []
    
    for a in candidates:
        # Check if a can be added to S
        # Condition: For all distinct b, c in S, a does not divide b*c
        # And for all b in S, b does not divide a*c for any c in S distinct from a,b?
        # The condition is symmetric in b,c but not a.
        # "For every THREE DISTINCT members a,b,c, require that a does not divide b*c."
        # This must hold for ALL permutations.
        # So if we add 'a' to S, we must check:
        # 1. For all b, c in S (distinct), a does not divide b*c.
        # 2. For all b, c in S (distinct, and c != a, b != a), b does not divide a*c.
        # 3. For all b, c in S (distinct, and c != a, b != a), c does not divide a*b.
        
        valid = True
        for b in S:
            for c in S:
                if b == c:
                    continue
                # Case 1: a is the divisor
                # We need to check if a divides b*c.
                # b and c are distinct members of current S.
                if (b * c) % a == 0:
                    valid = False
                    break
                # Case 2: b is the divisor
                # We need to check if b divides a*c.
                # a and c are distinct (c is from S, a is new).
                if (a * c) % b == 0:
                    valid = False
                    break
                # Case 3: c is the divisor
                # We need to check if c divides a*b.
                if (a * b) % c == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(a)
            
    return S
# EVOLVE-BLOCK-END
