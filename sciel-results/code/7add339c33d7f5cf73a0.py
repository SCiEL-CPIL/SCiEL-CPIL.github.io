"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    from collections import defaultdict

    # Increased limit to find N with more representations
    # 5*10^7 allows ~22000 powerful numbers, ~240M pairs, feasible in time
    LIMIT = 5 * 10**7
    
    powerful = set()
    # b must be square-free and b^3 <= LIMIT
    max_b = int(LIMIT ** (1/3))
    
    # Precompute squares of primes for faster square-free check
    def get_prime_squares(limit):
        is_prime = [True] * (int(math.isqrt(limit)) + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, int(math.isqrt(limit)) + 1, i):
                    is_prime[j] = False
        primes = [i for i, p in enumerate(is_prime) if p]
        return [p*p for p in primes]

    prime_squares = get_prime_squares(max_b)
    
    def is_square_free(n):
        for ps in prime_squares:
            if ps > n:
                break
            if n % ps == 0:
                return False
        return True

    # Generate all powerful numbers up to LIMIT
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b3 = b ** 3
        # a^2 <= LIMIT / b3
        max_a = int(math.isqrt(LIMIT // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= LIMIT:
                powerful.add(val)
                
    P = sorted(powerful)
    
    # Count representations for each sum N = a + b
    counts = defaultdict(int)
    n = len(P)
    
    # Optimize loop: use local variables
    P_list = P
    counts_dict = counts
    
    for i in range(n):
        a = P_list[i]
        # We only need a <= b, so j starts from i
        for j in range(i, n):
            b = P_list[j]
            s = a + b
            counts_dict[s] += 1

    # Find the N with the maximum count
    best_N = 0
    max_count = 0
    for s, c in counts_dict.items():
        if c > max_count:
            max_count = c
            best_N = s

    # Retrieve the pairs for the best N
    pairs = []
    powerful_set = powerful # Use set for O(1) lookup
    for a in P:
        b = best_N - a
        if a <= b and b in powerful_set:
            pairs.append([a, b])
            
    # Sort pairs for consistency (though not strictly required by interface, good practice)
    pairs.sort()

    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
