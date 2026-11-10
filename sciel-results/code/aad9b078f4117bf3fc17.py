def construct():
    N = 123457
    
    # Sieve primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # We'll try increasing B and check if there's a pair
    # But we can be smarter: start from small B
    
    def generate_smooth(B, N):
        """Generate all B-smooth numbers up to N."""
        # Get primes <= B
        p_list = [p for p in primes if p <= B]
        smooth = set([1])
        # Use BFS/DFS to generate
        # Start with 1, multiply by each prime as long as <= N
        # Use a list and expand
        smooth_list = [1]
        i = 0
        while i < len(smooth_list):
            x = smooth_list[i]
            i += 1
            for p in p_list:
                y = x * p
                if y <= N:
                    if y not in smooth:
                        smooth.add(y)
                        smooth_list.append(y)
                else:
                    break
        return smooth
    
    # Try B from 2 upwards
    for B in range(2, N+1):
        if B > N:
            break
        smooth_set = generate_smooth(B, N)
        found = False
        for a in smooth_set:
            b = N - a
            if b >= 1 and b in smooth_set:
                # Both are B-smooth
                # We found a pair with max prime factor <= B
                # But we need to verify that B is indeed the max prime factor
                # Actually, since we're going in order, the first B that works is optimal
                return [a, b]
    
    # Fallback
    return [1, N-1]
