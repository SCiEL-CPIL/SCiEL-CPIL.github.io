def construct():
    # The problem asks for a subset of [1, 512] with pairwise LCM <= 512.
    # A sufficient condition is that all elements divide a common number N <= 512.
    # To maximize the size, we choose N <= 512 with the maximum number of divisors.
    # Numbers <= 512 with the most divisors (24) include 360, 420, 480.
    # We return the divisors of 420.
    
    n = 420
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    
    # Ensure all are within [1, 512] (they are, since max is 420)
    # And non-empty
    return divisors
