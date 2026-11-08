import math

def construct():
    # Parameters for the construction
    # Using D = {0, 1, m} with base b = 2m + 1
    # This ensures |U+U| = 6^k and |U-U| = (2m+1)^k
    # Score approx 1 + log((2m+1)/6) / log(2m+1) -> 2 as m -> inf
    
    m = 10**7
    k = 14
    b = 2 * m + 1
    
    # Generate U
    # U consists of all sums sum_{i=0}^{k-1} d_i * b^i where d_i in {0, 1, m}
    U = [0]
    power = 1
    digits = [0, 1, m]
    
    for _ in range(k):
        new_U = []
        # Optimization: local variable lookup
        p = power
        d0, d1, dm = digits
        for x in U:
            new_U.append(x)
            new_U.append(x + p)
            new_U.append(x + dm * p)
        U = new_U
        power *= b
        
    return U
