def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    s = sum(vector) % 3
    
    # Calculate Lag-1 Autocorrelation Deviation
    # Expected sum of v_i * v_{i+1} for random v in {0,1,2} is n * (1/3 * 1/3) = n/9
    # We penalize deviation from this expected value to favor "random-like" (low autocorrelation) vectors
    if n > 1:
        ac_sum = 0
        for i in range(n - 1):
            ac_sum += vector[i] * vector[i + 1]
        expected_ac = (n - 1) / 9.0
        ac_dev = (ac_sum - expected_ac) ** 2
    else:
        ac_dev = 0.0

    # Fast deterministic tie-breaker
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) & 0x7FFFFFFF
    
    h_norm = h / 0x7FFFFFFF
    
    sum_bonus = {1: 2.0, 2: 1.0, 0: 0.0}[s]
    
    # Combine:
    # Primary: Global Balance (-dev)
    # Secondary: Local Randomness (-ac_dev)
    # Tertiary: Sum Mod 3 Bonus
    # Quaternary: Hash
    
    # Scale ac_dev. 
    # Max ac_dev: if all 0, ac_sum=0, exp~n/9, dev~n^2/81.
    # Max dev (balance): if all 0, dev = (n-n/3)^2 + (0-n/3)^2 + (0-n/3)^2 = 4n^2/9 + n^2/9 + n^2/9 = 6n^2/9 = 2n^2/3.
    # ac_dev is smaller in magnitude.
    # Let's weight ac_dev by 0.1 to make it a secondary effect.
    
    return -dev - 0.1 * ac_dev + sum_bonus * 0.01 + 1e-9 * h_norm
