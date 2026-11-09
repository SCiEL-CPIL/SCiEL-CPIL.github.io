def priority(vector, n):
    # Count zeros
    zeros = sum(1 for x in vector if x == 0)
    # Sum of non-zero elements
    non_zero_sum = sum(x for x in vector if x != 0)
    
    # Primary: more zeros is better
    # Secondary: lower non-zero sum is better (hypothesis)
    # We combine them into a single float.
    # To ensure zeros dominate, we use a large multiplier.
    # Max non_zero_sum is 2*n.
    # So multiplier should be > 2*n.
    # Let's use 10*n + 1 to be safe.
    
    weight = 10 * n + 1
    score = zeros * weight - non_zero_sum
    return float(score)
