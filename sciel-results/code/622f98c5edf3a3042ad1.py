def priority(vector, n):
    # Count zeros
    zeros = sum(1 for x in vector if x == 0)
    # Calculate weighted penalty for non-zeros
    # Bias more strongly towards 1s than 2s to exploit {0,1}^n structure
    penalty = sum(1 if x == 1 else 4 if x == 2 else 0 for x in vector)
    
    # Primary: more zeros is better
    # Secondary: lower penalty is better
    # Weight ensures zeros dominate
    weight = 10 * n + 1
    score = zeros * weight - penalty
    return float(score)
