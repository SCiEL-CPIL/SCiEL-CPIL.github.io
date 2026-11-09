# EVOLVE-BLOCK-START
import time
import math

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n.
    
    Strategy:
    1. Estimate collision probability using a probabilistic model.
    2. We assume that vectors with "balanced" symbol distributions are "safer" 
       or "riskier" depending on the specific structure.
    3. The seed program (score 1.21) preferred balanced vectors (max entropy).
    4. We will use a Quadratic (L2) penalty for deviation from uniform distribution, 
       which is smoother than the L1 penalty used before.
    5. We will also incorporate a "1-penalty" hypothesis: 
       In Z_3, 1 is the 'center'. Vectors with many 1s might be more prone to 
       forming APs with pairs that are 'far' in Hamming space? 
       Or conversely, they are 'central' and thus 'dense'.
       Let's try to slightly penalize 1s to see if pushing towards 0/2 dominance helps.
       Actually, let's stick to the "Max Entropy" (balanced) but use L2 norm.
       
    6. Tie-breaker: A deterministic hash to break symmetries.
    """
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Probabilistic Collision Estimate:
    # We model the "risk" as the deviation from the "ideal" distribution.
    # If the ideal is uniform (1/3, 1/3, 1/3), then Risk = ||p - u||^2.
    # We want to MINIMIZE Risk, so we MAXIMIZE -Risk.
    
    p0 = c0 / n
    p1 = c1 / n
    p2 = c2 / n
    
    # Quadratic deviation from uniform
    dev_sq = (p0 - 1/3)**2 + (p1 - 1/3)**2 + (p2 - 1/3)**2
    
    # Base score: Higher is better.
    # We invert the deviation.
    score = -dev_sq
    
    # Additional "Collision Density" Proxy:
    # Let's assume that vectors with high "1-count" are more likely to be midpoints of APs
    # involving "mixed" pairs (0,2).
    # Vectors with high "0/2-count" are midpoints of APs involving (1,2) or (0,0).
    # Let's add a small penalty for 1s to see if it improves.
    # Hypothesis: 1s are "dangerous" because they are the exact midpoint of 0 and 2.
    # If we have many 1s, we are "exposed" to 0-2 pairs.
    # Let's penalize c1 slightly.
    score -= 0.1 * (c1 / n)
    
    # Tie-breaker:
    # Use a hash to ensure unique ordering for identical scores.
    # Simple polynomial hash mod a large prime.
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) % 1000000007
    score += h * 1e-12
    
    return score

# EVOLVE-BLOCK-END
