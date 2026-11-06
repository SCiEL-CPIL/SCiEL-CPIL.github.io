"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.
    Higher priority = added earlier in greedy process.
    
    Key insight: In {0,1,2}^n, a 3-term AP x,y,z satisfies x+z=2y element-wise.
    - y[i]=0 requires x[i]=z[i]=0
    - y[i]=1 requires {x[i],z[i]}={0,2}  
    - y[i]=2 requires x[i]=z[i]=2
    
    So vectors with many 1s are more likely to be the middle of an AP.
    We should prioritize vectors with fewer 1s.
    
    Also, vectors with many equal adjacent coordinates are structurally risky.
    """
    vec = tuple(vector)
    
    # Count 1s - the critical factor for being a middle of AP
    num_ones = sum(1 for x in vec if x == 1)
    
    # Adjacent equal penalty - structural risk indicator
    adj_equal = 0
    for i in range(n - 1):
        if vec[i] == vec[i + 1]:
            adj_equal += 1
    
    # Primary: fewer 1s is better (less likely to form AP as middle)
    # Score from 1-count: range [0, 1], higher = fewer 1s
    one_score = 1.0 - (num_ones / max(n, 1))
    
    # Secondary: fewer adjacent equals is better
    adj_score = 1.0 - (adj_equal / max(n - 1, 1))
    
    # Tertiary: balance between 0s and 2s (avoid all-0 or all-2 extremes)
    num_zeros = sum(1 for x in vec if x == 0)
    num_twos = sum(1 for x in vec if x == 2)
    balance = 1.0 - abs(num_zeros - num_twos) / max(n, 1)
    
    # Weighted combination
    priority_val = 0.5 * one_score + 0.3 * adj_score + 0.2 * balance
    
    return priority_val
# EVOLVE-BLOCK-END
