"""Cap set greedy construction with a refined static priority heuristic."""
import time
import math
import random
from itertools import product


def priority(vector, n):
    """
    Static priority function for cap set greedy construction.
    
    Primary: Penalize vectors with many 1s (middle value, more likely to be AP midpoints).
    Secondary: Balance of digit counts - prefer vectors with more 0s and 2s.
    Tertiary: Position-weighted sum to break ties structurally.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Heavily penalize 1s since they're the "middle" value in F_3
    # Vectors with more 1s are more likely to be midpoints of 3-term APs
    one_penalty = c1 * c1 * 100.0
    
    # Reward balance between 0s and 2s (extreme values)
    extreme_balance = abs(c0 - c2)
    balance_reward = -(extreme_balance ** 2) * 10.0
    
    # Position-weighted sum: deterministic tie-breaker
    pos_weighted = sum(vector[i] * (i + 1) for i in range(n))
    
    # Combine: one_penalty dominates, then balance, then position
    return -one_penalty + balance_reward + pos_weighted * 0.1
