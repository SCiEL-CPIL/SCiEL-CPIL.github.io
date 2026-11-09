# EVOLVE-BLOCK-START
import time
import sys

# Strategy: Hierarchical Subspace Greedy Priority
#
# Rationale:
# The previous "Hash Scrambling" strategy treated all vectors as peers, 
# relying on random ordering to avoid local conflicts. However, cap sets 
# in the subspace {0,1}^n (vectors with no 2s) are structurally simpler 
# and can be packed more densely before introducing the complexity of the 
# digit '2'. 
#
# By prioritizing vectors with fewer '2's, we force the greedy algorithm 
# to first construct a large cap set within the {0,1}^n subspace. This 
# "core" set is robust. Once the core is maximized (or saturated), the 
# algorithm moves to vectors with one '2', then two, etc. 
#
# This hierarchical approach prevents the greedy algorithm from "wasting" 
# potential slots in the dense {0,1}^n subspace by prematurely adding 
# vectors with 2s that might conflict with future {0,1}^n vectors or 
# each other. It leverages the geometric embedding of {0,1}^n in {0,1,2}^n.

_START_TIME = time.time()
_TIMEOUT = 300.0  # Safety margin for the 330s limit

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def priority(vector, n):
    # Check time limit
    if _time_left() < 1.0:
        # If we are out of time, return a neutral score to stop processing
        # quickly if the evaluator is in a tight loop, though typically 
        # priority is called O(3^n) times total.
        return 0.0

    # 1. Primary Key: Minimize the number of 2s.
    # Vectors with 0 twos are in {0,1}^n.
    # Vectors with 1 two are in {0,1,2}^n \ {0,1}^n.
    # We want higher priority for fewer 2s.
    count_2s = sum(1 for v in vector if v == 2)
    
    # We assign a large negative penalty for each '2'.
    # This ensures all vectors with k twos are processed before any with k+1 twos.
    primary_score = -100.0 * count_2s

    # 2. Secondary Key: Within the same count of 2s, we want to avoid 
    #   clustering. A simple heuristic is to prefer vectors with a "balanced" 
    #   number of 1s, as extreme vectors (all 0s or all 1s) tend to create 
    #   many midpoints.
    count_1s = sum(1 for v in vector if v == 1)
    
    # Ideal count of 1s is n/2 for {0,1}^n subspace.
    # For subspaces with 2s, the distribution is more complex, but 
    # balance is generally a good proxy for "centrality" and lower conflict.
    target_1s = n / 2.0
    deviation_1s = abs(count_1s - target_1s)
    
    # Small penalty for deviation from balance.
    secondary_score = -0.1 * deviation_1s

    # 3. Tertiary Key: Deterministic hash for tie-breaking.
    # This ensures a consistent order for vectors that are identical in 
    # primary and secondary scores.
    h = 0
    MOD = 2147483647
    BASE = 31
    for i, v in enumerate(vector):
        h = (h * BASE + v) % MOD
    tertiary_score = h / MOD * 0.01  # Very small weight

    final_score = primary_score + secondary_score + tertiary_score

    return final_score

# EVOLVE-BLOCK-END
