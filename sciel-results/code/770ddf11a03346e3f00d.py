"""Rewritten program using MODEL-AND-OPTIMIZE approach."""
import time

N = 100
DEADLINE_LIMIT = 145  # Safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Model-and-Optimize approach for OneMax (100 bits).
    
    Model: Each bit $i$ is generated with probability $p_i$ of being '1'.
    Objective: Maximize the actual count of ones $J = \sum_{i=1}^N b_i$.
    
    Optimization:
    The optimal parameters are $p_i = 1.0$ for all $i$, which guarantees 
    $b_i = 1$ for all $i$, maximizing the objective to $N$.
    
    Implementation:
    Directly construct the optimal string '1' * N.
    """
    # The optimal solution is a string of N ones.
    # This corresponds to setting all model parameters p_i to 1.0.
    return '1' * N
# EVOLVE-BLOCK-END
