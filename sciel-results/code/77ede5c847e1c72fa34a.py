"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# We need to keep the same interface: priority(vector, n) -> float
# But the actual computation happens when the evaluator calls priority for each vector.
# The evaluator adds vectors greedily in descending priority.
# We need to return a priority such that the greedy order leads to a large cap set.

# Key insight: A cap set in {0,1,2}^n is a set with no 3-term arithmetic progression.
# In {0,1,2}^n, a 3-AP is x, x+d, x+2d where d is in {0,1,2}^n and d != 0.
# Actually, in {0,1,2}^n, the only way to have a 3-AP is if for each coordinate, the three values form an AP in {0,1,2}.
# The APs in {0,1,2} are: (0,0,0), (0,1,2), (1,0,1), (2,1,0), etc.
# More precisely, a 3-AP in {0,1,2}^n is a set {a, a+d, a+2d} where all coordinates are in {0,1,2}.

# Strategy: Use a priority based on a known good cap set construction.
# One approach: Use the greedy algorithm with a smart priority function.
# Another approach: Precompute a large cap set and assign high priority to its members.

# Since we have 300 seconds, we can do some precomputation.
# But the priority function is called for each vector, and we don't know n in advance until called.

# Let's think about the structure. For small n, we can precompute.
# For n=1: cap set size is 2 (any 2 elements, since 3 elements form an AP)
# For n=2: max cap set size is 4
# For n=3: max cap set size is 9
# For n=4: max cap set size is 20
# For n=5: max cap set size is 44
# For n=6: max cap set size is 92
# For n=7: max cap set size is 194
# For n=8: max cap set size is 400

# The greedy approach with a good priority can do well.
# A common heuristic: prioritize vectors with fewer 1s, or use a linear ordering.

# Let's try a different approach: Use a priority based on the sum of coordinates,
# or use a hash-based priority that correlates with being in a large cap set.

# Actually, let's use a greedy algorithm with a priority that favors vectors that are "far apart"
# from each other in some sense.

# For now, let's implement a simple but effective priority:
# Assign priority based on a weighted sum where weights are chosen to break ties in a way
# that avoids 3-APs.

# A known good strategy: Use the lexicographic order but with a twist.
# Or: Use a priority based on the number of 2s, then 0s, then 1s.

# Let's try: priority = -sum(vector) + small perturbation based on position.
# This favors vectors with more 2s (or fewer 0s).

# Actually, let me think about this differently. The evaluator adds vectors in descending priority.
# So we want high-priority vectors to be added first, and they should form a cap set.

# A simple effective strategy: prioritize vectors with more 2s.
# Because vectors with many 2s are "far" from vectors with many 0s.

def priority(vector, n):
    # Prioritize vectors with more 2s, then more 0s, then fewer 1s.
    # This creates a structure that tends to avoid 3-APs.
    count_2 = sum(1 for x in vector if x == 2)
    count_0 = sum(1 for x in vector if x == 0)
    count_1 = sum(1 for x in vector if x == 1)
    
    # High priority for more 2s, then more 0s
    # Use a large base to ensure count_2 dominates, then count_0
    priority_val = count_2 * (n + 1) + count_0
    return float(priority_val)

# EVOLVE-BLOCK-END
