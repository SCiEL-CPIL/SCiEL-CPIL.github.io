"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n.
    We want to encourage 'spread out' vectors to be added first.
    
    Strategy:
    1. Balance: Prefer vectors with counts of 0, 1, 2 close to n/3.
       This avoids corners (like all 0s) and edges.
    2. Variance/Entropy: Prefer vectors that are not 'pure' in any coordinate direction?
       Actually, for cap sets, we want to avoid arithmetic progressions.
       A simple heuristic that works well in greedy constructions is to prefer
       vectors that are 'far' from existing ones, but we don't have access to the set here.
       
       However, the prompt mentions: "A non-trivial O(n) structural ranking (variance/balance/low-density + deterministic tie-break) reorders the 3^n greedy insertions so accepted vectors stay spread out, delaying corner formation and letting the greedy process accept more vectors, directly raising cap set size. Measured: 1 -> 1.129."
       
       Let's implement a score based on:
       - Balance of symbols (0, 1, 2). Ideally n/3 each.
       - Avoiding 'monochromatic' or 'bichromatic' patterns that might form small progressions.
       
       Let's calculate:
       c0, c1, c2 = counts of 0, 1, 2.
       Deviation from uniform: d = (c0 - n/3)^2 + (c1 - n/3)^2 + (c2 - n/3)^2
       Lower deviation is better (higher priority).
       
       Also, consider the 'energy' of the vector. 
       Maybe we can use a hash-like function to break ties deterministically but randomly?
       No, deterministic tie-break is preferred for reproducibility, but a 'random' looking tie-break based on vector content might help avoid structured failures.
       
       Let's try:
       score = -deviation + small_noise_based_on_vector
       
       Actually, let's look at the known good heuristic for greedy cap sets.
       Often, maximizing the minimum distance or using a greedy algorithm with a good ordering is key.
       
       Let's use:
       1. Balance score: How close to n/3 each symbol count is.
       2. Sum of squares of coordinates? No.
       
       Let's try a simple effective heuristic:
       - Count 0s, 1s, 2s.
       - Penalize if any count is 0 or n (corners/edges).
       - Prefer balanced counts.
       
       Let's compute a 'balance' score:
       b = 1.0 / (1.0 + abs(c0 - n/3) + abs(c1 - n/3) + abs(c2 - n/3))
       
       To break ties and add variety, we can add a small term based on the vector itself.
       e.g., sum(i * v[i] for i in range(n)) mod something?
       
       Let's combine:
       base = - (abs(c0 - n/3) + abs(c1 - n/3) + abs(c2 - n/3))
       tie_break = sum((i+1) * v[i] for i in range(n)) * 1e-9
       
       Return base + tie_break
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
            
    # Target is n/3
    target = n / 3.0
    
    # Deviation from balance
    deviation = abs(c0 - target) + abs(c1 - target) + abs(c2 - target)
    
    # We want higher priority for lower deviation
    score = -deviation
    
    # Deterministic tie-breaker to avoid symmetry issues
    # A simple weighted sum of indices and values
    tie_break = 0.0
    for i, v in enumerate(vector):
        tie_break += (i + 1) * v
    score += tie_break * 1e-10
    
    return score

# EVOLVE-BLOCK-END
