"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 10.0

def priority(vector, n):
    """
    Prioritize vectors to maximize cap set size in greedy insertion.
    
    Strategy: Vectors with more 2s tend to be 'farther' in F_3^n and may
    create fewer 3-term progressions with earlier-added vectors. We also
    add a small perturbation based on position to break ties in a structured way.
    """
    # Count occurrences of each symbol
    count_0 = vector.count(0)
    count_1 = vector.count(1)
    count_2 = vector.count(2)
    
    # Primary: prefer vectors with more 2s (they're "opposite" to 0s)
    # This tends to spread the set in F_3^n
    primary = count_2 * 1000.0
    
    # Secondary: prefer balanced vectors (closer to n/3 each)
    balance = abs(count_0 - n/3.0) + abs(count_1 - n/3.0) + abs(count_2 - n/3.0)
    secondary = -balance * 100.0
    
    # Tertiary: lexicographic tie-break using weighted sum
    tertiary = sum(vector[i] * (n - i) for i in range(n)) * 0.1
    
    return primary + secondary + tertiary
# EVOLVE-BLOCK-END
