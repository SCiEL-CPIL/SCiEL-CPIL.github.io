"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import Counter

def _calculate_entropy_priority(vector, n):
    """
    Calculate priority based on the Shannon entropy of the digit distribution.
    Vectors with more balanced digit distributions (higher entropy) are prioritized.
    This serves as a proxy for 'local independence' or 'structural neutrality',
    as vectors with skewed distributions are more likely to lie in structured subspaces
    that facilitate 3-term arithmetic progressions.
    """
    if n == 0:
        return 0.0
    
    counts = Counter(vector)
    total = len(vector)
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
            
    # Normalize entropy to [0, 1] where 1 is max entropy (perfectly balanced)
    max_entropy = math.log(3)
    normalized_entropy = entropy / max_entropy
    
    # Higher priority for higher entropy
    return normalized_entropy

def priority(vector, n):
    """
    Entry point: returns a priority value for a vector.
    The evaluator uses this to greedily add vectors in descending priority order.
    We return a value reflecting the entropy of the coordinate distribution.
    """
    return _calculate_entropy_priority(vector, n)
# EVOLVE-BLOCK-END
