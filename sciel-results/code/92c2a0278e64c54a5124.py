"""Cap set greedy construction with combined structural and spatial heuristics."""
import time

_START_TIME = time.time()
TIME_LIMIT = 290  # seconds, safety margin before 300s limit

def priority(vector, n):
    """
    Static priority function for greedy cap set construction.
    
    Combines:
    1. Balanced digit distribution (primary)
    2. Spatial spread of 1-positions (secondary, reduces clustering)
    3. Adjacency penalty (tertiary, penalizes local structure)
    4. Geometric lex tie-breaker (quaternary, deterministic)
    """
    # Time guard: if we're running out of time, return a simple value
    if time.time() - _START_TIME > TIME_LIMIT:
        return 0.0
    
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # Primary: Minimize deviation from balanced distribution
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Secondary: Spatial spread of 1-positions (sum of pairwise distances)
    ones_positions = [i for i, v in enumerate(vector) if v == 1]
    spatial_spread = 0.0
    if len(ones_positions) >= 2:
        for i in range(len(ones_positions)):
            for j in range(i + 1, len(ones_positions)):
                spatial_spread += abs(ones_positions[i] - ones_positions[j])
    max_possible_spread = len(ones_positions) * (len(ones_positions) - 1) * (n - 1) / 2.0
    spread_norm = spatial_spread / max_possible_spread if max_possible_spread > 0 else 0.0
    
    # Tertiary: Adjacency penalty (penalize consecutive identical digits)
    adj_penalty = 0
    for i in range(n - 1):
        if vector[i] == vector[i + 1]:
            adj_penalty += 1
    
    # Quaternary: Geometric lex tie-breaker (base 3)
    lex_val = 0.0
    for i in range(n):
        lex_val = lex_val * 3.0 + vector[i]
    
    # Combine with carefully chosen weights:
    # - penalty dominates (scale by 100000)
    # - spread is significant (scale by 1000)  
    # - adjacency is moderate (scale by 100)
    # - lex breaks ties (small scale)
    return -penalty * 100000.0 + spread_norm * 1000.0 - adj_penalty * 100.0 + lex_val * 0.01
