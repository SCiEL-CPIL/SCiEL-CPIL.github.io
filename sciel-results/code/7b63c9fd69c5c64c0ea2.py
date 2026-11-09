import hashlib
import time

# We don't need time in priority, the evaluator handles the deadline.
# But we must be fast.

def priority(vector, n):
    # vector is a tuple of 0, 1, 2
    # Fast counting
    # Using count is O(n) but in C.
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = n - c0 - c1 # c2 = vector.count(2)
    
    # Balance metric: deviation from n/3
    # We want high priority for balanced vectors.
    # Deviation squared sum
    # d0 = c0 - n/3, etc.
    # score = - (d0^2 + d1^2 + d2^2)
    
    # To avoid floats if possible, or just use simple math.
    # n/3 might be float.
    # Let's use integer arithmetic for speed if possible, or just simple float.
    
    # Alternative: Hamming weight of non-zero? No.
    
    # Let's try a different heuristic: 
    # "Algebraic distance" from origin?
    # Or simply: prioritize vectors with more 1s? No.
    
    # The prompt suggests "Balanced-coordinate" is good.
    # Let's refine: 
    # Priority = - ( (c0 - c1)^2 + (c1 - c2)^2 + (c2 - c0)^2 )
    # This is a measure of imbalance. Lower is more balanced.
    # We want higher priority for balanced, so negate it.
    
    imbalance = (c0 - c1)**2 + (c1 - c2)**2 + (c2 - c0)**2
    base_score = -imbalance
    
    # Add jitter to break ties and explore.
    # Use a fast hash.
    # Converting to int:
    # val = 0
    # for i, v in enumerate(vector):
    #    val = val * 3 + v
    # This is O(n).
    
    # Is there a faster way?
    # Python's hash(vector) is O(n) too.
    # Let's use a simple LCG on the integer representation.
    
    # However, for n=12, 531k calls * O(12) is fine.
    # 531k * 12 = 6M ops. Very fast.
    
    val = 0
    for v in vector:
        val = val * 3 + v
        
    # Simple hash
    h = (val * 2654435761) & 0xFFFFFFFF
    # Normalize to [0, 1)
    jitter = (h % 1000) / 1000.0
    
    # Combine
    # Weight of jitter: 
    # If imbalance is large (e.g., 100), jitter (0-1) is negligible.
    # If imbalance is small (e.g., 0-1), jitter breaks ties.
    # This effectively groups vectors by their "shell" of balance, and randomizes within the shell.
    
    return base_score + jitter
