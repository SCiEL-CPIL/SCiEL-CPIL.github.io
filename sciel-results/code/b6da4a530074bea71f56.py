import time
import sys

# Global state for deadline tracking
_start_time = None
_DEADLINE = 300.0

def priority(vector: tuple, n: int) -> float:
    """
    Strategy: Sum of Squares Priority
    
    Instead of reacting to local density (load balancing), this strategy applies a 
    static, global bias that favors vectors with a higher sum of squared coordinates.
    
    In {0,1,2}^n, the sum of squares is:
    0^2 = 0
    1^2 = 1
    2^2 = 4
    
    Vectors with more 2s and 1s will have higher scores. This breaks the 
    lexicographic clustering (which tends to pick 000..., 001..., 010...) by 
    deprioritizing the "low energy" vectors that are dense in the origin corner.
    
    The hypothesis is that spreading the selection towards the "corners" of the 
    cube (high values) reduces the likelihood of forming arithmetic progressions 
    that are common in the low-value dense regions, or simply that the greedy 
    algorithm benefits from a more uniform distribution of "weight" rather than 
    just local coordinate balance.
    
    This is a structural change from the previous "stateful counter" approach to 
    a "static global potential" approach.
    """
    global _start_time
    
    # Initialize timer on first call
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: If we are very close to the limit, return a stable value 
    # to avoid any potential overhead, though this calculation is O(n) and fast.
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Return a neutral value if we are in the danger zone, though unlikely 
        # to be hit given the O(n) cost.
        return 0.0

    # Calculate sum of squares
    # Using a list comprehension or sum of generator is efficient enough for n <= ~20-30
    # If n is very large, we might need to optimize, but standard cap set problems 
    # in these contexts usually have manageable n.
    s = 0
    for v in vector:
        s += v * v
        
    return float(s)
