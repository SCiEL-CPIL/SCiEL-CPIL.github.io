import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of distinct integers in 1..256 maximizing size k
    such that subset sums of different cardinalities are disjoint.
    
    Analysis:
    For an Arithmetic Progression x_i = a + (i-1) with i=1..k,
    the condition that max(S_j) < min(S_{j+1}) for all j
    simplifies to a > max_{1<=j<k} j(k-1-j).
    
    The maximum value of j(k-1-j) is floor((k-1)^2 / 4).
    Let M_k = floor((k-1)^2 / 4).
    We need a >= M_k + 1.
    Also x_k = a + k - 1 <= 256 => a <= 257 - k.
    
    So we need M_k + 1 <= 257 - k.
    M_k + k <= 256.
    
    We search for the largest k satisfying this.
    """
    
    max_val = 256
    best_set = None
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    # Max possible k is bounded by 256 (if a=1, k=256? No, distinct)
    # But a must be large.
    # For k=1, M=0, a>=1, x_1<=256. OK.
    # We iterate downwards from a reasonable upper bound.
    # k cannot exceed 256.
    # Actually, if k is large, M_k is ~ k^2/4.
    # k^2/4 + k <= 256 => k^2 ~ 1024 => k ~ 32.
    # So we only need to check up to 40 or so.
    
    for k in range(40, 0, -1):
        if time.time() > deadline:
            break
            
        # Calculate max of j(k-1-j) for j in 1..k-1
        # This is a parabola opening downwards, vertex at (k-1)/2.
        # Max integer value is at floor((k-1)/2) or ceil((k-1)/2).
        j_opt = (k - 1) // 2
        # Check both neighbors just in case of integer division quirks
        candidates = [j_opt, j_opt + 1]
        # Ensure j is in [1, k-1]
        valid_j = [j for j in candidates if 1 <= j <= k - 1]
        if not valid_j:
             # For k=1, range is empty, max is 0
             max_j_val = 0
        else:
             max_j_val = max(j * (k - 1 - j) for j in valid_j)
        
        required_a = max_j_val + 1
        max_allowed_a = 256 - (k - 1)
        
        if required_a <= max_allowed_a:
            a = required_a
            set_vals = [a + i for i in range(k)]
            
            # Verify distinctness and range
            if len(set(set_vals)) == k and all(1 <= v <= 256 for v in set_vals):
                best_set = set_vals
                break
                
    if best_set is None:
        return [1]
        
    return best_set
# EVOLVE-BLOCK-END
