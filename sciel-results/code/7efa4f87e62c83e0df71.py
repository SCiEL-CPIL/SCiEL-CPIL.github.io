def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    s = sum(vector) % 3
    
    # Prioritize Sum=1 over 2 over 0?
    # Map to score: 1->0, 2->1, 0->2 (lower is better if we add? No, we return priority, higher is better)
    # So we want high priority for Sum=1.
    # Bonus: 1.0 for Sum=1, 0.5 for Sum=2, 0.0 for Sum=0?
    # This bonus should be smaller than the difference in -dev for different balance classes?
    # Or should it be primary?
    # If Balance is primary, Sum is secondary.
    
    # Use a fast deterministic tie-breaker based on vector pattern
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) & 0x7FFFFFFF
    
    h_norm = h / 0x7FFFFFFF
    
    sum_bonus = {1: 2.0, 2: 1.0, 0: 0.0}[s]
    
    # Scale sum_bonus so it doesn't override -dev
    # dev min is 0. max is roughly n.
    # Let's say dev is in [0, 10].
    # sum_bonus in [0, 2].
    # If we add sum_bonus * 0.1, it's small.
    
    return -dev + sum_bonus * 0.01 + 1e-9 * h_norm
