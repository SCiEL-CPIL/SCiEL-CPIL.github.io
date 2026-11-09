"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    subset_sums = {0}
    chosen = []
    candidate = 1
    
    while len(chosen) < n:
        # If candidate is greater than the maximum current subset sum,
        # then (subset_sums + candidate) is disjoint from subset_sums.
        max_sum = max(subset_sums)
        if candidate <= max_sum:
            # Check for collision
            collision = False
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    collision = True
                    break
            
            if collision:
                candidate += 1
                continue
        
        # Add candidate
        chosen.append(candidate)
        # Update subset sums efficiently
        subset_sums = subset_sums | {s + candidate for s in subset_sums}
        candidate += 1
        
    return chosen
# EVOLVE-BLOCK-END
