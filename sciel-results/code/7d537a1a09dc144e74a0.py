"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    # We want to find n distinct positive integers such that all 2^n subset sums are distinct.
    # This is equivalent to finding a set of n integers with all subset sums unique (a "sum-free" set in a strong sense).
    # The greedy approach: start with empty set, add the smallest positive integer that doesn't create a collision.
    
    # Current subset sums
    subset_sums = {0}
    chosen = []
    
    # We'll try to add numbers from 1 upwards
    candidate = 1
    while len(chosen) < n:
        # Check if adding 'candidate' creates a collision
        # New subset sums would be: existing_sums union (existing_sums + candidate)
        # Collision occurs if any (s + candidate) is already in subset_sums for some s in subset_sums
        # OR if candidate itself is already in subset_sums (which is a special case of s=0)
        
        # Efficient check: for each s in subset_sums, if (s + candidate) in subset_sums, collision.
        # But we can be smarter: the new sums are {s + candidate for s in subset_sums}.
        # We need intersection of {s + candidate for s in subset_sums} with subset_sums to be empty.
        
        # Since subset_sums is a set, we can check:
        collision = False
        for s in subset_sums:
            if (s + candidate) in subset_sums:
                collision = True
                break
        
        if not collision:
            # Add candidate to chosen list
            chosen.append(candidate)
            # Update subset_sums: new_sums = subset_sums | {s + candidate for s in subset_sums}
            new_sums = set()
            for s in subset_sums:
                new_sums.add(s + candidate)
            subset_sums = subset_sums | new_sums
            candidate += 1
        else:
            candidate += 1
    
    return chosen
# EVOLVE-BLOCK-END
