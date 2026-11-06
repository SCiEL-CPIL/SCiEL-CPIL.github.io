"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 85.0
    selected = []
    
    # Start with a known valid set or build greedily
    # The seed builds from 1 upwards. Let's replicate that logic but faster.
    # For a set S, the condition is that all sums a+b+c (a<=b<=c) are unique.
    # This is equivalent to saying that the set S is a B_3[1] set.
    
    # Optimization: 
    # 1. Use a set for O(1) lookup of existing sums.
    # 2. Only check new sums involving the candidate element x.
    #    Existing sums among selected elements are already unique.
    #    We need to check:
    #    - x+x+x
    #    - x+x+y for y in selected
    #    - x+y+y for y in selected
    #    - x+y+z for y,z in selected with y<=z
    #    Wait, the condition is for *unordered* triples.
    #    If we add x to S, the new triples are those containing x.
    #    Let S be the current set. New triples are:
    #    1. (x, x, x)
    #    2. (x, x, y) for y in S
    #    3. (x, y, y) for y in S
    #    4. (x, y, z) for y, z in S with y < z
    #    
    #    We need to ensure these new sums are not in the set of existing sums 
    #    and are not equal to each other.
    
    existing_sums = set()
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
            
        # Generate new sums involving x
        new_sums = []
        
        # (x, x, x)
        s1 = 3 * x
        new_sums.append(s1)
        
        # (x, x, y) and (x, y, y) and (x, y, z)
        # We iterate through existing elements y in selected
        # Note: selected is sorted because we iterate x upwards.
        
        for i, y in enumerate(selected):
            # (x, x, y) -> 2x + y
            new_sums.append(2 * x + y)
            # (x, y, y) -> x + 2y
            new_sums.append(x + 2 * y)
            
            # (x, y, z) where z > y and z in selected
            # Since selected is sorted, z is selected[i+1:]
            for z in selected[i+1:]:
                new_sums.append(x + y + z)
                
        # Check for duplicates within new_sums
        # And check against existing_sums
        
        if len(new_sums) != len(set(new_sums)):
            continue
            
        # Check intersection with existing sums
        if existing_sums.intersection(set(new_sums)):
            continue
            
        # Valid
        selected.append(x)
        existing_sums.update(new_sums)
        
        # Optional: If we want to maximize size, we should keep going.
        # But the range is up to 4096.
        
    return selected
# EVOLVE-BLOCK-END
