"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 12
    m = 40
    
    # Use a subcube of dimension 6 to maximize comparability density.
    # Vertices 0..5
    vertices = range(6)
    
    # Generate all subsets of size 0, 1, 2, 3
    # Total 1 + 6 + 15 + 20 = 42 sets.
    # We need 40, so we will remove 2 sets from the largest layer (size 3).
    
    family = []
    
    # Size 0
    family.append([])
    
    # Size 1
    for i in vertices:
        family.append([i])
        
    # Size 2
    for combo in itertools.combinations(vertices, 2):
        family.append(list(combo))
        
    # Size 3
    for combo in itertools.combinations(vertices, 3):
        family.append(list(combo))
        
    # We have 42 sets. Remove 2 from the end (which are size 3).
    # Removing any 2 size 3 sets is optimal as they have the minimum degree in this subgraph.
    family = family[:40]
    
    # Verify types and distinctness
    # Convert to lists of ints
    final_family = []
    for s in family:
        final_family.append([int(x) for x in s])
        
    # Check distinctness
    if len(set(tuple(s) for s in final_family)) != 40:
        # Fallback to simple greedy if something went wrong (should not happen)
        final_family = [[i] for i in range(12)] + [[i, j] for i in range(12) for j in range(i+1, 12)][:28]
        
    return final_family

# EVOLVE-BLOCK-END
