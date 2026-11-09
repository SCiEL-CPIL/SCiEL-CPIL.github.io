"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

# Precompute conflict structure
# For each pair of distinct integers (a, b) from 1..192, we need to know if there exist
# distinct c, d (both != a, b) such that a | (b+c+d) or b | (a+c+d)
# Actually, the constraint is: for any selected set S, and any x in S,
# there should be no 3 distinct OTHER elements y,z,w in S such that x | (y+z+w).

# Let's think about this differently.
# A set S is valid if for all x in S, and all distinct y,z,w in S\{x}, x does NOT divide y+z+w.

# We want to maximize |S| with S subset of {1, ..., 192}.

# Strategy: Use a greedy approach with local search, leveraging vectorized validation.
# 1. Start with a greedy construction (largest first, or some heuristic).
# 2. Use local search (add, remove, swap) to improve.
# 3. Use NumPy for fast validation of whether adding an element violates the constraint.

def _is_valid_set(selected):
    """Check if the selected set is valid."""
    n = len(selected)
    if n < 4:
        return True
    sel = np.array(selected)
    # For each x in sel, check if there exist distinct y,z,w in sel\{x} such that x | (y+z+w)
    # This is O(n^4) in worst case, but we can vectorize.
    # For each x (index i), consider all triples from the other n-1 elements.
    # There are C(n-1, 3) triples. For large n this is expensive.
    # Alternative: For each x, compute all sums of 3 distinct elements from S\{x},
    # and check if any is divisible by x.
    # Even better: precompute for each x, which triples (y,z,w) with y,z,w != x have y+z+w divisible by x.
    # But that's global precomputation which is heavy.
    
    # Let's use a different approach: for each x in selected, 
    # get all other elements, compute all triple sums, check divisibility.
    for i, x in enumerate(selected):
        others = [sel[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        others = np.array(others)
        # Generate all combinations of 3 from others
        # For efficiency, if len(others) is large, this is expensive.
        # But selected size is at most 48, so len(others) <= 47.
        # C(47,3) = 16215. For 48 elements, total ~ 48 * 16215 = 778k checks.
        # Each check is a modulo. This might be too slow in Python loops.
        
        # Vectorized approach:
        # Get all pairs from others, then for each pair, the remaining elements form triples.
        # Actually, let's just iterate over triples but use numpy for the sum and mod.
        # Better: use numpy to generate all triple indices.
        m = len(others)
        if m < 3:
            continue
        # Generate all triples of indices
        # This is still O(m^3) in memory if we generate all at once.
        # For m=47, C(47,3)=16215, which is fine.
        idx1 = np.repeat(np.arange(m), np.repeat(m-1, m-1))  # This is getting complicated
        # Let's just use itertools but check in batches
        found = False
        for t in combinations(range(m), 3):
            s = others[t[0]] + others[t[1]] + others[t[2]]
            if s % x == 0:
                found = True
                break
        if found:
            return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    # x must not divide sum of any 3 distinct elements in selected
    # AND no element in selected must divide sum of any 2 elements in selected + x
    # Wait, the constraint is: no selected integer divides sum of any 3 distinct OTHER selected integers.
    # If we add x to S, we get S' = S ∪ {x}.
    # For x: no 3 distinct elements in S should sum to a multiple of x.
    # For each y in S: no 3 distinct elements in S' \ {y} should sum to a multiple of y.
    #   S' \ {y} = (S \ {y}) ∪ {x}
    #   The 3 elements can be:
    #     (a) all from S\{y}: already checked when S was valid
    #     (b) 2 from S\{y} and x
    #     (c) 1 from S\{y} and... wait, we need 3 distinct elements.
    #         If we pick x and 2 from S\{y}, that's 3 elements.
    #         If we pick x and 1 from S\{y}, we need another element, but it must be distinct from x and the other.
    #         Actually, the 3 distinct OTHER elements for y are chosen from S' \ {y}.
    #         S' \ {y} has |S| elements: (S\{y}) plus x.
    #         Triples from S' \ {y}:
    #           - 3 from S\{y}: already valid
    #           - 2 from S\{y} + x
    #           - 1 from S\{y} + x + ... no, we only have x as the new element.
    #           So the only new triples are those containing x and 2 elements from S\{y}.
    # So we need to check:
    # 1. For x: no triple in S sums to multiple of x.
    # 2. For each y in S: no pair (a,b) in S\{y} such that a+b+x is divisible by y.
    
    sel = list(selected)
    n = len(sel)
    
    # Check 1: x does not divide any triple sum from S
    if n >= 3:
        sel_arr = np.array(sel)
        for t in combinations(range(n), 3):
            s = sel_arr[t[0]] + sel_arr[t[1]] + sel_arr[t[2]]
            if s % x == 0:
                return False
    
    # Check 2: For each y in S, no pair (a,b) in S\{y} with a+b+x ≡ 0 (mod y)
    for i, y in enumerate(sel):
        others = [sel[j] for j in range(n) if j != i]
        if len(others) < 2:
            continue
        others_arr = np.array(others)
        # Check all pairs
        for t in combinations(range(len(others)), 2):
            s = others_arr[t[0]] + others_arr[t[1]] + x
            if s % y == 0:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    # Greedy initialization: try multiple orderings
    best_set = []
    best_size = 0
    
    # Try greedy with different orderings
    orderings = [
        list(range(192, 0, -1)),  # Largest first
        list(range(1, 193)),      # Smallest first
    ]
    
    for ordering in orderings:
        selected = []
        for x in ordering:
            if _can_add(selected, x):
                selected.append(x)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Local search
    current_set = best_set[:]
    current_size = best_size
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements
        for x in range(1, 193):
            if x not in current_set and _can_add(current_set, x):
                current_set.append(x)
                current_size += 1
                improved = True
                break
        
        # Try swapping: remove one, add one
        if not improved:
            for i in range(len(current_set)):
                for x in range(1, 193):
                    if x in current_set:
                        continue
                    new_set = current_set[:i] + current_set[i+1:] + [x]
                    if _is_valid_set(new_set):
                        current_set = new_set
                        improved = True
                        break
                if improved:
                    break
    
    return current_set
# EVOLVE-BLOCK-END
