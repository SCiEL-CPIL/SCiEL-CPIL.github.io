"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute: for each x in [1,256], the set of pairs (a,b) such that {a,b,x} forms a 3-AP
    # A 3-AP among {a,b,x} means one is the midpoint of the other two.
    # We'll use a bitmask for the selected set for fast membership checks.
    
    # For efficiency, precompute for each x the list of (a, b) pairs that would form a 3-AP with x
    # where a < b and all in [1, 256]
    # The three possible configurations:
    # 1) a, b, x with a < b < x and b - a == x - b => a + x = 2b
    # 2) a, x, b with a < x < b and x - a == b - x => a + b = 2x  
    # 3) x, a, b with x < a < b and a - x == b - a => x + b = 2a
    
    # For checking if adding x to set S creates a 3-AP:
    # There exist a, b in S such that {a, b, x} is a 3-AP
    # This happens iff:
    #   - There exists a in S such that 2*x - a is in S and 2*x - a != x (i.e., a != x, which is true since x not in S)
    #     Wait, let me think more carefully.
    #   - x is the middle: exists a, b in S with a < x < b and b - x == x - a, i.e., a + b = 2x
    #   - x is the largest: exists a, b in S with a < b < x and x - b == b - a, i.e., a + x = 2b, so b = (a+x)/2
    #   - x is the smallest: exists a, b in S with x < a < b and a - x == b - a, i.e., x + b = 2a, so a = (x+b)/2
    
    # Efficient check using bitmask:
    # For a given x and bitmask M representing selected set:
    # 1) x is middle: for some a in S with a < x, check if (2*x - a) is in S and (2*x - a) > x
    # 2) x is largest: for some a in S with a < x, check if (a + x) is even and ((a+x)/2) is in S and ((a+x)/2) < x
    # 3) x is smallest: for some b in S with b > x, check if (x + b) is even and ((x+b)/2) is in S and ((x+b)/2) < b
    
    # Actually, simpler: for each pair (a, b) in S with a < b, check if the third element of the AP is x.
    # The three possible APs containing x and two elements from S:
    #   (a, b, x): x = 2b - a, need a < b < x
    #   (a, x, b): x = (a+b)/2, need a < x < b  
    #   (x, a, b): x = 2a - b, need x < a < b
    
    # Let me use a different approach: precompute for each x, the list of "conflict pairs"
    # conflict_pairs[x] = list of (a, b) such that {a, b, x} is a 3-AP and a < b
    
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        pairs = []
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                # Check if {a, b, x} forms a 3-AP
                # One of them must be the average of the other two
                if 2 * a == b + x or 2 * b == a + x or 2 * x == a + b:
                    pairs.append((a, b))
        conflict_pairs[x] = pairs
    
    # Greedy construction: add elements in order of decreasing 1/x (i.e., increasing x)
    # but only if they don't create a 3-AP
    # Actually, to maximize reciprocal sum, we want to include small numbers.
    # Greedy: for x = 1, 2, ..., 256, add x if it doesn't create a 3-AP with existing elements.
    
    selected_mask = 0  # bitmask where bit i is set if (i+1) is selected (0-indexed)
    selected_list = []
    
    for x in range(1, N + 1):
        # Check if adding x creates a 3-AP
        can_add = True
        for (a, b) in conflict_pairs[x]:
            # Check if both a and b are in selected_mask
            if (selected_mask >> (a - 1)) & 1 and (selected_mask >> (b - 1)) & 1:
                can_add = False
                break
        if can_add:
            selected_mask |= (1 << (x - 1))
            selected_list.append(x)
    
    # Current reciprocal sum
    current_sum = Fraction(0, 1)
    for x in selected_list:
        current_sum += Fraction(1, x)
    
    best_mask = selected_mask
    best_sum = current_sum
    best_list = selected_list[:]
    
    # Hill climbing: try to improve by:
    # 1. Removing an element (if it doesn't hurt)
    # 2. Adding an element (if it doesn't create AP)
    # 3. Swapping: remove one, add another
    # 4. Remove two, add three, etc.
    
    # Precompute reciprocal values as Fractions for quick comparison
    recips = [Fraction(0, 1)] + [Fraction(1, i) for i in range(1, N + 1)]
    
    def is_valid_add(mask, x):
        """Check if adding x to the set represented by mask is valid (no 3-AP)."""
        for (a, b) in conflict_pairs[x]:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                return False
        return True
    
    def is_valid_remove(mask, x):
        """Removing x is always valid since we're reducing the set."""
        return True
    
    # Local search with multiple move types
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all single removals
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if (best_mask >> (x - 1)) & 1:
                new_mask = best_mask & ~(1 << (x - 1))
                new_sum = best_sum - recips[x]
                if new_sum > best_sum:  # This won't happen since we're removing
                    pass
                # Removal alone never improves, so skip pure removals
        
        # Try all single additions
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((best_mask >> (x - 1)) & 1):
                if is_valid_add(best_mask, x):
                    new_mask = best_mask | (1 << (x - 1))
                    new_sum = best_sum + recips[x]
                    if new_sum > best_sum:
                        best_mask = new_mask
                        best_sum = new_sum
                        improved = True
        
        # Try swaps: remove x, add y
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if (best_mask >> (x - 1)) & 1:
                trial_mask = best_mask & ~(1 << (x - 1))
                for y in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if y == x:
                        continue
                    if not ((trial_mask >> (y - 1)) & 1):
                        if is_valid_add(trial_mask, y):
                            new_sum = best_sum - recips[x] + recips[y]
                            if new_sum > best_sum:
                                best_mask = trial_mask | (1 << (y - 1))
                                best_sum = new_sum
                                improved = True
                                break  # restart scan after a swap
    
    # Extract final list
    final_list = []
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            final_list.append(x)
    
    return final_list
# EVOLVE-BLOCK-END
