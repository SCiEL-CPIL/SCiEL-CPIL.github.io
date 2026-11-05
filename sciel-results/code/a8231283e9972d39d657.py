"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 60s
    
    # The problem: Maximize sum(1/x) for x in S, S subset [2, 256]
    # Constraint: lcm(a, b) > 256 for all distinct a, b in S.
    
    # Note: lcm(a,b) = a*b/gcd(a,b).
    # If gcd(a,b)=1, lcm(a,b) = a*b.
    # If a*b <= 256, then lcm(a,b) <= 256, so they CANNOT be in S together.
    # Thus, for any pair in S, if they are coprime, their product must be > 256.
    # This implies that small numbers are very restrictive.
    # e.g., 2 cannot be with any y where lcm(2,y)<=256. 
    # lcm(2,y) = y if y even, 2y if y odd.
    # If y is even, lcm=y. So y must be > 256? No, y is in [2,256]. 
    # So if y is even, lcm(2,y)=y <= 256. Thus 2 cannot be with ANY even number.
    # If y is odd, lcm(2,y)=2y. We need 2y > 256 => y > 128.
    # So 2 can only be with odd numbers in (128, 256].
    
    # Let's use a greedy approach sorted by value (ascending value of x means descending 1/x).
    # We want to include small x if possible.
    
    candidates = list(range(2, 257))
    
    # Precompute gcds or just use math.gcd on the fly.
    # For speed, we can precompute compatibility.
    # But 255 elements is small.
    
    def is_compatible(a, b):
        return math.lcm(a, b) > 256
    
    # Greedy Initialization: Sort by ascending value (highest reciprocal first)
    # Actually, we want to maximize sum(1/x). So we prefer smaller x.
    # Sort candidates by x ascending.
    sorted_candidates = sorted(candidates)
    
    selected = []
    
    # Pass 1: Greedy add
    for x in sorted_candidates:
        if time.time() > deadline:
            break
        # Check if x can be added to current selected set
        can_add = True
        for y in selected:
            if not is_compatible(x, y):
                can_add = False
                break
        if can_add:
            selected.append(x)
    
    # Current sum
    def calc_sum(S):
        return sum(1.0/x for x in S)
    
    current_sum = calc_sum(selected)
    selected_set = set(selected)
    
    # Local Search: Try to improve by swapping.
    # We iterate through potential swaps: remove y from selected, add x (not in selected).
    # We prefer to remove large y (small 1/y) and add small x (large 1/x).
    
    # To be efficient, we can sort 'selected' by descending value (largest x first) 
    # and 'unselected' by ascending value (smallest x first).
    
    # We will do several passes of local search.
    # A simple hill-climbing swap:
    # For each y in selected (sorted largest to smallest):
    #   For each x in unselected (sorted smallest to largest):
    #     If 1/x > 1/y (i.e., x < y):
    #       Check if (selected - {y}) U {x} is valid.
    #       If yes, perform swap, update current_sum, break inner loop (restart or continue?).
    
    # Since the space is small, we can afford a more thorough search if time permits.
    # But simple swaps might get stuck in local minima.
    # Let's try a randomized restart or a more systematic swap.
    
    # Better approach: 
    # 1. Start with greedy.
    # 2. Try to add any unselected x that fits (shouldn't exist if greedy was perfect, but maybe order matters? 
    #    Greedy by ascending x is optimal for "maximum cardinality" in some matroids, but here objective is sum.
    #    However, greedy by ascending x is a good heuristic for sum because 1/x is convex).
    
    # Let's focus on swaps.
    # We maintain a list of unselected candidates.
    
    unselected = [x for x in candidates if x not in selected_set]
    unselected.sort() # Smallest first (highest value)
    
    # Sort selected to prioritize removing large elements (low value)
    # We'll iterate through selected in descending order of x (ascending 1/x is bad, so remove big x first)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through selected elements to potentially remove
        # We want to remove elements with small 1/x (large x).
        # So iterate selected sorted descending.
        selected_sorted_desc = sorted(selected, reverse=True)
        
        for y in selected_sorted_desc:
            if time.time() > deadline:
                break
            # Try to replace y with some x < y from unselected
            # We only care about x < y because if x > y, 1/x < 1/y, sum decreases.
            # So we look for x in unselected such that x < y.
            
            # To make this faster, we can pre-filter unselected for x < y.
            # But unselected changes.
            
            # Let's just iterate unselected (sorted ascending) and break if x >= y.
            for x in unselected:
                if x >= y:
                    break
                # Potential swap: remove y, add x
                # Check validity of new set
                # New set = selected - {y} + {x}
                # We need to check x against all elements in selected except y.
                
                valid = True
                for z in selected:
                    if z == y:
                        continue
                    if not is_compatible(x, z):
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    selected.remove(y)
                    selected.append(x)
                    selected_set.remove(y)
                    selected_set.add(x)
                    
                    unselected.remove(x)
                    unselected.append(y)
                    unselected.sort() # Keep unselected sorted
                    
                    current_sum += (1.0/x - 1.0/y)
                    improved = True
                    break # Restart the inner loop for this y? Or just move to next y?
                    # Breaking inner loop means we try to swap out the next largest y.
                    # This is a standard hill climbing move.
        
        # If no swap was made in this pass, we are at a local optimum for single swaps.
    
    return selected

# EVOLVE-BLOCK-END
