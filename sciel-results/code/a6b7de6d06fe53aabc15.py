"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We stop searching at 85s.
    # Return immediately if we hit 88s to ensure process termination.
    deadline = start_time + 85.0
    hard_deadline = start_time + 88.0
    
    # Candidates: 2 to 256
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute compatibility: comp[a][b] is True if lcm(a,b) > 256
    # lcm(a,b) = a*b // gcd(a,b)
    # Condition: a*b // gcd(a,b) > 256
    # To avoid overflow (not an issue in Python) and for speed, we can precompute a matrix.
    # 256x256 matrix is small.
    # Indices: 0..255 correspond to values 2..256? Or just use dict or 2D list indexed by value.
    # Let's use a 2D list indexed by value 0..256 for O(1) lookup.
    compat = [[False] * 257 for _ in range(257)]
    for a in range(2, 257):
        for b in range(a + 1, 257):
            g = gcd(a, b)
            if (a * b) // g > 256:
                compat[a][b] = True
                compat[b][a] = True
    
    # Precompute reciprocal values as Fractions for exact scoring
    # We also need a fast float approximation for pruning/heuristics if needed, 
    # but Fraction addition is reasonably fast for small sets.
    # To speed up local search, we can track the sum incrementally.
    
    # Helper to check if a candidate x can be added to a set S
    # S is a list of ints.
    def can_add(x, S):
        for y in S:
            if not compat[x][y]:
                return False
        return True

    # Helper to calculate sum of 1/x for a set
    def calc_sum(S):
        s = Fraction(0)
        for x in S:
            s += Fraction(1, x)
        return s

    best_set = []
    best_sum = Fraction(0)
    
    def update_best(S):
        nonlocal best_set, best_sum
        if not S:
            return
        # Check validity just in case (should be valid by construction)
        # But local search moves must preserve validity.
        s = calc_sum(S)
        if s > best_sum:
            best_sum = s
            best_set = list(S)

    # Greedy Initialization
    # Sort candidates by 1/x descending (i.e., x ascending)
    # Smaller x gives higher reward.
    # Try to add smallest x first.
    def greedy_init():
        S = []
        # We can shuffle candidates to break ties or explore different greedy paths?
        # Standard greedy: sort by value (1/x) desc, then by "restrictiveness" asc?
        # Here value is strictly 1/x.
        # Let's try multiple greedy variants.
        
        # Variant 1: Sort by x ascending (highest value first)
        for x in sorted(candidates):
            if can_add(x, S):
                S.append(x)
        return S

    # Local Search:
    # Moves:
    # 1. Add a candidate x not in S if valid.
    # 2. Remove a candidate y from S.
    # 3. Swap y in S with x not in S if valid.
    
    # To be efficient, we need to quickly find valid candidates.
    # Since N=255, iterating through all candidates for each move is O(N^2) per iteration.
    # With N=255, N^2 ~ 65k operations. 
    # If we do 1000 iterations, that's 65M operations. In Python, this might be too slow (seconds).
    # We need to be smarter or limit iterations.
    
    # Optimization:
    # Maintain a list of "candidates not in S".
    # For "Add" move: iterate through not_in_S, check can_add.
    # For "Swap" move: iterate y in S, then x in not_in_S, check can_add(x, S \ {y}).
    
    # Let's implement a simple local search with a time budget.
    
    def local_search(initial_S):
        S = list(initial_S)
        current_sum = calc_sum(S)
        update_best(S)
        
        # Precompute set of S for fast lookup? Set is O(1).
        S_set = set(S)
        
        no_improve_count = 0
        max_no_improve = 50 # Stop if stuck for a while
        
        while True:
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try Add moves
            # Candidates to add: x in candidates if x not in S_set
            # We prioritize adding small x (high value)
            # To speed up, we can just iterate through candidates sorted by x asc
            for x in candidates:
                if x in S_set:
                    continue
                if can_add(x, S):
                    # Check if adding x improves sum (always does, since 1/x > 0)
                    S.append(x)
                    S_set.add(x)
                    current_sum += Fraction(1, x)
                    update_best(S)
                    improved = True
                    # Accept first improvement found (steepest descent or first fit?)
                    # First fit is faster.
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. Try Swap moves
            # For each y in S, try to replace with x not in S
            # This allows removing a restrictive element to add a better one.
            # Note: Removing y might allow adding x that was previously blocked by y.
            # Or removing y allows adding a small x that was blocked by other elements? No, if x was blocked by z, removing y doesn't help unless y was blocking x.
            
            # Efficient swap search:
            # Iterate y in S (sorted by x asc? or random?)
            # For each y, calculate S_without_y.
            # Check if any x not in S is compatible with S_without_y.
            
            # To avoid O(N^2) per swap check, we can precompute conflicts?
            # Given time limits, let's just do a bounded search.
            
            # Try to find a swap that increases the sum.
            # Gain = 1/x_new - 1/y_old.
            # We want x_new < y_old.
            
            # Iterate y in S
            # Sort S by value? 
            # Let's try swapping y with the smallest possible x that fits.
            
            # Optimization: 
            # For each y in S:
            #   S_temp = S - {y}
            #   Find smallest x not in S such that can_add(x, S_temp)
            #   If found and 1/x > 1/y, perform swap.
            
            # This is O(|S| * N * |S|) in worst case.
            # |S| is likely small (maybe < 20). N=255.
            # 20 * 255 * 20 = 100k ops per iteration. 
            # If we do 100 iterations, 10M ops. Doable in Python? Maybe 1-2 seconds.
            
            for y in S:
                # Create S without y
                # We can just check can_add(x, S) for x where compat[x][y] is False?
                # No, x must be compatible with ALL z in S \ {y}.
                # This is equivalent to:
                # For all z in S, z != y: compat[x][z] is True.
                
                # Let's find the best x for this y.
                best_x_for_y = None
                best_gain = Fraction(0)
                
                # Iterate x in candidates not in S
                # Sort by x asc to find high gain (small x) quickly?
                # If we find a small x, gain is high.
                for x in candidates:
                    if x in S_set:
                        continue
                    # Check if x is compatible with all z in S except y
                    # If x is not compatible with y, it might still be compatible with S\{y}.
                    # We need to check all z in S where z != y.
                    valid = True
                    for z in S:
                        if z == y:
                            continue
                        if not compat[x][z]:
                            valid = False
                            break
                    if valid:
                        # Check gain
                        gain = Fraction(1, x) - Fraction(1, y)
                        if gain > best_gain:
                            best_gain = gain
                            best_x_for_y = x
                            # Can we break early? 
                            # Since we iterate x in ascending order (small x first), 
                            # the first valid x found will have the highest 1/x (lowest x).
                            # So gain is maximized for fixed y.
                            break
                
                if best_x_for_y is not None and best_gain > 0:
                    # Perform swap
                    S.remove(y)
                    S_set.remove(y)
                    S.append(best_x_for_y)
                    S_set.add(best_x_for_y)
                    current_sum += best_gain
                    update_best(S)
                    improved = True
                    break # Restart loop with new S
            
            if improved:
                no_improve_count = 0
                continue
            
            # 3. Try Remove moves?
            # Removing an element always decreases sum. So only useful if it enables a better Add/Swap later.
            # This is effectively a 2-step move. 
            # Given time, let's skip explicit remove moves unless we are stuck and want to escape local minima.
            # Random restarts will handle global exploration.
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            
    # Main Loop: Multiple Restarts
    # 1. Greedy Init
    # 2. Local Search
    # 3. Perturb (Random Remove/Add) -> Local Search
    
    # Run greedy first
    S_greedy = greedy_init()
    local_search(S_greedy)
    
    # Check time
    if time.time() > deadline:
        return best_set
    
    # Random Restarts
    # We can try different random seeds or perturbations.
    # Strategy: Start from best_set, remove k random elements, add random valid elements, then local search.
    
    # Also try different initializations:
    # - Sort by random order?
    # - Sort by "flexibility" (number of compatible partners) descending?
    
    # Let's compute flexibility
    flex = [0] * 257
    for a in range(2, 257):
        for b in range(2, 257):
            if a != b and compat[a][b]:
                flex[a] += 1
    
    # Init 2: Sort by flexibility ascending (most restrictive first)
    # This might lead to a different packing.
    def greedy_init_flex():
        S = []
        for x in sorted(candidates, key=lambda x: flex[x]):
            if can_add(x, S):
                S.append(x)
        return S
    
    if time.time() < deadline:
        S_flex = greedy_init_flex()
        local_search(S_flex)
        
    # Init 3: Random Greedy
    # Shuffle candidates, then greedy add
    while time.time() < deadline:
        # Random restart
        # Take current best, perturb it
        if best_set:
            S_perturbed = list(best_set)
            # Remove 2-5 random elements
            num_remove = min(len(S_perturbed), random.randint(2, 5))
            if num_remove > 0:
                to_remove = random.sample(S_perturbed, num_remove)
                for y in to_remove:
                    S_perturbed.remove(y)
            
            # Add some random valid elements
            # Try to add random candidates
            for _ in range(10):
                x = random.choice(candidates)
                if x not in S_perturbed and can_add(x, S_perturbed):
                    S_perturbed.append(x)
            
            local_search(S_perturbed)
        else:
            # If best is empty (shouldn't happen), do random greedy
            S_random = []
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, S_random):
                    S_random.append(x)
            local_search(S_random)

    return best_set

# EVOLVE-BLOCK-END
