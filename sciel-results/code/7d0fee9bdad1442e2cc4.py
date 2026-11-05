"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def construct():
    n = 256
    start_time = time.time()
    time_limit = 55  # seconds, leave buffer for evaluation overhead
    
    # Precompute float reciprocals and suffix sums for fast upper bounds
    float_reciprocals = [0.0] * (n+1)
    for i in range(1, n+1):
        float_reciprocals[i] = 1.0 / i
    
    float_suffix = [0.0] * (n+2)
    for i in range(n, 0, -1):
        float_suffix[i] = float_suffix[i+1] + float_reciprocals[i]
        
    # Exact fractions for final result
    reciprocals = [Fraction(0) for _ in range(n+1)]
    for i in range(1, n+1):
        reciprocals[i] = Fraction(1, i)

    # Precompute conflict pairs for validity check
    # conflict_pairs[x] contains (a, b) such that a < b < x and a + x = 2*b
    # If a and b are selected, x cannot be selected.
    conflict_pairs = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        for b in range(1, x):
            a = 2*b - x
            if 1 <= a < b:
                conflict_pairs[x].append((a, b))
                
    # Precompute masks for "less than x"
    mask_lt = [0] * (n+1)
    for x in range(1, n+1):
        mask_lt[x] = (1 << (x-1)) - 1
        
    # Global best tracking
    global_best_float = 0.0
    global_best_set = set()
    global_best_frac = Fraction(0)
    timeout_flag = [False]
    
    def calculate_exact_sum(s):
        return sum(reciprocals[x] for x in s)

    def greedy_solution():
        selected = set()
        selected_bits = 0
        for x in range(1, n+1):
            # Check validity
            valid = True
            for a, b in conflict_pairs[x]:
                if (selected_bits >> (a-1)) & 1 and (selected_bits >> (b-1)) & 1:
                    valid = False
                    break
            if valid:
                selected.add(x)
                selected_bits |= (1 << (x-1))
        return selected

    def local_search(initial_set):
        """Improve the set using single moves (add, remove, swap)."""
        current_set = set(initial_set)
        current_sum = sum(float_reciprocals[x] for x in current_set)
        current_bits = 0
        for x in current_set:
            current_bits |= (1 << (x-1))
            
        improved = True
        max_iters = 1000
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > time_limit:
                break
                
            # Try adding an element
            for x in range(1, n+1):
                if x in current_set:
                    continue
                # Check if x can be added
                valid = True
                for a, b in conflict_pairs[x]:
                    if (current_bits >> (a-1)) & 1 and (current_bits >> (b-1)) & 1:
                        valid = False
                        break
                if valid:
                    # Adding x is always an improvement in weight
                    current_set.add(x)
                    current_bits |= (1 << (x-1))
                    current_sum += float_reciprocals[x]
                    improved = True
                    break # Restart loop after change
            
            if improved:
                continue
                
            # Try removing and adding (swap)
            # This is expensive, so maybe just try removing low weight elements?
            # Or just try removing any element and see if we can add a higher one?
            # For now, let's stick to simple additions and maybe some removals if stuck?
            # Actually, if no single addition is possible, we might be in a local max.
            # We can try removing a small element to allow adding a larger one?
            # But removing reduces weight. Only beneficial if it allows adding multiple or a much larger one.
            # Given time constraints, simple greedy + add-only local search might be enough for a good lower bound.
            
            # Let's try a few random removals to escape local maxima?
            # No, keep it deterministic for reproducibility in the seed.
            
        return current_set, current_sum

    # Initialize with greedy and local search
    initial_set = greedy_solution()
    initial_set, initial_sum = local_search(initial_set)
    
    global_best_set = initial_set
    global_best_float = initial_sum
    global_best_frac = calculate_exact_sum(initial_set)
    
    # Branch and Bound
    def branch_and_bound(x, selected_bits, forbidden_bits, current_float_sum, forbidden_float_sum):
        nonlocal global_best_float, global_best_set, global_best_frac, timeout_flag
        
        if timeout_flag[0]:
            return
            
        # Check time periodically
        if x % 100 == 0 and time.time() - start_time > time_limit:
            timeout_flag[0] = True
            return
            
        if x > n:
            if current_float_sum > global_best_float:
                global_best_float = current_float_sum
                # Reconstruct set
                s = set()
                for i in range(1, n+1):
                    if (selected_bits >> (i-1)) & 1:
                        s.add(i)
                global_best_set = s
                global_best_frac = calculate_exact_sum(s)
            return
        
        # Upper Bound
        # Sum of remaining reciprocals, minus those forbidden
        # We maintain forbidden_float_sum, but it includes forbidden elements < x too?
        # No, we should only subtract forbidden elements >= x.
        # To make this efficient, we can just use the loose bound:
        # UB = current_float_sum + float_suffix[x]
        # This is valid because float_suffix[x] includes all y >= x, and forbidden ones are a subset.
        # It's a valid upper bound, just potentially weak.
        # A tighter bound: current_float_sum + (float_suffix[x] - sum(1/y for y in forbidden, y>=x))
        # Calculating the sum of forbidden y>=x at each node is expensive.
        # Let's stick to the loose bound first for speed, or precompute something?
        # Actually, if we maintain `forbidden_bits`, we can iterate set bits >= x.
        # But that's O(N) per node.
        # Let's use the loose bound. It prunes less but is faster.
        
        ub = current_float_sum + float_suffix[x]
        if ub <= global_best_float:
            return
        
        # If x is forbidden, skip
        if (forbidden_bits >> (x-1)) & 1:
            branch_and_bound(x+1, selected_bits, forbidden_bits, current_float_sum, forbidden_float_sum)
            return
        
        # Try selecting x
        # Check validity
        valid = True
        for a, b in conflict_pairs[x]:
            if (selected_bits >> (a-1)) & 1 and (selected_bits >> (b-1)) & 1:
                valid = False
                break
        
        if valid:
            new_selected_bits = selected_bits | (1 << (x-1))
            
            # Calculate new forbidden bits
            new_forbidden_bits = forbidden_bits
            # For each a in selected (< x), c = 2x - a. If c > x, forbid c.
            # selected_bits & mask_lt[x] gives the bits for a < x.
            sel_bits = selected_bits & mask_lt[x]
            while sel_bits:
                # Get lowest set bit
                lsb = sel_bits & -sel_bits
                a_idx = lsb.bit_length() - 1 # 0-indexed bit pos, so a = a_idx + 1
                a = a_idx + 1
                c = 2*x - a
                if c > x and c <= n:
                    new_forbidden_bits |= (1 << (c-1))
                    # We don't update forbidden_float_sum here because we use loose bound
                sel_bits ^= lsb
            
            branch_and_bound(x+1, new_selected_bits, new_forbidden_bits, 
                             current_float_sum + float_reciprocals[x], 
                             forbidden_float_sum) # forbidden_float_sum not used in loose bound
        
        # Try not selecting x
        branch_and_bound(x+1, selected_bits, forbidden_bits, current_float_sum, forbidden_float_sum)

    branch_and_bound(1, 0, 0, 0.0, 0.0)
    
    return list(global_best_set)
# EVOLVE-BLOCK-END
