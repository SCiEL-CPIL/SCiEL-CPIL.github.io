"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precomputation:
    # 1. valid_pair[i][j] = True if {i, j} can coexist in a Sidon set.
    #    For distinct i, j, this is always True (sums 2i, i+j, 2j are distinct iff i!=j).
    #    So the only constraint is on the SET of sums.
    #
    # 2. For a set S, the set of sums is Sums(S) = {a+b | a,b in S, a<=b}.
    #    A number x is "blocked" by S if adding x to S creates a duplicate sum.
    #    This happens if:
    #    - 2x is in Sums(S)
    #    - x+a is in Sums(S) for some a in S
    #
    # Let's precompute for each pair (a, b) with a <= b, the sum s = a+b.
    # We can represent the set of used sums as a bitmask of size 2*N+1 (0 to 2N).
    # Max sum is 512. Bitmask size 513 bits. Python int handles this.
    
    # To check if x is addable to S:
    # new_sums = {2x} U {x+a for a in S}
    # If new_sums intersects Sums(S), x is not addable.
    
    # We want to find a small set S such that NO x in [1, N]\S is addable.
    # This is equivalent to: For every x not in S, x is blocked by S.
    
    # Strategy: Simulated Annealing / Local Search
    # State: A set S (represented as a sorted list or a boolean array).
    # Objective: Minimize |S|.
    # Constraints: 
    #   1. S is a Sidon set.
    #   2. S is maximal (no element can be added).
    
    # We can start with a valid maximal Sidon set (from a greedy heuristic) 
    # and then try to remove elements. But removal might break maximality.
    # If removal breaks maximality, we must add some other elements to restore maximality, 
    # but keep the total size small.
    
    # Better: Direct search over subsets.
    # Let's use a "repair" heuristic.
    
    # Precompute sum contributions for speed?
    # is_addable(x, S) is O(|S|).
    # is_maximal(S) is O(N * |S|).
    # With |S| ~ 10-20, this is fast.
    
    def get_sum_mask(S):
        mask = 0
        for i in S:
            mask |= (1 << (2*i))
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                mask |= (1 << (S[i] + S[j]))
        return mask

    def is_addable(x, S, sum_mask):
        # Check if x can be added to S
        # New sums: 2x, x+a for a in S
        # Conflict if any new sum is in sum_mask
        if (1 << (2*x)) & sum_mask:
            return False
        for a in S:
            if (1 << (x+a)) & sum_mask:
                return False
        return True

    def is_maximal(S, sum_mask):
        S_set = set(S)
        for x in range(1, N+1):
            if x not in S_set:
                if is_addable(x, S, sum_mask):
                    return False
        return True

    def is_valid_sidon(S):
        # Check if S is a Sidon set
        sum_mask = 0
        for i in S:
            if (1 << (2*i)) & sum_mask:
                return False
            sum_mask |= (1 << (2*i))
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                s = S[i] + S[j]
                if (1 << s) & sum_mask:
                    return False
                sum_mask |= (1 << s)
        return True

    def get_initial_solution():
        # Greedy construction of a maximal Sidon set
        S = []
        sum_mask = 0
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_addable(x, S, sum_mask):
                S.append(x)
                # Update sum_mask
                sum_mask |= (1 << (2*x))
                for a in S[:-1]:
                    sum_mask |= (1 << (x+a))
        
        # Ensure maximality (greedy usually produces maximal, but let's verify/fix)
        # If not maximal, add elements until maximal
        S_set = set(S)
        sum_mask = get_sum_mask(S)
        while True:
            added = False
            for x in range(1, N+1):
                if x not in S_set and is_addable(x, S, sum_mask):
                    S.append(x)
                    S_set.add(x)
                    sum_mask |= (1 << (2*x))
                    for a in S[:-1]:
                        sum_mask |= (1 << (x+a))
                    added = True
                    break
            if not added:
                break
        return S

    def local_search(S, time_limit):
        # S is a valid maximal Sidon set.
        # We try to reduce its size.
        # Move: Remove an element x from S.
        # If S\{x} is still maximal, accept (size decreases).
        # If S\{x} is not maximal, we need to add elements to make it maximal again.
        # To keep size small, we only add the MINIMUM number of elements needed to restore maximality.
        # This is a "repair" step.
        
        current_S = S[:]
        current_size = len(current_S)
        best_S = current_S[:]
        best_size = current_size
        
        # Precompute sum_mask for current_S
        current_sum_mask = get_sum_mask(current_S)
        
        # We will use a simple hill-climbing with restarts or simulated annealing.
        # Since the landscape is rugged, let's try random removals and repairs.
        
        # To speed up, we can precompute which elements are "critical" for blocking others.
        # But let's just try random removals.
        
        attempts = 0
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            attempts += 1
            if not current_S:
                break
                
            # Pick a random element to remove
            idx = random.randint(0, len(current_S) - 1)
            x = current_S[idx]
            
            # Try removing x
            new_S = current_S[:idx] + current_S[idx+1:]
            
            # Check if new_S is Sidon (it is, subset of Sidon set)
            # Check if new_S is maximal
            new_sum_mask = get_sum_mask(new_S)
            
            if is_maximal(new_S, new_sum_mask):
                # Accept removal
                current_S = new_S
                current_sum_mask = new_sum_mask
                if len(current_S) < best_size:
                    best_size = len(current_S)
                    best_S = current_S[:]
            else:
                # Not maximal. Try to repair by adding elements.
                # We want to add the FEWEST elements to make it maximal.
                # This is a Set Cover problem: Find min subset A of [1,N]\new_S 
                # such that new_S U A is maximal.
                # Heuristic: Greedily add elements that block the most "unblocked" numbers.
                
                # First, identify which numbers are NOT blocked by new_S.
                # blocked_by_new_S = {y | y is not addable to new_S}
                # unblocked = [1, N] \ (new_S U blocked_by_new_S)
                
                # Actually, we just need to add elements until no more can be added.
                # But we want to minimize the number added.
                # Let's try adding elements that are "high value" (block many others).
                
                # This repair is expensive. Let's limit the repair depth or try a different move.
                # Alternative move: Swap. Remove x, add y.
                # If we can find a y that restores maximality, size stays same.
                # But we want to DECREASE size. So removal is the key.
                
                # If removal fails, we could try a "swap" to a better position, 
                # but that doesn't reduce size.
                
                # Let's try a "2-opt" like move: Remove x, add y, then try to remove z.
                # Too complex.
                
                # Let's just accept the current state and try another removal.
                # Or, we can try to repair new_S by adding ONE element.
                # If adding one element makes it maximal, size becomes len(new_S)+1 = len(current_S).
                # This doesn't improve size, but might lead to a better configuration for future removals.
                
                # Let's try to find a y such that new_S U {y} is maximal.
                # If found, we can move to that state (same size, potentially better structure).
                
                found_repair = False
                # Candidate y's are those that are addable to new_S.
                # We want one that blocks the most remaining unblocked numbers.
                
                # For speed, let's just try a few random addable y's.
                addable_y = []
                for y in range(1, N+1):
                    if y not in new_S and is_addable(y, new_S, new_sum_mask):
                        addable_y.append(y)
                        
                if addable_y:
                    # Shuffle and try a few
                    random.shuffle(addable_y)
                    for y in addable_y[:5]: # Try up to 5
                        test_S = new_S + [y]
                        test_sum_mask = new_sum_mask | (1 << (2*y))
                        for a in new_S:
                            test_sum_mask |= (1 << (y+a))
                        
                        if is_maximal(test_S, test_sum_mask):
                            # Accept swap
                            current_S = test_S
                            current_sum_mask = test_sum_mask
                            found_repair = True
                            break
                            
                if not found_repair:
                    # No good swap found. Maybe we should restart from a new random maximal set?
                    # Or just continue.
                    pass
        
        return best_S

    # Main search loop
    best_result = None
    best_size = float('inf')
    
    # Run multiple iterations with different random seeds
    num_iterations = 50
    time_per_iter = (deadline - time.time()) / num_iterations
    
    for i in range(num_iterations):
        if time.time() > deadline - 5:
            break
            
        # Get an initial maximal Sidon set
        S0 = get_initial_solution()
        
        # Run local search with a time limit
        t_start = time.time()
        t_limit = min(time_per_iter, deadline - time.time() - 5)
        if t_limit < 0.1:
            break
            
        S_opt = local_search(S0, t_limit)
        
        if len(S_opt) < best_size:
            best_size = len(S_opt)
            best_result = S_opt[:]
            
        # Early exit if we find a very small set
        if best_size <= 10:
            break

    if best_result is None:
        # Fallback: return a valid maximal set
        best_result = get_initial_solution()
        
    return best_result
# EVOLVE-BLOCK-END
