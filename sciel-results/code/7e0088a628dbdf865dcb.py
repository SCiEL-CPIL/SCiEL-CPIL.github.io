import random
import time

def construct():
    # Max possible sum for 30 elements of max 256
    MAX_SUM = 256 * 30 + 10
    # Precompute multiples for divisibility check
    # For a given bitset B, we need to check if any s in B divides another t in B.
    # This is equivalent to: for each s in B, check if B has any bits set at multiples of s.
    
    # To speed up the check, we can precompute for each s, a mask of its multiples?
    # Or just iterate.
    
    def is_valid(bits, max_sum):
        # Check if bits contains any pair s, t such that s < t and t % s == 0
        # Iterate through set bits
        # Optimization: Only check s up to max_sum // 2
        # We can extract set bits efficiently
        
        # A faster way: 
        # For each s from 1 to max_sum//2:
        #   if bit s is set:
        #      check if any bit 2s, 3s... is set.
        
        # We can precompute a list of multiples for each s?
        # Or just loop.
        
        # Let's use a helper to check if any multiple of s is in bits
        # We can use bit operations.
        
        # Actually, since MAX_SUM is ~7700, iterating s from 1 to 3850 is fine.
        
        for s in range(1, max_sum // 2 + 1):
            if (bits >> s) & 1:
                # Check multiples
                multiple = s * 2
                while multiple <= max_sum:
                    if (bits >> multiple) & 1:
                        return False
                    multiple += s
        return True

    # Greedy construction
    # We want to maximize the number of elements.
    # Let's try to build a set of size K.
    # We will use a randomized greedy approach with backtracking if necessary, 
    # but simple greedy with a good ordering might suffice for small K.
    
    # Let's try to find a set of size 10, 12, 15...
    
    best_set = []
    
    # Candidates
    candidates = list(range(1, 257))
    
    # Try different orderings
    # 1. Descending
    # 2. Randomized
    
    # To improve the score, we should try to find a larger set.
    # The current best is 3.
    
    # Let's run a search for a few seconds.
    start_time = time.time()
    timeout = 50  # Leave 10s buffer
    
    # We'll use a simple backtracking to try to extend the set.
    # State: current list, current bits
    # We try to add elements one by one.
    
    # To avoid getting stuck in a local minimum with a bad first element,
    # we can try random restarts.
    
    best_size = 0
    
    # Heuristic: Start with larger numbers to reduce divisibility risk.
    # But we need to allow all 1..256.
    
    # Let's try a greedy approach that picks the "best" candidate at each step.
    # "Best" is one that doesn't break validity.
    # To maximize size, we should pick numbers that are "least likely" to cause conflicts in the future.
    # This is hard to predict.
    
    # Simple strategy:
    # 1. Shuffle candidates.
    # 2. Iterate, if valid, add.
    # 3. Repeat with different shuffles.
    
    # To bias towards larger sets, we can try to force inclusion of certain numbers?
    # No, just random shuffles.
    
    # We can also try a "descending" shuffle.
    
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        # Randomize candidates, but maybe bias towards larger numbers?
        # Actually, small numbers are more dangerous (divisors).
        # So picking large numbers first is a good heuristic.
        
        # Let's try a mix: 
        # 50% chance descending, 50% chance random.
        
        if attempts % 2 == 0:
            cand = sorted(candidates, reverse=True)
        else:
            cand = candidates[:]
            random.shuffle(cand)
            
        current_set = []
        bits = 0
        current_sum = 0
        
        for x in cand:
            # Check if adding x is valid
            # New bits = bits | (bits << x) | (1 << x)
            # But we must ensure no collisions in the *new* sums that didn't exist before?
            # If bits has a collision, current_set is already invalid.
            # Since we only add if valid, bits represents a valid set of sums.
            
            new_bits = bits | (bits << x) | (1 << x)
            
            # Check divisibility in new_bits
            # Optimization: Only check sums that are NEW?
            # No, adding x creates new sums. Existing sums are already safe among themselves.
            # We only need to check:
            # 1. New sum divides an old sum.
            # 2. Old sum divides a new sum.
            # 3. New sum divides another new sum.
            
            # My is_valid(bits) checks ALL pairs. This is O(N^2) effectively.
            # It's fast enough for N=7700 bits.
            
            if is_valid(new_bits, MAX_SUM):
                bits = new_bits
                current_set.append(x)
                current_sum += x
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we reach a very high size, we can stop or continue
            if best_size >= 15:
                # Maybe break? Or keep trying to find 16?
                # Let's keep trying for a bit more.
                pass

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
