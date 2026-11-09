"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    MAX_N = 512
    TARGET_SIZE = 48
    
    def is_valid_set(A):
        """
        Check if set A is valid.
        Valid means: No element is the mean of any subset of at least 2 OTHER elements.
        """
        n = len(A)
        if n < 3:
            return True
            
        # For each element x, check if it is the mean of a subset of A \ {x}
        for x in A:
            # We need to check if there exists S subset of A\{x}, |S|>=2, sum(S) = |S|*x
            # Equivalent to: sum(y - x for y in S) = 0
            diffs = [y - x for y in A if y != x]
            
            # DP to find if 0 is reachable with size >= 2
            # Max possible absolute sum: 48 * 512 = 24576
            # We use a dict or list. List is faster if range is known.
            # Range of sums: [-24576, 24576]
            OFFSET = 25000
            RANGE = 50001
            # dp[s] is a bitmask of sizes that can achieve sum s
            # Bit k is set if size k is possible
            dp = [0] * RANGE
            dp[OFFSET] = 1 # Sum 0 with size 0
            
            for d in diffs:
                # new_dp = dp.copy()
                # We iterate backwards to avoid using the same element twice?
                # No, we are adding elements one by one.
                # We need to update dp based on previous state.
                # Since we want to add d, we shift the existing sums by d.
                # new_dp[s+d] |= dp[s] << 1
                # We can do this in-place if we iterate carefully?
                # No, we need the state from BEFORE adding d.
                # So we copy or use a second array.
                # Given the range, copying 50k ints is fast.
                new_dp = dp[:]
                # Iterate through all possible sums
                # Optimization: only iterate through non-zero entries?
                # Or just full range.
                for s in range(RANGE):
                    if dp[s]:
                        new_s = s + d
                        if 0 <= new_s < RANGE:
                            new_dp[new_s] |= (dp[s] << 1)
                dp = new_dp
                
            # Check if sum 0 is reachable with size >= 2
            # dp[OFFSET] contains bits for sizes.
            # We need any bit k >= 2 to be set.
            # mask = ~3 (binary ...11111100)
            if (dp[OFFSET] & ~3) != 0:
                return False
                
        return True

    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        # Candidate pool
        candidates = list(range(1, MAX_N + 1))
        random.shuffle(candidates)
        
        selected = []
        
        for x in candidates:
            if len(selected) >= TARGET_SIZE:
                break
            
            # Try to add x
            new_selected = selected + [x]
            
            # Check validity
            # To speed up, we can check only the new element x against selected?
            # No, adding x can invalidate existing elements.
            # However, if we pick x such that it is "far" from others, it's less likely.
            # For now, full check.
            
            if is_valid_set(new_selected):
                selected = new_selected
                
        return selected

    best_set = []
    best_size = 0
    
    # Run multiple random restarts
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        s = greedy_construct(iterations)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            if best_size == TARGET_SIZE:
                break
                
    return best_set

# EVOLVE-BLOCK-END
