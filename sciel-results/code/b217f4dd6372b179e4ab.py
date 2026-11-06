import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin
    
    # Initialize
    N = 100
    target = 200
    # x is a boolean array of length N+1 (1-indexed)
    x = np.ones(N+1, dtype=bool)
    
    best_valid_set = []
    
    # Helper: Find a subset of current x summing to target
    def find_violation(x, target, N):
        # DP to find if target is reachable
        # dp[s] = True if sum s is reachable
        # We also need to reconstruct the set
        # For reconstruction, we can store the last element used
        dp = np.zeros(target + 1, dtype=bool)
        dp[0] = True
        parent = np.zeros((target + 1, N + 1), dtype=int) # parent[s][i] = 1 if i was used to reach s? No, standard backtracking is easier with a set of reachable sums and storing predecessors.
        
        # Efficient reconstruction:
        # reachable_sums = {0}
        # prev = {0: None}
        # This might be slow in Python.
        
        # Let's use a standard DP array for reachability and a separate way to backtrack.
        # Actually, for N=100, K=200, a simple set-based DP is fast enough.
        
        reachable = {0}
        # To backtrack, we need to know which items were added.
        # We can store for each sum, the item that completed it.
        # But multiple items can complete a sum.
        
        # Simpler: Just find IF it's reachable.
        # If reachable, we need ONE subset.
        # We can use a recursive backtracking or store a witness.
        
        # Let's use a dictionary: sum -> (prev_sum, item)
        witness = {0: None}
        curr_reachable = {0}
        
        for i in range(1, N+1):
            if not x[i]:
                continue
            # Update reachable sums
            # We need to be careful not to use the same item twice.
            # Since we iterate i, and update based on previous state, it's fine.
            new_sums = set()
            for s in curr_reachable:
                ns = s + i
                if ns <= target:
                    if ns not in witness:
                        witness[ns] = (s, i)
                    new_sums.add(ns)
            curr_reachable |= new_sums
            
            if target in curr_reachable:
                # Backtrack
                subset = []
                curr_s = target
                while witness[curr_s] is not None:
                    prev_s, item = witness[curr_s]
                    subset.append(item)
                    curr_s = prev_s
                return subset
        
        return None

    # Greedy Removal Loop
    # To improve on simple greedy, we can try removing the "best" element from the violation.
    # "Best" element: one that minimizes the number of future violations?
    # Approximation: Remove the element that is in the most other potential violations?
    # Hard to compute.
    
    # Let's use a randomized restart or local search.
    # Since N is small, we can try a simple heuristic:
    # When a violation J is found, try removing each x in J.
    # Pick the one that leaves the "easiest" state.
    # "Easiest" state: Fewest reachable sums close to target?
    
    while time.time() < deadline:
        violation = find_violation(x, target, N)
        if violation is None:
            # Valid set found
            current_set = [i for i in range(1, N+1) if x[i]]
            if len(current_set) > len(best_valid_set):
                best_valid_set = current_set
            # Try to see if we can add back any removed elements?
            # This is hard. Let's just return if we are confident.
            # Or continue searching for a larger one?
            # Since we started with all 100 and only removed, we can't get larger than current.
            # So this is the max for this path.
            # We can break and return.
            return best_valid_set if best_valid_set else [1] # Ensure non-empty
            
        # We have a violation.
        # Choose an element to remove.
        # Heuristic: Remove the element in violation that has the highest value?
        # Or the one that appears in the most "dense" parts of the sum space?
        
        # Let's try: Remove the element that minimizes the number of reachable sums in the next step?
        # This is expensive.
        
        # Simple Greedy: Remove the largest element in the violation.
        # Why? Large elements are more "expensive" in terms of sum, so removing them might break fewer other sums?
        # Actually, small elements are more versatile. Removing a small element might break many sums.
        # Removing a large element breaks fewer sums.
        # So we want to KEEP small elements?
        # If we keep small elements, we can form many sums.
        # So we should REMOVE small elements to prevent forming sums?
        # Wait. If we remove small elements, we reduce the number of possible sums.
        # So removing small elements is better for avoiding sums.
        # Therefore, prefer removing the SMALLEST element in the violation.
        
        # Let's test this hypothesis.
        # Violation J. Remove min(J).
        
        # Let's try a few candidates and pick the one that results in the smallest "reachable count" immediately after removal.
        candidates = sorted(violation) # Smallest first
        
        best_candidate = candidates[0]
        min_reachable_count = float('inf')
        
        # To speed up, just pick the smallest one.
        # If we want to be smarter, we can simulate.
        # Given time, let's stick to a slightly smarter greedy:
        # Try each x in violation (if |violation| is small, say < 10).
        # If |violation| is large, just pick the smallest.
        
        if len(violation) <= 10:
            for x_val in violation:
                # Temporarily remove x_val
                x[x_val] = False
                # Check how many sums are reachable?
                # This is O(N*K) per candidate.
                # 10 * 100 * 200 = 200,000 ops. Fast.
                curr_reachable = {0}
                for i in range(1, N+1):
                    if not x[i]:
                        continue
                    new_sums = set()
                    for s in curr_reachable:
                        ns = s + i
                        if ns <= target:
                            new_sums.add(ns)
                    curr_reachable |= new_sums
                count = len(curr_reachable)
                if count < min_reachable_count:
                    min_reachable_count = count
                    best_candidate = x_val
                # Restore
                x[x_val] = True
        
        # Remove the best candidate
        x[best_candidate] = False
        
        # Update best valid set?
        # The current set is NOT valid.
        # We only update best_valid_set when we find a valid one.
        
    # If loop finishes without finding valid (should not happen as we eventually remove all), return best found
    if not best_valid_set:
         # Fallback: Return a known valid set, e.g. {100} or seed result
         # Seed result is valid.
         return [100]
         
    return best_valid_set
