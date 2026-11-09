import time
import sys

START_TIME = time.time()
DEADLINE = 55.0  # Leave 5s for safety

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_conflict(dp):
    """
    dp is a list of sets. dp[k] contains sums of subsets of size k.
    Returns True if there is a conflict (intersection between dp[i] and dp[j] for i != j).
    """
    n = len(dp)
    # Optimization: only check non-empty sets
    non_empty = [i for i in range(n) if dp[i]]
    for i in range(len(non_empty)):
        for j in range(i+1, len(non_empty)):
            if dp[non_empty[i]] & dp[non_empty[j]]:
                return True
    return False

def construct():
    best_set = []
    
    # Try to find a set of size K
    # Max possible size is bounded by log2(256 * K) roughly? 
    # Powers of 2 give 9.
    # Let's search from 15 down to 1.
    
    for K in range(15, 0, -1):
        if is_done():
            break
            
        # Backtracking search for a set of size K
        # We use a stack to avoid recursion depth issues, or just recursion with limit
        # Recursion is fine for depth 15.
        
        def dfs(idx, current_set, dp):
            nonlocal best_set
            if is_done():
                return
            
            if len(current_set) == K:
                best_set = current_set[:]
                return True # Found a solution
            
            # Pruning: if we can't reach K, return
            # Max elements we can add is 256 - current_set[-1]
            # But we need to be smarter.
            
            # Heuristic: try to add elements that are "safe"
            # Start from current_set[-1] + 1 if current_set else 1
            start_val = current_set[-1] + 1 if current_set else 1
            
            # To improve speed, we can try a limited number of candidates
            # or use a smarter heuristic.
            # For now, let's iterate.
            
            for v in range(start_val, 257):
                if is_done():
                    return False
                
                # Update DP
                new_dp = [set(s) for s in dp] # Shallow copy of list, deep copy of sets?
                # Actually, we can update in place if we backtrack carefully, 
                # but copying is safer and fast enough for small K.
                
                # Update sums
                # new_dp[k] = dp[k] union {s + v | s in dp[k-1]}
                # We iterate k from current_size down to 0?
                # dp has size K+1 (indices 0..K)
                curr_size = len(current_set)
                
                # We only need to update up to curr_size + 1
                # new_dp[k] for k > curr_size + 1 is empty
                
                # Let's just compute the new sums for k = curr_size + 1 down to 1
                # And keep the old sums for k <= curr_size?
                # No, the old sums are still valid subsets.
                # So new_dp[k] = dp[k] for k <= curr_size
                # new_dp[curr_size + 1] = {s + v | s in dp[curr_size]}
                
                # Wait, this is wrong.
                # If we add v, we can form subsets of size k using v or not.
                # If not using v, the sum is in dp[k].
                # If using v, the sum is s + v where s is in dp[k-1].
                # So new_dp[k] = dp[k] union {s + v | s in dp[k-1]}
                
                # We must ensure we don't use v twice.
                # Since we are building incrementally, dp[k-1] does not contain v.
                
                # Implementation:
                # new_dp = [set(s) for s in dp]
                for k in range(1, curr_size + 2):
                    if k <= len(dp) and dp[k-1]:
                        new_sums = {s + v for s in dp[k-1]}
                        new_dp[k] = new_dp[k] | new_sums
                
                # Check conflict
                if check_conflict(new_dp):
                    continue
                
                # Recurse
                if dfs(idx + 1, current_set + [v], new_dp):
                    return True
                
            return False
        
        # Initialize dp for empty set
        # dp[0] = {0}, dp[k>0] = empty
        init_dp = [set() for _ in range(K + 1)]
        init_dp[0] = {0}
        
        if dfs(0, [], init_dp):
            # Found a valid set of size K
            return best_set
        
    # If no set found (should not happen, size 1 is always valid)
    return [1]

# Fallback if something goes wrong
if __name__ == "__main__":
    print(construct())
