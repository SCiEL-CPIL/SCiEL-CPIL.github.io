import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds to be safe against 60s/90s limits
    
    best_seq = []
    best_len = 0
    
    # To avoid global state issues in recursion, we pass state
    def dfs(current_seq, last_val, sums_set, suffix_sums):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning: Can we possibly beat best_len?
        # Max possible additional elements: 256 - last_val
        if len(current_seq) + (256 - last_val) <= best_len:
            return
            
        # Update best
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        # Try adding next values
        # Heuristic: Try smaller values first to maximize length potential
        # We iterate from last_val + 1 to 256
        # However, if we are close to the deadline, we might want to be smarter.
        
        # Optimization: If the gap to 256 is small, just check them.
        
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            # Check if x can be added
            # New sums: x, x + suffix_sums[0], x + suffix_sums[1], ...
            # suffix_sums[k] is the sum of the last k+1 elements of current_seq
            # suffix_sums[0] = last_val
            # suffix_sums[1] = last_val + second_last_val
            
            # Check for collisions
            valid = True
            new_sums = []
            # The new sums are x + s for s in [0] + suffix_sums
            # Actually, the new consecutive blocks ending at x are:
            # [x], [prev, x], [prev2, prev, x], ...
            # Sums: x, x + last_val, x + last_val + second_last, ...
            # So we check x + s for s in [0] + suffix_sums
            
            # We need to check if any of these are in sums_set
            # And also check if any of the new sums collide with each other?
            # No, consecutive blocks of different lengths ending at x have different sums
            # because the sequence is strictly increasing (positive integers).
            # So x < x + last_val < x + last_val + second_last ...
            # So they are distinct among themselves.
            
            # We just need to check against existing sums_set
            
            # To speed up, we can break early if we find a collision
            # Also, we should collect the new sums to update sums_set if valid
            
            # Let's compute the new sums
            # s_0 = 0
            # s_i = suffix_sums[i-1] for i > 0
            # new_sum_i = x + s_i
            
            # Check x
            if x in sums_set:
                continue
                
            # Check x + suffix_sums[0]
            # ...
            
            # Efficient check:
            # Generate new sums and check intersection
            # We can do this in a loop
            
            collision = False
            new_sums_list = [x]
            # suffix_sums is a list where suffix_sums[i] is sum of last i+1 elements
            # So suffix_sums[0] is last_val
            # suffix_sums[1] is last_val + second_last
            # We need x + suffix_sums[i] for all i
            
            for s in suffix_sums:
                ns = x + s
                new_sums_list.append(ns)
                if ns in sums_set:
                    collision = True
                    break
            
            if collision:
                continue
                
            # If no collision, proceed
            # Update sums_set and suffix_sums
            new_sums_set = sums_set.union(new_sums_list)
            
            # New suffix sums:
            # The new sequence is current_seq + [x]
            # New suffix sums:
            # last 1: x
            # last 2: x + last_val
            # last 3: x + last_val + second_last
            # ...
            # So new_suffix_sums[0] = x
            # new_suffix_sums[i] = x + suffix_sums[i-1] for i >= 1
            
            new_suffix_sums = [x]
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
                
            # Recurse
            dfs(current_seq + [x], x, new_sums_set, new_suffix_sums)
            
            # Backtrack: No need to undo if we passed copies
            # But creating new lists/sets is expensive.
            # For performance, we might want to mutate and undo.
            # Given the depth is likely < 20, copying might be acceptable if the branching factor is controlled.
            # However, branching factor is up to 256.
            # Let's try to optimize by using mutable state if needed, but let's start with copies for correctness.
            
    # Initial call
    # We can start with any number. 
    # To maximize length, starting with small numbers is better.
    # We can iterate the first choice from 1 to 256.
    # But we can just call dfs with empty sequence?
    # dfs([], 0, set(), [])
    # If current_seq is empty, last_val is 0.
    # Next x in 1..256.
    # suffix_sums is empty.
    # new_sums_list = [x].
    # Check if x in sums_set (empty) -> OK.
    # new_suffix_sums = [x].
    # Recurse.
    
    dfs([], 0, set(), [])
    
    return best_seq

# Test locally
if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
