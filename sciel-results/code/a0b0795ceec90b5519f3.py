import time
import sys

# Global variables
best_set = []
best_size = 0
start_time = 0
time_limit = 115.0

SUMS_SIZE = 570

def construct():
    global best_set, best_size, start_time
    
    start_time = time.time()
    
    initial_set = [1, 2, 5, 11]
    initial_sums = bytearray(SUMS_SIZE)
    
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            s = initial_set[i] + initial_set[j]
            initial_sums[s] = 1
            
    best_set = initial_set[:]
    best_size = len(initial_set)
    
    candidates = []
    for x in range(12, 285):
        valid = True
        if 2 * x < SUMS_SIZE and initial_sums[2 * x]:
            valid = False
        if valid:
            for y in initial_set:
                s = x + y
                if s < SUMS_SIZE and initial_sums[s]:
                    valid = False
                    break
        if valid:
            candidates.append(x)
            
    # Heuristic: Sort candidates.
    # Descending often works for spreading marks.
    candidates.sort(reverse=True)
    
    K = len(candidates)
    
    # Pre-compute candidate indices
    # We will use a bitmask of size K.
    
    def dfs(current_set, current_sums, avail_mask, target_size):
        global best_set, best_size
        
        if time.time() - start_time > time_limit:
            return False
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if current_size == target_size:
            return True
            
        # Pruning: Upper bound
        # Python 3.10+ has int.bit_count()
        # Fallback: bin(avail_mask).count('1')
        remaining = avail_mask.bit_count() if hasattr(avail_mask, 'bit_count') else bin(avail_mask).count('1')
        if current_size + remaining < target_size:
            return False
            
        # Iterate through available candidates
        mask = avail_mask
        while mask:
            # Pick the lowest set bit (or highest? Let's try lowest index in sorted list, which is largest value)
            # Since candidates are sorted descending, index 0 is largest.
            # lsb gives smallest index.
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            mask ^= lsb
            
            x = candidates[i]
            
            # Check if x can be added
            # 1. 2*x
            if 2 * x < SUMS_SIZE and current_sums[2 * x]:
                continue
                
            # 2. x + y for y in current_set
            can_add = True
            for y in current_set:
                s = x + y
                if s < SUMS_SIZE and current_sums[s]:
                    can_add = False
                    break
                    
            if not can_add:
                continue
                
            # Add x
            current_set.append(x)
            current_sums[2 * x] = 1
            for y in current_set:
                if y != x:
                    current_sums[x + y] = 1
            
            # Compute new_avail_mask
            new_mask = avail_mask & ~lsb  # Remove x
            
            # Filter out candidates that conflict with x given current_set
            # A candidate v (index j) is invalid if:
            # 1. 2*v is in current_sums (updated)
            # 2. v+y is in current_sums (updated) for any y in current_set
            
            # To be efficient, we iterate through the remaining bits in new_mask
            # and check validity.
            # This is O(K * |S|) per node.
            
            temp_mask = new_mask
            filtered_mask = 0
            while temp_mask:
                lsb_j = temp_mask & -temp_mask
                j = lsb_j.bit_length() - 1
                temp_mask ^= lsb_j
                
                v = candidates[j]
                
                # Check v against updated current_sums
                if 2 * v < SUMS_SIZE and current_sums[2 * v]:
                    continue
                valid_v = True
                for y in current_set:
                    s = v + y
                    if s < SUMS_SIZE and current_sums[s]:
                        valid_v = False
                        break
                if valid_v:
                    filtered_mask |= lsb_j
            
            if dfs(current_set, current_sums, filtered_mask, target_size):
                return True
                
            # Backtrack
            current_set.pop()
            current_sums[2 * x] = 0
            for y in current_set:
                current_sums[x + y] = 0
                
        return False

    # Try sizes from 20 down to best_size + 1
    for target in range(20, best_size, -1):
        if time.time() - start_time > time_limit - 5:
            break
            
        init_set_copy = initial_set[:]
        init_sums_copy = initial_sums[:]
        
        # Initial mask: all candidates are available
        init_mask = (1 << K) - 1
        
        if dfs(init_set_copy, init_sums_copy, init_mask, target):
            return best_set
            
    return best_set
