import time
import random

def construct():
    start_time = time.time()
    time_limit = 58.0
    
    # Baseline: Powers of 2 up to 256
    # 1, 2, 4, 8, 16, 32, 64, 128, 256
    # Length 9
    baseline = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_seq = baseline[:]
    best_len = len(best_seq)
    
    def get_sums(seq):
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        return sums

    # Precompute sums for baseline
    current_sums = get_sums(baseline)

    def dfs(current_seq, last_val, current_sums, depth):
        nonlocal best_seq, best_len
        
        if time.time() > time_limit:
            return False
        
        if depth > best_len:
            best_seq = current_seq[:]
            best_len = depth
            
        # If we have used all numbers or reached a reasonable depth limit for this branch
        # We don't have a hard depth limit other than 256, but practically we won't reach it.
        
        # Generate candidates
        # We want to try values > last_val
        # To avoid getting stuck in a specific local optimum, we shuffle
        candidates = list(range(last_val + 1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Calculate new sums
            # New sums are x, x+last, x+last+prev, ...
            new_sums = set(current_sums)
            s = x
            # We need to check if any of these new sums are in current_sums
            # And also if they are distinct among themselves (they are, because seq is increasing and positive)
            # Wait, are they distinct among themselves?
            # s_1 = x
            # s_2 = x + a_{L}
            # s_3 = x + a_{L} + a_{L-1}
            # Since a_i > 0, s_1 < s_2 < s_3 ...
            # So new sums are distinct.
            # We just need to check if any new sum is in current_sums.
            
            conflict = False
            s = x
            if s in current_sums:
                conflict = True
            else:
                # Add to new_sums later if no conflict
                pass
            
            if not conflict:
                # Iterate backwards through current_seq to build suffix sums
                # We can do this in a loop
                temp_s = x
                for a in reversed(current_seq):
                    temp_s += a
                    if temp_s in current_sums:
                        conflict = True
                        break
            
            if conflict:
                continue
            
            # If no conflict, we can recurse
            # We need to pass the updated sums set
            # Creating a new set is O(N) where N is number of sums.
            # Number of sums is L(L+1)/2. For L=20, 210 sums.
            # This is fast enough.
            
            new_sums_set = set(current_sums)
            temp_s = x
            new_sums_set.add(temp_s)
            for a in reversed(current_seq):
                temp_s += a
                new_sums_set.add(temp_s)
                
            if dfs(current_seq + [x], x, new_sums_set, depth + 1):
                return True
                
        return False

    # Run DFS from the baseline
    # We might want to restart from scratch or from different prefixes
    # Let's run DFS from the baseline first
    
    # To allow for different exploration, we can run multiple DFS instances
    # but the global timer prevents running many long ones.
    # One deep DFS with randomization is likely sufficient to find a good sequence.
    
    # Start DFS
    # We pass the baseline state
    # Note: dfs returns True if it finds a solution deeper than best_len?
    # My implementation updates best_seq inside. It returns True if it successfully extends?
    # Actually, the return value logic in my draft is a bit weak.
    # Let's just let it run and update best_seq.
    
    dfs(baseline, baseline[-1], current_sums, len(baseline))
    
    return best_seq
