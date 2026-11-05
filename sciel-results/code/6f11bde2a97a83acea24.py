import time
import sys

def construct():
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin
    
    # 1. Run the seed greedy to get a baseline length
    def greedy_construct():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    baseline_seq = greedy_construct()
    best_seq = baseline_seq
    best_len = len(best_seq)
    
    # 2. Try to find longer sequences using backtracking
    # We will try to find a sequence of length L for L = best_len + 1, best_len + 2, ...
    
    def check_valid(seq):
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Pre-compute prefix sums for faster sum calculation in DFS?
    # In DFS, we maintain the list of suffix sums for the current sequence.
    # Let suffix_sums be a list where suffix_sums[i] is the sum of seq[-i-1:]
    # Actually, the seed computes "tails".
    # tails = [x, x+last, x+last+prev, ...]
    # These are exactly the new sums introduced by appending x.
    
    def dfs(seq, used_sums, last_val, target_len, suffix_sums):
        """
        seq: current sequence
        used_sums: set of all sums so far
        last_val: last value in seq
        target_len: desired length
        suffix_sums: list of sums of suffixes of seq. 
                     suffix_sums[0] = seq[-1]
                     suffix_sums[1] = seq[-1] + seq[-2]
                     ...
        """
        nonlocal best_seq, best_len, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if len(seq) == target_len:
            best_seq = seq[:]
            best_len = target_len
            return True
            
        if last_val >= 256:
            return False
            
        # Heuristic: Try smaller x first
        for x in range(last_val + 1, 257):
            # Calculate new suffix sums
            # New suffixes are:
            # x
            # x + seq[-1]
            # x + seq[-1] + seq[-2]
            # ...
            # x + sum(seq)
            
            new_suffixes = [x]
            current_sum = x
            # Iterate through existing suffix sums
            # suffix_sums contains sums of suffixes of the previous seq.
            # Let's say prev_seq = [a1, ..., ak]
            # suffix_sums[0] = ak
            # suffix_sums[1] = ak + a{k-1}
            # ...
            # suffix_sums[k-1] = sum(prev_seq)
            
            # New suffixes for new_seq = [a1, ..., ak, x]
            # 1. x
            # 2. x + ak
            # 3. x + ak + a{k-1}
            # ...
            
            # So new_suffixes[i] = x + suffix_sums[i-1] for i > 0
            # And new_suffixes[0] = x
            
            valid = True
            new_suffixes_list = [x]
            if x in used_sums:
                valid = False
            
            if valid:
                for s in suffix_sums:
                    current_sum += s
                    new_suffixes_list.append(current_sum)
                    if current_sum in used_sums:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # If valid, recurse
            # New used_sums = used_sums | set(new_suffixes_list)
            # New suffix_sums = new_suffixes_list
            
            new_used_sums = used_sums.union(new_suffixes_list)
            
            if dfs(seq + [x], new_used_sums, x, target_len, new_suffixes_list):
                return True
                
        return False

    # Start iterative deepening
    # We know baseline length. Try to find longer.
    # To avoid infinite loops or slow searches, we cap the target length.
    # Max possible length is limited by range 1-256.
    # Let's try up to 30 or until time runs out.
    
    current_target = best_len + 1
    max_target = 30  # Arbitrary cap, adjust if needed
    
    while current_target <= max_target:
        # Initialize DFS state
        # We need to start from empty? No, we can start from scratch.
        # But we can prune heavily.
        
        # It's more efficient to search for a specific length from scratch
        # or extend the baseline?
        # Extending the baseline might be stuck in a local optimum.
        # Searching from scratch for target_len allows finding a completely different structure.
        
        # However, starting from scratch with target_len > best_len is hard.
        # Let's try to find ANY sequence of length current_target.
        
        # To speed up, we can use the baseline sequence as a hint?
        # Or just run DFS.
        
        # DFS from empty
        found = dfs([], set(), 0, current_target, [])
        
        if found:
            # We found a sequence of length current_target
            # Update best
            current_target += 1
        else:
            # No sequence of this length found (or timeout)
            break

    return best_seq
