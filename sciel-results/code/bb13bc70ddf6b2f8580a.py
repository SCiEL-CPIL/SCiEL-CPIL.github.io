import sys
import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # Leave 5s buffer

    def get_greedy():
        values, sums = [], set()
        for x in range(1, 257):
            # Calculate suffix sums ending at x
            # Sequence: ... a_{i-1}, a_i, x
            # Sums: x, x+a_i, x+a_i+a_{i-1}, ...
            # We need to check if these sums are in `sums`.
            # Note: `sums` contains sums of ALL consecutive blocks in `values`.
            # The new blocks are exactly the suffixes ending at x.
            
            # Optimization: We can store the sequence and recompute, 
            # but for greedy, we can just keep the list.
            # Actually, we can just iterate.
            
            # Let's just use the simple loop as in the seed, it's fast enough for N=256.
            new_sums = []
            s = 0
            for a in reversed(values):
                s += a
                new_sums.append(s)
            # Wait, the loop above misses x itself.
            # Suffixes of [..., a_last, x]:
            # 1. [x] -> sum x
            # 2. [a_last, x] -> sum a_last + x
            # 3. [a_prev, a_last, x] -> sum a_prev + a_last + x
            
            # Correct calculation:
            new_sums = [x]
            s = x
            for a in reversed(values):
                s += a
                new_sums.append(s)
                
            if not sums.intersection(new_sums):
                values.append(x)
                sums.update(new_sums)
        return values

    baseline = get_greedy()
    best_seq = baseline
    best_len = len(baseline)
    
    # Try to extend
    # We will try to find sequences of length best_len + 1, +2, etc.
    
    # Helper for backtracking
    # To make it fast, we pass around the current sequence and the set of sums.
    # We also pass the index we are filling.
    
    def try_find_length(target_len, max_nodes=100000, seed=None):
        if seed is not None:
            random.seed(seed)
            
        # We use a list for the sequence to avoid creating new lists at every step
        seq = [0] * target_len
        sums = set()
        
        def backtrack(idx, last_val):
            if time.time() - start_time > TIME_LIMIT:
                return False
            
            if idx == target_len:
                return True
            
            # Pruning: not enough numbers left
            if 256 - last_val < target_len - idx:
                return False
                
            # Candidates: last_val + 1 to 256
            # We want to try candidates that are "good".
            # Random shuffle helps escape local traps.
            candidates = list(range(last_val + 1, 257))
            random.shuffle(candidates)
            
            # Heuristic: Maybe try smaller numbers first? 
            # Or maybe numbers that are "far" from others?
            # Let's stick to random for now, but we can bias towards smaller 
            # to keep sums small? No, distinctness is the key.
            
            for x in candidates:
                # Calculate new sums
                # Suffixes ending at x
                new_sums = [x]
                s = x
                # We need the previous elements: seq[0:idx]
                for i in range(idx - 1, -1, -1):
                    s += seq[i]
                    new_sums.append(s)
                    
                # Check conflict
                conflict = False
                for ns in new_sums:
                    if ns in sums:
                        conflict = True
                        break
                        
                if conflict:
                    continue
                
                # Accept
                seq[idx] = x
                sums.update(new_sums)
                
                if backtrack(idx + 1, x):
                    return True
                    
                # Backtrack
                for ns in new_sums:
                    sums.remove(ns)
                    
            return False

        # Initial call
        # We can try multiple times with different seeds if the first fails quickly
        # But we have a time limit.
        
        # To improve success, we can try to fill from the end? 
        # No, increasing sequence is fixed order.
        
        # Let's run the search.
        # We might need to run it multiple times if it fails, but time is limited.
        # Let's just run it once with a high node limit, but time check inside.
        
        # To make it more robust, we can try a few different random seeds if it fails fast.
        attempts = 0
        while time.time() - start_time < TIME_LIMIT - 5:
            random.seed() # Fresh random
            if backtrack(0, 0):
                return seq[:]
            attempts += 1
            # If we've tried many times, maybe stop?
            # The time check inside backtrack will stop it.
            # But if backtrack returns False quickly, we loop.
            # Let's add a small sleep or node count limit per attempt?
            # The node count isn't tracked globally here easily without nonlocal.
            # Let's rely on time.
            
        return None

    # Try to find longer sequences
    # Start from baseline length + 1
    # Go up to maybe 25?
    
    for target_len in range(best_len + 1, best_len + 10):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Try to find a sequence of this length
        # We can try multiple random seeds
        found = None
        # Let's try up to 5 random restarts for each length
        for i in range(5):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            res = try_find_length(target_len)
            if res:
                found = res
                break
                
        if found:
            best_seq = found
            best_len = target_len
        else:
            # If we failed to find length L, it might be possible with better search,
            # but if we failed for L, it's unlikely we'll find L+1 easily.
            # However, sometimes L is impossible but L+1 is? Unlikely.
            # Let's stop if we fail for a length, assuming monotonic difficulty.
            # Actually, it's possible that L is hard but L+1 is easy? No, L+1 contains L as a subsequence?
            # Not necessarily. But generally, longer is harder.
            # Let's break if we fail.
            break
            
    return best_seq
