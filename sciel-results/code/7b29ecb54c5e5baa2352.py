import sys
import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # Leave 5s buffer

    # We will attempt to find the maximum length sequence.
    # The known maximum for this problem (consecutive blocks distinct) is around 10-15 for small bounds?
    # Actually, for [1, 256], the length can be quite long.
    # Let's start with a greedy baseline to get a lower bound.
    
    def get_greedy():
        values = []
        sums = set()
        for x in range(1, 257):
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
    best_len = len(baseline)
    best_seq = baseline

    # Helper to check if a candidate x is valid given current seq and sums
    def get_new_sums(x, seq):
        new_sums = [x]
        s = x
        for a in reversed(seq):
            s += a
            new_sums.append(s)
        return new_sums

    def is_valid(x, seq, used_sums):
        new_sums = get_new_sums(x, seq)
        for s in new_sums:
            if s in used_sums:
                return False
        return True

    # 1-step lookahead: check if there is at least one valid y > x for the next position
    # This is expensive, so we only do it if the remaining search space is small enough or periodically.
    # Actually, for small target lengths, it's fast.
    def has_valid_next(x, seq_with_x, used_sums_with_x, target_len, current_idx):
        if current_idx + 1 >= target_len:
            return True # This is the last element
        
        # We need to find at least one y > x such that is_valid(y, seq_with_x, used_sums_with_x)
        # To be fast, we can just check a few candidates or use a heuristic.
        # If we check all, it's O(N * N).
        # Let's just check if the *smallest* possible next value (x+1) is valid?
        # No, x+1 might be invalid but x+2 valid.
        # For pruning, we want to know if the *set* of valid next values is empty.
        # If we don't check all, we might miss a valid path.
        # However, checking all y in [x+1, 256] is O(256 * len(seq)).
        # With len(seq) ~ 10-20, this is ~5000 ops.
        # If we do this for every candidate x in DFS, it might be slow.
        # Let's only do full lookahead if the number of candidates is small or we are deep in the search.
        
        # Heuristic: If the gap between x and 256 is small, check all.
        # Otherwise, check a sample.
        
        # Let's just check if ANY y works.
        # Optimization: We can break early if we find one.
        for y in range(x + 1, 257):
            if is_valid(y, seq_with_x, used_sums_with_x):
                return True
        return False

    def dfs(target_len, seq, used_sums, idx, last_val):
        if time.time() - start_time > TIME_LIMIT:
            return False
        
        if idx == target_len:
            return True
            
        # Pruning: Not enough numbers left
        if 256 - last_val < target_len - idx:
            return False
            
        candidates = list(range(last_val + 1, 257))
        
        # Heuristic: Try smaller values first?
        # Smaller values tend to create smaller sums, which might collide more easily?
        # Or do they leave room for larger sums later?
        # The greedy method (smallest first) is a good baseline.
        # Let's stick to smallest first, but use lookahead.
        
        for x in candidates:
            # Basic validity check
            if not is_valid(x, seq, used_sums):
                continue
            
            # Lookahead check
            # To save time, we only do the full lookahead if we are close to the limit
            # or if the candidate count is low.
            # Actually, let's just do it. It's a strong pruner.
            # But wait, is_valid is O(L). has_valid_next is O(N*L).
            # Total per node: O(N*L^2).
            # If L=10, N=256, that's 25600 ops per node.
            # If we have 1000 nodes, that's 25M ops. Should be fine in 55s.
            
            seq_with_x = seq + [x]
            new_sums = get_new_sums(x, seq)
            used_sums_with_x = used_sums.union(new_sums)
            
            if has_valid_next(x, seq_with_x, used_sums_with_x, target_len, idx):
                if dfs(target_len, seq_with_x, used_sums_with_x, idx + 1, x):
                    return True
                    
        return False

    # Try to extend the sequence
    # We start from best_len + 1
    # We use random restarts if the deterministic (smallest first) fails?
    # The DFS above is deterministic (smallest first).
    # If it fails, maybe a different order works?
    
    # Let's add a randomized component to the candidate order if deterministic fails.
    # But first, let's try deterministic DFS with lookahead.
    
    for target_len in range(best_len + 1, best_len + 15):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        found = False
        # Try deterministic search
        if dfs(target_len, [], set(), 0, 0):
            # Re-run to get the actual sequence? 
            # The dfs above returns bool. We need to store the sequence.
            # Let's modify dfs to return the sequence or store it in a nonlocal/global.
            pass
            
        # The previous dfs didn't store the result. Let's rewrite it to store the best sequence found.
        
        # Refactored DFS that stores the solution
        def dfs_store(target_len, seq, used_sums, idx, last_val, result_container):
            if time.time() - start_time > TIME_LIMIT:
                return False
            
            if idx == target_len:
                result_container[0] = seq[:]
                return True
                
            if 256 - last_val < target_len - idx:
                return False
                
            candidates = list(range(last_val + 1, 257))
            
            # Try smallest first
            for x in candidates:
                if not is_valid(x, seq, used_sums):
                    continue
                    
                seq_with_x = seq + [x]
                new_sums = get_new_sums(x, seq)
                used_sums_with_x = used_sums.union(new_sums)
                
                # Lookahead
                if has_valid_next(x, seq_with_x, used_sums_with_x, target_len, idx):
                    if dfs_store(target_len, seq_with_x, used_sums_with_x, idx + 1, x, result_container):
                        return True
                        
            return False

        result = [None]
        if dfs_store(target_len, [], set(), 0, 0, result):
            best_seq = result[0]
            best_len = target_len
        else:
            # If deterministic fails, try a few randomized searches
            # But the deterministic search with lookahead is quite strong.
            # If it fails, it's likely impossible or very hard.
            # Let's break to save time for other lengths? 
            # No, maybe L is impossible but L+1 is? Unlikely.
            break
            
    return best_seq
