import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0
    best_seq = [1]
    best_len = 1

    def try_add(seq, sums, x):
        suffix_sums = [0]
        s = 0
        for a in reversed(seq):
            s += a
            suffix_sums.append(s)
        
        for ss in suffix_sums:
            if x + ss in sums:
                return False, None
        
        new_sums = set()
        for ss in suffix_sums:
            new_sums.add(x + ss)
        
        return True, new_sums

    def greedy_random(iterations=100, time_slice=5.0):
        best_local = []
        end_time = start_time + time_slice
        for _ in range(iterations):
            if time.time() > end_time:
                break
            seq = []
            sums = set()
            last = 0
            while True:
                valid_candidates = []
                for x in range(last + 1, 257):
                    valid, _ = try_add(seq, sums, x)
                    if valid:
                        valid_candidates.append(x)
                
                if not valid_candidates:
                    break
                
                if random.random() < 0.8:
                    x = valid_candidates[0]
                else:
                    x = random.choice(valid_candidates)
                
                valid, new_sums = try_add(seq, sums, x)
                if valid:
                    seq.append(x)
                    sums.update(new_sums)
                    last = x
                else:
                    break
            
            if len(seq) > len(best_local):
                best_local = seq.copy()
        
        return best_local

    def dfs_extend(initial_seq, target_len, deadline):
        seq = initial_seq.copy()
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        
        last = seq[-1] if seq else 0
        
        if 256 - last < target_len - len(seq):
            return False

        def recurse(depth):
            nonlocal seq, sums, last
            if time.time() > deadline:
                return False
            if depth == target_len:
                return True
            if 256 - last < target_len - depth:
                return False
            
            cands = list(range(last + 1, 257))
            cands.sort()
            
            for x in cands:
                valid, new_sums = try_add(seq, sums, x)
                if valid:
                    seq.append(x)
                    sums.update(new_sums)
                    old_last = last
                    last = x
                    
                    if recurse(depth + 1):
                        return True
                    
                    seq.pop()
                    sums.difference_update(new_sums)
                    last = old_last
            return False
        
        return recurse(len(seq))

    # 1. Generate a pool of candidate sequences
    # Run greedy for 15 seconds
    candidates = []
    end_greedy = start_time + 15.0
    while time.time() < end_greedy:
        seq = greedy_random(iterations=50, time_slice=2.0)
        if len(seq) > 0:
            candidates.append(seq)
    
    # Sort candidates by length
    candidates.sort(key=lambda x: len(x), reverse=True)
    
    # Keep top 5
    top_candidates = candidates[:5]
    
    if top_candidates:
        best_seq = top_candidates[0]
        best_len = len(best_seq)

    # 2. Try to extend each top candidate
    for cand in top_candidates:
        if time.time() - start_time > time_limit - 5:
            break
        
        current_len = len(cand)
        # Try to extend to current_len + 1, + 2, etc.
        # We limit the extension attempts to avoid spending too much time on one
        for target_len in range(current_len + 1, current_len + 10):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Give 2 seconds per extension attempt
            deadline = min(time.time() + 2.0, start_time + time_limit - 2)
            
            # We need to pass a copy or handle state carefully
            # dfs_extend modifies internal state, returns bool
            # If True, we need to retrieve the extended sequence
            # Let's modify dfs_extend to return the sequence or None
            
            # Revised dfs_extend to return sequence
            def dfs_extend_ret(initial_seq, target_len, deadline):
                seq = initial_seq.copy()
                sums = set()
                n = len(seq)
                for i in range(n):
                    s = 0
                    for j in range(i, n):
                        s += seq[j]
                        sums.add(s)
                
                last = seq[-1] if seq else 0
                
                if 256 - last < target_len - len(seq):
                    return None

                def recurse(depth):
                    nonlocal seq, sums, last
                    if time.time() > deadline:
                        return None
                    if depth == target_len:
                        return seq.copy()
                    if 256 - last < target_len - depth:
                        return None
                    
                    cands = list(range(last + 1, 257))
                    cands.sort()
                    
                    for x in cands:
                        valid, new_sums = try_add(seq, sums, x)
                        if valid:
                            seq.append(x)
                            sums.update(new_sums)
                            old_last = last
                            last = x
                            
                            res = recurse(depth + 1)
                            if res is not None:
                                return res
                            
                            seq.pop()
                            sums.difference_update(new_sums)
                            last = old_last
                    return None
                
                return recurse(len(seq))

            extended = dfs_extend_ret(cand, target_len, deadline)
            if extended is not None:
                if len(extended) > best_len:
                    best_seq = extended
                    best_len = len(extended)
                # If we found an extension, we can try to extend further from this new best
                # But the loop continues to target_len + 1, which is good.
                # However, `cand` is the old one. We should update `cand` to `extended`?
                # No, the loop is for `target_len` in range.
                # If we found length L+1, we try L+2 next.
                # The base for L+2 should be the sequence of length L+1.
                # So we should update the base sequence for this candidate.
                # Let's do that.
                cand = extended
            else:
                # Failed to extend to target_len
                # Try next target_len? No, if we can't reach L+1, we can't reach L+2.
                # Break out of target_len loop
                break

    return best_seq
