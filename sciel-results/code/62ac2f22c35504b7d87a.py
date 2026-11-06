import time

def construct():
    deadline = time.time() + 85  # 85s safety margin (limit is 90s)
    
    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums
    
    # Get greedy baseline
    base_seq, base_sums = greedy()
    best = base_seq
    
    # Backtracking to extend beyond greedy
    def backtrack(seq, sums):
        nonlocal best
        
        if time.time() > deadline:
            return
        
        current_len = len(seq)
        if current_len > len(best):
            best = list(seq)
        
        last = seq[-1] if seq else 0
        
        # Pruning: If we can't possibly beat the best, return
        # Max elements we can add is (256 - last) since min gap is 1
        if current_len + (256 - last) <= len(best):
            return

        # Try candidates from last+1 to 256
        for x in range(last + 1, 257):
            if time.time() > deadline:
                return
            
            # Compute new tails
            tails = []
            total = x
            tails.append(total)
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            # Check if any tail is in sums
            if any(t in sums for t in tails):
                continue
            
            # Valid - recurse
            seq.append(x)
            sums.update(tails)
            backtrack(seq, sums)
            sums.difference_update(tails)
            seq.pop()
            
            # Optional: If we found a solution longer than best, we might prune more aggressively?
            # The pruning above handles it.

    # Try to extend the greedy solution
    backtrack(list(base_seq), set(base_sums))
    
    return best
