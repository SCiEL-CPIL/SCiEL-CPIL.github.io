import time

def construct():
    start_time = time.time()
    deadline = start_time + 50

    # 1. Greedy Baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
            
    best_seq = values
    best_len = len(values)
    
    # 2. Try to find a longer sequence
    # Target: Length 13
    # Strategy: Find a prefix of length 12 with sum <= 255, then append sum+1.
    # Or find a prefix of length 11 with sum <= 127, then append sum+1, 2*sum+1.
    
    # Backtracking search
    def search(vals, sums_set, last_val, current_sum, target_prefix_len, max_sum):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return False
        
        if len(vals) == target_prefix_len:
            # We have a prefix of desired length
            # Can we extend it?
            if current_sum + 1 <= 256:
                # We can append current_sum + 1
                # Check if it's valid (it always is if > max_sum)
                # Max sum of prefix is current_sum.
                # New sums are current_sum + 1, current_sum + 1 + last, ...
                # Min new sum is current_sum + 1.
                # Max old sum is current_sum.
                # No collision.
                new_seq = vals + [current_sum + 1]
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
                    return True
            
            # Can we append 2 more?
            if 2 * current_sum + 1 <= 256:
                new_seq = vals + [current_sum + 1, 2 * current_sum + 1]
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
                    return True
            return False

        # Pruning
        # If we need 'k' more numbers, min sum is last+1 + ... + last+k
        k = target_prefix_len - len(vals)
        min_add = k * last_val + k * (k + 1) // 2
        if current_sum + min_add > max_sum:
            return False

        for x in range(last_val + 1, 257):
            # Check collision
            tails, total = [x], x
            for a in reversed(vals):
                total += a
                tails.append(total)
            
            if sums_set.intersection(tails):
                continue
            
            # Valid
            new_vals = vals + [x]
            new_sums = sums_set.union(tails)
            new_sum = current_sum + x
            
            # Recurse
            if search(new_vals, new_sums, x, new_sum, target_prefix_len, max_sum):
                return True

    # Try to find prefix of length 12 with sum <= 255
    # Start with empty
    # To speed up, we can start with the greedy prefix?
    # No, greedy might be suboptimal.
    # But we can try to search from scratch with strong pruning.
    
    # Let's try to find a prefix of length 11 with sum <= 127 first (easier)
    # If found, we get length 13.
    search([], set(), 0, 0, 11, 127)
    
    if best_len >= 13:
        return best_seq

    # If not, try prefix of length 12 with sum <= 255
    search([], set(), 0, 0, 12, 255)
    
    return best_seq
