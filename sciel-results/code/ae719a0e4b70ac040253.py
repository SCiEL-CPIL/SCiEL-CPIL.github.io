import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    def is_valid_extension(seq, used_sums, x):
        """
        Check if x can be appended to seq.
        used_sums is a set of all consecutive block sums of seq.
        Returns new_used_sums if valid, None otherwise.
        """
        if x in used_sums:
            return None
        
        new_sums = set()
        total = x
        new_sums.add(total)
        
        # Check sums ending at x
        for i in range(len(seq) - 1, -1, -1):
            total += seq[i]
            if total in used_sums or total in new_sums:
                return None
            new_sums.add(total)
            
        # Check if any new sums collide with each other (already checked in loop via new_sums set)
        # But we also need to ensure new_sums doesn't collide with used_sums (checked in loop)
        
        return used_sums.union(new_sums)

    def backtrack(seq, used_sums, last_val, target_len):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
            
        if len(seq) == target_len:
            return
            
        # Pruning: if we can't reach target_len with remaining numbers
        # Max possible length is 256 - last_val + 1 (if we use all consecutive)
        # But we need distinct sums, so it's harder.
        # Simple prune: if len(seq) + (256 - last_val) < target_len: return
        
        # Try adding next elements
        # Start from last_val + 1 to 256
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            new_used = is_valid_extension(seq, used_sums, x)
            if new_used is not None:
                seq.append(x)
                backtrack(seq, new_used, x, target_len)
                seq.pop()
                
                # Optional: break after first success if we just want ANY solution of target_len?
                # No, we want to maximize length, so we should explore more if possible.
                # But if we found target_len, we can stop this branch.
                # Actually, if we find a solution of target_len, we record it and can stop searching for longer ones in this branch?
                # The function is called with a specific target_len.
                
    # Initial best: powers of 2
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_len = len(best_seq)
    
    # Try to find a sequence of length 10
    # We can try different prefixes
    prefixes = [
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 2, 4, 8, 16, 32, 64, 100], # Random variation
        [1, 2, 4, 8, 16, 32, 50, 100],
        [1, 3, 5, 9, 17, 33, 65, 129],
    ]
    
    for prefix in prefixes:
        if time.time() > deadline:
            break
        
        # Validate prefix
        used = set()
        valid = True
        for i in range(len(prefix)):
            total = 0
            for j in range(i, -1, -1):
                total += prefix[j]
                if total in used:
                    valid = False
                    break
                used.add(total)
            if not valid:
                break
        
        if not valid:
            continue
            
        seq = prefix[:]
        last_val = prefix[-1]
        
        # Try to extend to length 10
        backtrack(seq, used, last_val, 10)
        
    # If we haven't found 10, try random restarts for length 10
    if best_len < 10:
        for seed in range(50):
            if time.time() > deadline:
                break
            random.seed(seed)
            # Generate a random increasing sequence of length 8
            prefix = sorted(random.sample(range(1, 257), 8))
            
            # Validate prefix
            used = set()
            valid = True
            for i in range(len(prefix)):
                total = 0
                for j in range(i, -1, -1):
                    total += prefix[j]
                    if total in used:
                        valid = False
                        break
                    used.add(total)
                if not valid:
                    break
            
            if not valid:
                continue
                
            seq = prefix[:]
            last_val = prefix[-1]
            backtrack(seq, used, last_val, 10)
            
    return best_seq
