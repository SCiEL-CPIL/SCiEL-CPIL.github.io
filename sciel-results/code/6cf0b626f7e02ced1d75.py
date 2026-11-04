import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # Convert to list of ints for easier manipulation
    S = list(word)
    
    # 2. Initialize counts
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
    
    best_S = S[:]
    best_len = len(S)
    
    # 3. Local Search with multiple strategies
    def is_valid(S, n):
        """Check if S contains all permutations of 1..n"""
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        return len(seen) == 2520 if n == 7 else len(seen) == 1
    
    def try_remove_pass(S, counts, n, deadline):
        """Try to remove characters, return new S and counts if any removal was made"""
        removed = False
        i = len(S) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            
            # Calculate affected old windows
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            # Check if removal is valid
            valid = True
            old_perms_to_dec = []
            
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                i -= 1
                continue
            
            # Perform the removal
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # Add new straddling windows
            new_len = len(S) - 1
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, new_len - n)
            
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            removed = True
            # Don't decrement i, since the next position is now at i
            
        return S, counts, removed
    
    # Multiple passes of removal
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        S, counts, removed = try_remove_pass(S, counts, n, deadline)
        
        if removed:
            best_S = S[:]
            best_len = len(S)
        else:
            # Stuck - try a perturbation
            if len(S) > n:
                # Try swapping two random characters
                if random.random() < 0.3:
                    i, j = random.sample(range(len(S)), 2)
                    S[i], S[j] = S[j], S[i]
                    # Rebuild counts
                    counts = {}
                    for k in range(len(S) - n + 1):
                        p = tuple(S[k:k+n])
                        counts[p] = counts.get(p, 0) + 1
                else:
                    # Try inserting a random character
                    if len(S) < best_len * 1.1:
                        pos = random.randint(0, len(S))
                        S.insert(pos, random.randint(1, n))
                        # Rebuild counts
                        counts = {}
                        for k in range(len(S) - n + 1):
                            p = tuple(S[k:k+n])
                            counts[p] = counts.get(p, 0) + 1
            else:
                break
    
    # Final validity check
    if not is_valid(best_S, n):
        # Fall back to the original construction if invalid
        return "".join(map(str, word))
    
    return "".join(map(str, best_S))
