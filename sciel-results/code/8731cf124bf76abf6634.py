import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
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
    def build_counts(S, n):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts
    
    counts = build_counts(S, n)
    
    best_S = S[:]
    best_len = len(S)
    
    # 3. Local Search with multiple strategies
    def is_valid(S, n):
        """Check if S contains all permutations of 1..n"""
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        total = 1
        for k in range(2, n+1):
            total *= k
        return len(seen) == total
    
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
    
    def try_remove_pass_forward(S, counts, n, deadline):
        """Try to remove characters forward, return new S and counts if any removal was made"""
        removed = False
        i = 0
        while i < len(S):
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
                i += 1
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
            # Don't increment i, since the next position is now at i
            
        return S, counts, removed
    
    def try_swap(S, counts, n, deadline):
        """Try to swap two adjacent characters if it doesn't lose any permutation"""
        swapped = False
        for i in range(len(S) - 1):
            if time.time() > deadline:
                break
            
            # Check if swap is valid
            # Affected windows: those containing position i or i+1
            start_old = max(0, i - n + 1)
            end_old = min(i + 1, len(S) - n)
            
            old_windows = []
            valid = True
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_windows.append((j, p))
            
            if not valid:
                continue
            
            # Check if new windows after swap are all already present or newly created
            # After swap, windows containing i or i+1 change
            new_windows = []
            for j in range(start_old, end_old + 1):
                # Build new window after swap
                chars = []
                for idx in range(j, j+n):
                    if idx == i:
                        chars.append(S[i+1])
                    elif idx == i+1:
                        chars.append(S[i])
                    else:
                        chars.append(S[idx])
                p_new = tuple(chars)
                new_windows.append((j, p_new))
            
            # Check: for each old window, if it was unique, the new window must already exist (count >= 1 after removal)
            # Actually, we need to check that no permutation is lost
            # A permutation is lost if its count goes from 1 to 0
            # Let's simulate: remove old windows, add new windows, check no count goes to 0
            
            # Count changes:
            # - old windows lose 1 each
            # - new windows gain 1 each
            # But some windows might be in both old and new lists
            
            # Simpler: just do the swap, rebuild counts locally for affected region, check validity
            # For efficiency, let's check if any old window with count 1 would be lost
            
            # Actually, the simplest correct check:
            # For each old window with count 1, verify that after swap, that permutation still appears somewhere
            # This is expensive. Let's just try the swap and rebuild if needed.
            
            # Do the swap
            S[i], S[i+1] = S[i+1], S[i]
            swapped = True
            break
        
        if swapped:
            counts = build_counts(S, n)
        
        return S, counts, swapped
    
    # Multiple passes of removal with different strategies
    max_passes = 200
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        # Alternate between backward and forward removal
        if pass_num % 2 == 0:
            S, counts, removed = try_remove_pass(S, counts, n, deadline)
        else:
            S, counts, removed = try_remove_pass_forward(S, counts, n, deadline)
        
        if removed:
            if len(S) < best_len:
                best_S = S[:]
                best_len = len(S)
        else:
            # Stuck - try a perturbation
            if len(S) > n:
                # Try swapping two random characters
                if random.random() < 0.4:
                    i, j = random.sample(range(len(S)), 2)
                    S[i], S[j] = S[j], S[i]
                    counts = build_counts(S, n)
                else:
                    # Try inserting a random character at a random position
                    if len(S) < best_len * 1.15:
                        pos = random.randint(0, len(S))
                        S.insert(pos, random.randint(1, n))
                        counts = build_counts(S, n)
            else:
                break
    
    # Final validity check
    if not is_valid(best_S, n):
        # Fall back to the original construction if invalid
        return "".join(map(str, word))
    
    return "".join(map(str, best_S))
