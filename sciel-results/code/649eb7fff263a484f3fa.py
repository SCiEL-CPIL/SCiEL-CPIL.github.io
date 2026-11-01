import time
import sys

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin

    # 1. Generate initial superpermutation (Recursive)
    # This yields length sum(k! for k in 1..n)
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
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Local Search Deletion
    # We use a list of integers.
    # We maintain a dictionary of counts for each permutation (encoded as int).
    
    def encode(window):
        val = 0
        for d in window:
            val = val * n + d
        return val

    # Initial encoding
    L = len(word)
    counts = {}
    window_codes = [0] * L  # window_codes[i] is the code of window starting at i in current word
    
    for i in range(L - n + 1):
        code = encode(word[i:i+n])
        window_codes[i] = code
        counts[code] = counts.get(code, 0) + 1

    # Check if we have all permutations (sanity check)
    # target_perms = factorial(n)
    # if len(counts) < target_perms: return ...
    
    # Deletion Loop
    # We will do multiple passes.
    # In each pass, we scan from left to right.
    # If we delete at i, we don't increment i (because the next element shifted left).
    
    while time.time() < deadline:
        deleted_any = False
        i = 0
        while i < L - n + 1: # We can only delete if there are windows to check? 
            # Actually we can delete any position, but if L < n, it's invalid.
            # If L is small, we stop.
            if L < n:
                break
            
            # Check if we can delete word[i]
            # Old windows: start at i-n+1 to i
            # New windows: start at i-n+1 to i-1 (in new string)
            
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            # Collect old codes
            old_codes = []
            can_delete = True
            
            for w in range(start_w, end_w + 1):
                code = window_codes[w]
                old_codes.append(code)
                if counts[code] == 1:
                    # This code is unique, so it must appear in the new set
                    # We'll check this later
                    pass
            
            # Collect new codes
            new_codes_set = set()
            for j in range(start_w, i):
                # New window starts at j in new string.
                # It consists of word[j:i] + word[i+1 : i+1 + (n - (i-j))]
                # Length of prefix: i - j
                # Length of suffix: n - (i - j)
                # Suffix starts at i+1 in old string, ends at i+1 + n - (i-j) - 1 = j + n
                # So suffix is word[i+1 : j+n+1]
                
                # Check bounds
                if j + n + 1 > L:
                    # This window would extend beyond the end of the string
                    # Wait, if j+n >= L, the window doesn't exist in the new string?
                    # The new string has length L-1.
                    # Valid windows in new string start at 0 to L-1-n.
                    # So j must be <= L-1-n.
                    # Since j < i and i <= L-1, this might be an issue.
                    # If i is near the end, some new windows might not exist.
                    # Specifically, if j+n >= L, the window is not fully formed in the new string.
                    # But wait, if we delete i, the max index is L-2.
                    # Window at j ends at j+n-1.
                    # We need j+n-1 <= L-2 => j <= L-n-1.
                    # So if j > L-n-1, this window doesn't exist.
                    continue
                    
                # Construct the window
                # word[j:i] + word[i+1:j+n+1]
                # We can just slice and concat, n is small.
                new_window = word[j:i] + word[i+1:j+n+1]
                code = encode(new_window)
                new_codes_set.add(code)
                
            # Check validity
            for code in old_codes:
                if counts[code] == 1 and code not in new_codes_set:
                    can_delete = False
                    break
            
            if can_delete:
                # Perform deletion
                # Update counts
                for code in old_codes:
                    counts[code] -= 1
                    if counts[code] == 0:
                        del counts[code]
                        
                for code in new_codes_set:
                    counts[code] = counts.get(code, 0) + 1
                    
                # Update word
                word.pop(i)
                L -= 1
                deleted_any = True
                
                # Update window_codes
                # This is tricky. The window_codes array needs to be updated.
                # Windows starting at < i-n+1 are unchanged.
                # Windows starting at >= i are shifted.
                # Windows in [i-n+1, i-1] are new.
                # Windows at i are gone.
                
                # Instead of updating the array incrementally which is error-prone,
                # we can just recompute the affected part or the whole array?
                # Recomputing whole array is O(L*n) = 42k.
                # Doing it after every deletion might be slow if we have many deletions.
                # But we only have ~10 deletions possible?
                # 5913 -> 5905 is 8 deletions.
                # So we only do this 8 times.
                # So recomputing is fine.
                
                # Recompute window_codes
                for k in range(L - n + 1):
                    window_codes[k] = encode(word[k:k+n])
                
                # Do not increment i, because the element at i is now the one that was at i+1
                # And we should check if we can delete that one too.
                # But we also need to check if the windows around i are still valid?
                # The check for i was done. Now i points to the next element.
                # The windows involving the new i might be different.
                # So we just continue the loop with the same i.
                
            else:
                i += 1
        
        if not deleted_any:
            break

    return "".join(map(str, word))
