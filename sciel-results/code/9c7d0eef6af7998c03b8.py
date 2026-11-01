import time
import sys
from collections import defaultdict

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def classic_construct(n):
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Classic is optimal for 4 (33)
        return "".join(map(str, classic_construct(n)))
        
    word = classic_construct(n)
    target_count = factorial(n)
    
    # Initialize counts
    # counts: dict from tuple to int
    counts = defaultdict(int)
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
        
    # Verify initial state
    if len(counts) != target_count:
        # Should not happen for classic construct
        return "".join(map(str, word))

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        L = len(word)
        for i in range(L):
            if time.time() > deadline - 5.0:
                break
            
            # Identify windows in OLD word containing index i
            # Start j such that j <= i <= j+n-1
            # j >= 0, j <= L-n
            # j >= i-n+1
            min_j = max(0, i - n + 1)
            max_j = min(L - n, i)
            
            old_windows = []
            for j in range(min_j, max_j + 1):
                w = tuple(word[j:j+n])
                old_windows.append(w)
                
            # Check if any of these windows have count 1
            # If count is 1, removing this instance makes it 0 -> Invalid
            valid = True
            for w in old_windows:
                if counts[w] == 1:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If valid so far, check if we need to add new windows
            # New windows span the junction in the NEW word.
            # In NEW word, index i is removed.
            # Indices < i are same. Indices > i are shifted left by 1.
            # Junction is between new index i-1 (old i-1) and new index i (old i+1).
            # New windows start at k in [i-n+1, i-1] in NEW coordinates.
            # Let's map to OLD coordinates to get the content.
            # A new window starting at k (new) corresponds to:
            # If k < i:
            #   It takes elements from word[k : i] (old) and word[i+1 : i + (n - (i-k))]
            #   Wait, simpler:
            #   The new window content is formed by taking the prefix from old word ending at i-1
            #   and the suffix from old word starting at i+1.
            
            # Let's just construct the new window contents directly from the old word list.
            # For a new window starting at new index k:
            # It covers new indices k, k+1, ..., k+n-1.
            # Mapping to old indices:
            # If index < i: old_idx = new_idx
            # If index >= i: old_idx = new_idx + 1
            
            new_windows_to_add = []
            for k in range(max(0, i - n + 1), min(L - n, i - 1) + 1):
                # Note: k is the start index in the NEW word.
                # The window ends at k+n-1 in NEW word.
                # We need to ensure the window is valid in NEW word (length n).
                # k >= 0
                # k+n-1 <= L-2 (since new length is L-1)
                # So k <= L - n - 1?
                # Max start in new word is (L-1) - n + 1 = L - n.
                # So k ranges from max(0, i-n+1) to min(L-n, i-1).
                
                # Construct tuple
                # Part 1: old indices k to i-1
                # Part 2: old indices i+1 to i+1 + (k+n-1 - i)
                # Number of elements from part 1: i - k
                # Number of elements from part 2: n - (i - k) = n - i + k
                
                # Check bounds
                if i - k > n: continue # Should not happen due to range
                if n - i + k > n: continue
                
                # Slice
                p1 = word[k:i]
                p2_len = n - len(p1)
                p2 = word[i+1:i+1+p2_len]
                
                if len(p1) + len(p2) == n:
                    new_w = tuple(p1 + p2)
                    new_windows_to_add.append(new_w)
            
            # If we add these new windows, do we break any count?
            # We are DECREASING counts of old_windows.
            # We are INCREASING counts of new_windows.
            # We already checked that no old_window count drops to 0.
            # Adding to new_windows cannot make a count 0 (it increases it).
            # So the deletion is VALID.
            
            # Commit
            for w in old_windows:
                counts[w] -= 1
                if counts[w] == 0:
                    del counts[w] # Optional cleanup, or just leave 0
            for w in new_windows_to_add:
                counts[w] += 1
                
            # Remove from word
            word = word[:i] + word[i+1:]
            improved = True
            break # Restart loop

    return "".join(map(str, word))
