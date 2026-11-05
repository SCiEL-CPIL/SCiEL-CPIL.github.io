import time
import sys
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

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
            # Find max overlap
            overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    L = len(word)
    
    # 2. Initial counts using integer encoding for speed
    # Map permutations to IDs? Integers are fine for n=7 (7^7 < 2^24)
    # Base 7 encoding: w[0]*7^(n-1) + ... + w[n-1]
    
    def get_window_id(start):
        val = 0
        for k in range(n):
            val = val * 7 + word[start+k]
        return val

    window_counts = Counter()
    for i in range(L - n + 1):
        window_counts[get_window_id(i)] += 1
    
    total_perms = 1
    for i in range(1, n+1):
        total_perms *= i
        
    missing = total_perms - len(window_counts)
    
    if missing > 0:
        pass

    improved = True
    # To speed up, we can try to delete from the middle out, or just scan.
    # Scanning is fine if fast enough.
    while improved and time.time() < deadline:
        improved = False
        for i in range(L):
            if time.time() > deadline:
                break
            
            # Simulate deletion of word[i]
            
            # 1. Remove windows in current word containing i
            removed_ids = []
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            for j in range(start_min, start_max + 1):
                wid = get_window_id(j)
                removed_ids.append(wid)
                window_counts[wid] -= 1
                if window_counts[wid] == 0:
                    del window_counts[wid]
                    missing += 1
            
            # 2. Add new windows in new word crossing the join
            new_ids = []
            L_prime = L - 1
            if L_prime >= n:
                new_start_min = max(0, i - n + 1)
                new_start_max = min(i - 1, L_prime - n)
                for k in range(new_start_min, new_start_max + 1):
                    # Construct window ID from new word
                    # New word is word[0:i] + word[i+1:L]
                    # Window k in new word:
                    # If k+n <= i: all in left part -> word[k:k+n]
                    # If k >= i: all in right part -> word[k+1:k+1+n]
                    # Else: crosses -> word[k:i] + word[i+1:i+1+(n-(i-k))]
                    
                    val = 0
                    # Left part
                    left_len = min(n, i - k)
                    for idx in range(k, k + left_len):
                        val = val * 7 + word[idx]
                    
                    # Right part
                    if left_len < n:
                        right_len = n - left_len
                        # Right part starts at i+1 in old word
                        # In new word, it starts at i
                        # So indices in old word are i+1, i+2, ...
                        for idx in range(i + 1, i + 1 + right_len):
                            val = val * 7 + word[idx]
                    
                    new_ids.append(val)
                    if val in window_counts:
                        window_counts[val] += 1
                    else:
                        window_counts[val] = 1
                        missing -= 1

            # 3. Check validity
            if missing == 0:
                # Accept deletion
                del word[i]
                L -= 1
                improved = True
                break # Restart scan as indices changed
            else:
                # Revert
                for wid in removed_ids:
                    if wid in window_counts:
                        window_counts[wid] += 1
                    else:
                        window_counts[wid] = 1
                        missing -= 1
                for wid in new_ids:
                    window_counts[wid] -= 1
                    if window_counts[wid] == 0:
                        del window_counts[wid]
                        missing += 1

    return "".join(map(str, word))
