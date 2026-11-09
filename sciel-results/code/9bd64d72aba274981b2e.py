import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin: 15s buffer for GC, I/O, etc.
    
    # 1. Generate Classic Superpermutation
    def classic_superperm(n):
        if n == 1:
            return [1]
        prev = classic_superperm(n - 1)
        # Extract (n-1)-permutations from prev
        perms = []
        seen = set()
        for i in range(len(prev) - (n - 2)):
            window = tuple(prev[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [n] + list(p)
            # Overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new

    word = classic_superperm(n)
    
    if n <= 2:
        return "".join(map(str, word))

    # 2. Prepare for Greedy Pruning
    # We need to map permutations to IDs to maintain counts efficiently.
    # For n=7, 5040 perms.
    # We can use the tuple itself as the key in a dict.
    
    # Initialize counts
    counts = {}
    total_perms = 1
    for i in range(1, n + 1):
        total_perms *= i
        
    zero_count = total_perms
    
    # Pre-compute initial counts
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1
            
    # Pruning Loop
    # We scan left to right. If we remove a character, we stay at the same index (now holding the next char).
    i = 0
    while i < len(word) and time.time() < deadline:
        L = len(word)
        if L < n:
            break
            
        # Determine range of old windows affected by removing word[i]
        # Old windows starting at s where s <= i <= s+n-1
        # s in [i-n+1, i]
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        # Check if removal is safe
        # Safe if no essential (count==1) permutation is in the affected range
        safe = True
        for s in range(start_min, start_max + 1):
            w_old = tuple(word[s:s+n])
            if counts.get(w_old, 0) == 1:
                safe = False
                break
        
        if not safe:
            i += 1
            continue
        
        # Removal is safe. Update counts.
        # 1. Decrement old windows
        for s in range(start_min, start_max + 1):
            w = tuple(word[s:s+n])
            counts[w] -= 1
            if counts[w] == 0:
                del counts[w]
        
        # 2. Increment new windows
        # New windows are formed at starts s in [start_min, i] in the NEW word?
        # Let's trace.
        # Old word W. Remove W[i]. New word W'.
        # Windows in W' that are different from W are those that cover the gap.
        # In W', the gap is at i.
        # A window in W' starting at s covers the gap if s <= i-1 and s+n-1 >= i?
        # Actually, the window W'[s:s+n] is different from W[s:s+n] if the range [s, s+n-1] in W'
        # includes index i (which was removed) or is shifted.
        # If s < i, W'[s:s+n] = W[s:i] + W[i+1:s+n+1].
        # This is different from W[s:s+n] = W[s:i] + W[i:s+n].
        # So all s in [max(0, i-n+1), i-1] create new windows?
        # Also, what about s=i? W'[i:i+n] = W[i+1:i+n+1].
        # In W, the window at i was W[i:i+n].
        # So W'[i:i+n] is also a "new" window relative to the old set at that position?
        # Yes.
        
        # So new starts are s in [start_min, i].
        # But we must ensure s+n <= L-1 (new length).
        # Max s is L-1-n.
        new_start_min = start_min
        new_start_max = min(i, L - 1 - n)
        
        for s in range(new_start_min, new_start_max + 1):
            # Construct new window W'[s:s+n]
            # W'[s:s+n] corresponds to W[s:i] + W[i+1:s+n+1]
            # Note: if s >= i, then W[s:i] is empty? No, if s==i, W[i:i] is empty.
            # If s < i, W[s:i] is W[s]...W[i-1].
            # The second part is W[i+1]...W[s+n].
            # Wait, if s=i, W[i:i] is empty. W[i+1:i+n+1] is W[i+1]...W[i+n].
            # This is correct.
            
            if s < i:
                w_new = tuple(word[s:i] + word[i+1:s+n+1])
            else: # s == i
                w_new = tuple(word[i+1:i+n+1])
                
            if w_new in counts:
                counts[w_new] += 1
            else:
                counts[w_new] = 1
        
        # Perform removal
        word.pop(i)
        # Do not increment i, as the next element has shifted into position i
            
    return "".join(map(str, word))

if __name__ == "__main__":
    # For testing
    res = construct(7)
    print(len(res))
