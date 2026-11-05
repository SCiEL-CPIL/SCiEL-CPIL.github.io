import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Generate classic superpermutation
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
            # Max overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Map permutations to indices
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    
    def get_idx(window):
        return perm_to_idx.get(window, -1)
        
    # Count initial windows
    for i in range(len(word) - n + 1):
        idx = get_idx(tuple(word[i:i+n]))
        if idx != -1:
            counts[idx] += 1
            
    # Greedy deletion - optimized
    i = 0
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(word):
            if time.time() > deadline:
                break
                
            # Check if removing word[i] is valid
            valid = True
            affected_indices = []
            
            # Windows that will be removed (contain index i)
            for k in range(n):
                start = i - k
                if 0 <= start <= len(word) - n:
                    window = tuple(word[start:start+n])
                    idx = get_idx(window)
                    if idx != -1:
                        if counts[idx] == 1:
                            valid = False
                            break
                        affected_indices.append(idx)
            
            if not valid:
                i += 1
                continue
                
            # If valid, remove word[i]
            # We need to update counts
            # 1. Decrement removed windows
            for idx in affected_indices:
                counts[idx] -= 1
            
            # 2. Increment new windows
            # New windows are formed by the shift
            # The new string is word[0:i] + word[i+1:]
            # New windows start at i-k in the new string
            # Which corresponds to old indices:
            # New window at j (j < i) uses old word[j] ... word[j+n-1]
            # But if j+n-1 >= i, it uses word[j]...word[i-1] and word[i+1]...
            
            new_word = word[:i] + word[i+1:]
            
            # We only need to check windows that cross the boundary
            # i.e., windows in new_word that start at i-n+1 ... i-1
            for k in range(1, n):
                start = i - k
                if 0 <= start <= len(new_word) - n:
                    window = tuple(new_word[start:start+n])
                    idx = get_idx(window)
                    if idx != -1:
                        counts[idx] += 1
            
            # Commit the removal
            word = new_word
            improved = True
            # Do not increment i, because the next element is now at i
            # i stays the same
            
    return "".join(map(str, word))
