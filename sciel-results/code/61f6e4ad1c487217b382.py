import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # 1. Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all permutations of length m-1 from word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # 2. Greedy Deletion
    # Precompute all valid permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Function to get all windows of length n from a list
    def get_windows(w):
        return [tuple(w[i:i+n]) for i in range(len(w) - n + 1)]
        
    # Initialize counts
    perm_counts = {}
    for p in get_windows(word):
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    # Helper to check if a window is a permutation
    def is_perm(w):
        return w in all_perms
        
    # Greedy deletion loop
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try deleting from left to right
        i = 0
        while i < len(word):
            # Check if we can delete word[i]
            # Windows affected are those containing index i
            # Indices: max(0, i-n+1) to min(i, len(word)-n)
            start_idx = max(0, i - n + 1)
            end_idx = min(i, len(word) - n)
            
            removable = True
            old_perms = []
            
            for k in range(start_idx, end_idx + 1):
                p = tuple(word[k:k+n])
                if is_perm(p):
                    if perm_counts.get(p, 0) == 1:
                        removable = False
                        break
                    old_perms.append(p)
                    
            if removable:
                # Perform deletion
                # 1. Decrement old counts
                for p in old_perms:
                    perm_counts[p] -= 1
                    if perm_counts[p] == 0:
                        del perm_counts[p]
                        
                # 2. Remove character
                word.pop(i)
                
                # 3. Calculate new windows formed by the merge
                # The merge happens around index i (which is now the position of the char after the deleted one)
                # The new windows are those that span the gap.
                # The gap is at i-1 and i.
                # Windows starting from max(0, i-n) to min(i-1, len(word)-n)
                new_start = max(0, i - n)
                new_end = min(i - 1, len(word) - n)
                
                for k in range(new_start, new_end + 1):
                    p = tuple(word[k:k+n])
                    if is_perm(p):
                        perm_counts[p] = perm_counts.get(p, 0) + 1
                        
                improved = True
                # Do not increment i, as the next character is now at index i
            else:
                i += 1
                
    return "".join(map(str, word))
