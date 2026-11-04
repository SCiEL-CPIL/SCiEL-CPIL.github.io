import time
import itertools
import random

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses recursive construction followed by greedy deletion with count-based validity check.
    """
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            target_overlap = m - 2
            overlap = 0
            if len(new_word) >= target_overlap and len(block) >= target_overlap:
                if new_word[-target_overlap:] == block[:target_overlap]:
                    overlap = target_overlap
                else:
                    max_overlap = min(len(new_word), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new_word[len(new_word) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    current_word = word[:]
    
    # Pre-compute all permutations and map to integer codes for speed
    all_perms = list(itertools.permutations(range(1, n+1)))
    num_perms = len(all_perms)
    
    # Use base-8 encoding for windows (digits 1-7 fit in base 8)
    pow8 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow8[i] = pow8[i-1] * 8
    
    # Map window code to permutation index
    code_to_pidx = {}
    for idx, p in enumerate(all_perms):
        c = 0
        for i in range(n):
            c += p[i] * pow8[n - 1 - i]
        code_to_pidx[c] = idx
    
    def window_code(w, start):
        c = 0
        for i in range(n):
            c += w[start + i] * pow8[n - 1 - i]
        return c
    
    # Initialize counts
    counts = [0] * num_perms
    
    # Count initial windows
    L = len(current_word)
    num_windows = L - n + 1
    for i in range(num_windows):
        wc = window_code(current_word, i)
        pidx = code_to_pidx.get(wc, -1)
        if pidx >= 0:
            counts[pidx] += 1
    
    # Greedy deletion with multiple strategies
    # Strategy 1: Backward pass, delete as many as possible per scan
    for pass_num in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        L = len(current_word)
        num_windows = L - n + 1
        
        # Rebuild window codes for this pass
        wcodes = [0] * num_windows
        for i in range(num_windows):
            wcodes[i] = window_code(current_word, i)
        
        # Iterate backwards
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            
            start_j = max(0, i - n + 1)
            end_j = min(i, num_windows - 1)
            
            if start_j > end_j:
                continue
            
            # Check validity: no affected window is the last of its perm
            valid = True
            affected = []
            for j in range(start_j, end_j + 1):
                pidx = code_to_pidx.get(wcodes[j], -1)
                if pidx >= 0:
                    if counts[pidx] == 1:
                        valid = False
                        break
                    affected.append(pidx)
            
            if valid:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
                break  # Need to rebuild wcodes
        
        if not improved:
            break
    
    # Strategy 2: Forward pass
    for pass_num in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        L = len(current_word)
        num_windows = L - n + 1
        
        wcodes = [0] * num_windows
        for i in range(num_windows):
            wcodes[i] = window_code(current_word, i)
        
        for i in range(L):
            if time.time() > deadline:
                break
            
            start_j = max(0, i - n + 1)
            end_j = min(i, num_windows - 1)
            
            if start_j > end_j:
                continue
            
            valid = True
            affected = []
            for j in range(start_j, end_j + 1):
                pidx = code_to_pidx.get(wcodes[j], -1)
                if pidx >= 0:
                    if counts[pidx] == 1:
                        valid = False
                        break
                    affected.append(pidx)
            
            if valid:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
                break
        
        if not improved:
            break
    
    # Strategy 3: Random sampling for additional deletions
    for pass_num in range(10):
        if time.time() > deadline:
            break
        
        improved = False
        L = len(current_word)
        
        # Rebuild counts
        counts = [0] * num_perms
        num_windows = L - n + 1
        for i in range(num_windows):
            wc = window_code(current_word, i)
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                counts[pidx] += 1
        
        positions = list(range(L))
        random.shuffle(positions)
        
        for i in positions:
            if time.time() > deadline:
                break
            if i >= len(current_word):
                continue
            
            L = len(current_word)
            num_windows = L - n + 1
            
            start_j = max(0, i - n + 1)
            end_j = min(i, num_windows - 1)
            
            if start_j > end_j:
                continue
            
            valid = True
            affected = []
            for j in range(start_j, end_j + 1):
                wc = window_code(current_word, j)
                pidx = code_to_pidx.get(wc, -1)
                if pidx >= 0:
                    if counts[pidx] == 1:
                        valid = False
                        break
                    affected.append(pidx)
            
            if valid:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
        
        if not improved:
            break
    
    return "".join(map(str, current_word))
