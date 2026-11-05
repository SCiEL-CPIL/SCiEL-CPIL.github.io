import time
import itertools
import random
import numpy as np

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    """
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin from 150s limit
    
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
    
    # Use base-8 encoding for windows
    pow8 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow8[i] = pow8[i-1] * 8
        
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

    L = len(current_word)
    num_windows = L - n + 1
    wcodes = [0] * num_windows
    for i in range(num_windows):
        wcodes[i] = window_code(current_word, i)
    
    counts = [0] * num_perms
    for wc in wcodes:
        pidx = code_to_pidx.get(wc, -1)
        if pidx >= 0:
            counts[pidx] += 1

    def rebuild():
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        num_windows = L - n + 1
        wcodes = [0] * num_windows
        for i in range(num_windows):
            wcodes[i] = window_code(current_word, i)
        counts = [0] * num_perms
        for wc in wcodes:
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                counts[pidx] += 1

    def try_delete(pos):
        """Try to delete position pos. Returns True if successful."""
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        num_windows = L - n + 1
        start_j = max(0, pos - n + 1)
        end_j = min(pos, num_windows - 1)
        
        valid = True
        for j in range(start_j, end_j + 1):
            pidx = code_to_pidx.get(wcodes[j], -1)
            if pidx >= 0 and counts[pidx] == 1:
                valid = False
                break
        
        if valid:
            current_word.pop(pos)
            rebuild()
            return True
        return False

    def try_swap(pos):
        """Swap pos and pos+1 if valid. Returns True if successful."""
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        if pos >= L - 1:
            return False
        
        start_j = max(0, pos - n + 1)
        end_j = min(pos + 1, L - n)
        
        # Check validity
        old_wcodes = {}
        for j in range(start_j, end_j + 1):
            old_wcodes[j] = wcodes[j]
            
        # Simulate swap
        current_word[pos], current_word[pos+1] = current_word[pos+1], current_word[pos]
        
        valid = True
        for j in range(start_j, end_j + 1):
            new_code = window_code(current_word, j)
            old_code = old_wcodes[j]
            
            old_pidx = code_to_pidx.get(old_code, -1)
            new_pidx = code_to_pidx.get(new_code, -1)
            
            if old_pidx >= 0 and new_pidx != old_pidx:
                if counts[old_pidx] == 1:
                    valid = False
                    break
            
        if not valid:
            current_word[pos], current_word[pos+1] = current_word[pos+1], current_word[pos]
            return False
            
        # Apply swap - update wcodes and counts
        for j in range(start_j, end_j + 1):
            old_code = old_wcodes[j]
            new_code = window_code(current_word, j)
            
            old_pidx = code_to_pidx.get(old_code, -1)
            new_pidx = code_to_pidx.get(new_code, -1)
            
            if old_pidx >= 0:
                counts[old_pidx] -= 1
            if new_pidx >= 0:
                counts[new_pidx] += 1
            wcodes[j] = new_code
            
        return True

    def batch_delete_backward():
        """Try to delete positions from back to front, multiple per pass."""
        nonlocal wcodes, counts, L, num_windows
        improved = False
        L = len(current_word)
        i = L - 1
        while i >= 0:
            if time.time() > deadline:
                break
            if try_delete(i):
                improved = True
                L = len(current_word)
            i -= 1
        return improved

    def batch_delete_forward():
        """Try to delete positions from front to back, multiple per pass."""
        nonlocal wcodes, counts, L, num_windows
        improved = False
        L = len(current_word)
        i = 0
        while i < L:
            if time.time() > deadline:
                break
            if try_delete(i):
                improved = True
                L = len(current_word)
            else:
                i += 1
        return improved

    def batch_delete_random():
        """Try to delete random positions, multiple per pass."""
        nonlocal wcodes, counts, L, num_windows
        improved = False
        L = len(current_word)
        positions = list(range(L))
        random.shuffle(positions)
        for i in positions:
            if time.time() > deadline:
                break
            # Position may have shifted due to previous deletions
            # Just try it; if out of range, skip
            if 0 <= i < len(current_word):
                if try_delete(i):
                    improved = True
        return improved

    def batch_swap_random(count=50):
        """Try random swaps, multiple per pass."""
        nonlocal wcodes, counts, L, num_windows
        improved = False
        for _ in range(count):
            if time.time() > deadline:
                break
            L = len(current_word)
            if L > 1:
                pos = random.randint(0, L - 2)
                if try_swap(pos):
                    improved = True
        return improved

    # Main Loop
    no_improve_count = 0
    while time.time() < deadline:
        improved = False
        
        # 1. Backward Greedy (multiple deletions per pass)
        if batch_delete_backward():
            no_improve_count = 0
            continue
        
        # 2. Forward Greedy (multiple deletions per pass)
        if batch_delete_forward():
            no_improve_count = 0
            continue
            
        # 3. Random Deletion (multiple per pass)
        if batch_delete_random():
            no_improve_count = 0
            continue
            
        # 4. Random Swaps (batch)
        if batch_swap_random(100):
            no_improve_count = 0
            continue
        
        # 5. Stuck - more aggressive random swaps
        if batch_swap_random(500):
            no_improve_count = 0
            continue
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break

    return "".join(map(str, current_word))
