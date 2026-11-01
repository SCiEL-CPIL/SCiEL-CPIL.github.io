import math
import time
def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate recursive superpermutation
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
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            win = tuple(w[i:i+n])
            if len(set(win)) != n:
                return False
            seen.add(win)
        return len(seen) == math.factorial(n)

    if not is_valid(word):
        return "".join(map(str, word))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Identify candidates
        # Count occurrences of each window
        win_counts = {}
        for i in range(len(word) - n + 1):
            win = tuple(word[i:i+n])
            win_counts[win] = win_counts.get(win, 0) + 1
        
        candidates = []
        for i in range(len(word)):
            # Check if any window containing i has count > 1
            # Windows containing i: start j in [i-n+1, i]
            has_dup = False
            for j in range(max(0, i-n+1), min(i, len(word)-n) + 1):
                win = tuple(word[j:j+n])
                if win_counts[win] > 1:
                    has_dup = True
                    break
            if has_dup:
                candidates.append(i)
        
        # Try deleting candidates
        # Randomize order? Or reverse?
        # Reverse might be safer as it doesn't shift earlier indices?
        # But we recompute validity from scratch.
        import random
        random.shuffle(candidates)
        
        for i in candidates:
            if time.time() > deadline:
                break
            
            new_word = word[:i] + word[i+1:]
            if is_valid(new_word):
                word = new_word
                improved = True
                break
    
    return "".join(map(str, word))
