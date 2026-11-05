import time
import itertools

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate standard superpermutation
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
    
    if n > 6:
        start_time = time.time()
        deadline = start_time + 110  # 10s buffer
        
        all_perms = set(itertools.permutations(range(1, n+1)))
        
        w = word[:]
        
        # Helper to check if w is a valid superpermutation
        def is_valid(w):
            present = set()
            for i in range(len(w) - n + 1):
                present.add(tuple(w[i:i+n]))
            return all_perms.issubset(present)
        
        # Greedy deletion
        i = 0
        # We might need multiple passes to remove all possible chars
        # But a single pass left-to-right is a good start.
        # If we remove a char, the indices shift.
        
        while i < len(w) - n + 1:
            if time.time() > deadline:
                break
            
            # Try removing w[i]
            w_new = w[:i] + w[i+1:]
            
            if is_valid(w_new):
                w = w_new
                # Do not increment i, check the new w[i]
            else:
                i += 1
                
        return "".join(map(str, w))

    return "".join(map(str, word))
