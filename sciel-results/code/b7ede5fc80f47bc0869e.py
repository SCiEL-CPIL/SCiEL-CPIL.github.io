import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 140
    
    if n <= 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        best_new = None
        best_len = float('inf')
        
        strategies = [
            lambda p: list(p) + [m] + list(p),
            lambda p: [m] + list(p) + list(p),
            lambda p: list(p) + list(p) + [m],
        ]
        
        for make_block in strategies:
            for perm_order in [perms, list(reversed(perms))]:
                new = []
                for p in perm_order:
                    block = make_block(p)
                    max_ov = min(len(new), len(block))
                    overlap = 0
                    for o in range(max_ov, -1, -1):
                        if o == 0 or new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                if len(new) < best_len:
                    best_len = len(new)
                    best_new = new
        
        word = best_new
    
    # Local pruning for small n
    if n <= 5 and time.time() < deadline - 5:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            perms_set = set(itertools.permutations(range(1, n+1)))
            for i in range(len(word) - n + 1):
                if time.time() > deadline - 2:
                    break
                test = word[:i] + word[i+1:]
                windows = set()
                for j in range(len(test) - n + 1):
                    windows.add(tuple(test[j:j+n]))
                if perms_set.issubset(windows):
                    word = test
                    changed = True
                    break
    
    return "".join(map(str, word))
