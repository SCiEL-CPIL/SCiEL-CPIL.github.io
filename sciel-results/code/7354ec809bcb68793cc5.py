import time
import itertools
import random

def _build_classic_superperm(n):
    word = [1]
    for m in range(2, n + 1):
        seen = set()
        order = []
        for i in range(len(word) - m + 2):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                order.append(w)
        new = []
        for p in order:
            block = list(p) + [m] + list(p)
            max_ov = min(len(new), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    ov = o
                    break
            new += block[ov:]
        word = new
    return word

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 145
    
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    
    initial_word = _build_classic_superperm(n)
    best_len = len(initial_word)
    best_word = list(initial_word)
    
    # Run multiple passes
    for seed in range(10):
        if time.time() - start_time > deadline - 10:
            break
            
        random.seed(seed)
        word = list(initial_word)
        L = len(word)
        
        counts = [0] * num_perms
        for i in range(L - n + 1):
            w = tuple(word[i:i+n])
            wid = perm_to_id[w]
            counts[wid] += 1
            
        pos_min_count = [0] * L
        for i in range(L):
            min_c = float('inf')
            s_min = max(0, i - n + 1)
            s_max = min(i, L - n)
            for s in range(s_min, s_max + 1):
                w = tuple(word[s:s+n])
                wid = perm_to_id[w]
                c = counts[wid]
                if c < min_c:
                    min_c = c
            pos_min_count[i] = min_c
            
        safe_set = set(i for i in range(L) if pos_min_count[i] > 1)
        
        current_len = L
        
        while safe_set and (time.time() - start_time) < deadline - 5:
            i = random.choice(list(safe_set))
            
            # Decrement counts for old windows covering i
            s_min_old = max(0, i - n + 1)
            s_max_old = min(i, L - n)
            for s in range(s_min_old, s_max_old + 1):
                w = tuple(word[s:s+n])
                wid = perm_to_id[w]
                counts[wid] -= 1
                
            # Pop i
            word.pop(i)
            L -= 1
            current_len = L
            
            # Increment counts for new windows spanning gap
            s_min_new = max(0, i - n + 1)
            s_max_new = min(i - 1, L - n)
            for s in range(s_min_new, s_max_new + 1):
                w = tuple(word[s:s+n])
                wid = perm_to_id[w]
                counts[wid] += 1
                
            # Shift pos_min_count
            pos_min_count[i:] = pos_min_count[i+1:]
            pos_min_count.pop()
            
            # Rebuild safe_set
            safe_set = set(p for p in range(L) if pos_min_count[p] > 1)
            
            if current_len < best_len:
                best_len = current_len
                best_word = list(word)
                
    return "".join(map(str, best_word))
