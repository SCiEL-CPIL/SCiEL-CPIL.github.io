import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    prescribed = [1, 2, 5, 11]
    max_val = 284
    
    used_sums = 0
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            used_sums |= (1 << (prescribed[i] + prescribed[j]))
            
    candidates = []
    for x in range(12, max_val + 1):
        valid = True
        # Check against prescribed sums
        # 2x
        if (used_sums >> (2 * x)) & 1:
            valid = False
        else:
            for p in prescribed:
                if (used_sums >> (x + p)) & 1:
                    valid = False
                    break
        if valid:
            candidates.append(x)
            
    best_size = len(prescribed)
    best_set = prescribed[:]
    
    # Greedy to get a better initial bound
    g_set = prescribed[:]
    g_sums = used_sums
    for x in candidates:
        ok = True
        if (g_sums >> (2 * x)) & 1:
            ok = False
        else:
            for y in g_set:
                if (g_sums >> (x + y)) & 1:
                    ok = False
                    break
        if ok:
            g_set.append(x)
            g_sums |= (1 << (2 * x))
            for y in g_set[:-1]: # y in g_set before x added
                g_sums |= (1 << (x + y))
            # Note: g_set[:-1] is g_set without x
            # Wait, g_set.append(x) happens. So g_set[:-1] is correct.
            
    if len(g_set) > best_size:
        best_size = len(g_set)
        best_set = g_set[:]

    # B&B
    # We pass current_set as a list.
    # To speed up, we can maintain the list and just append/pop.
    
    def backtrack(i, current_set, used_sums, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if current_size + (len(candidates) - i) <= best_size:
            return
            
        if i == len(candidates):
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return

        x = candidates[i]
        
        # Try Add
        # Check validity
        valid = True
        if (used_sums >> (2 * x)) & 1:
            valid = False
        else:
            for y in current_set:
                if (used_sums >> (x + y)) & 1:
                    valid = False
                    break
                    
        if valid:
            new_sums = used_sums | (1 << (2 * x))
            for y in current_set:
                new_sums |= (1 << (x + y))
                
            current_set.append(x)
            backtrack(i + 1, current_set, new_sums, current_size + 1)
            current_set.pop()
            
        # Try Skip
        backtrack(i + 1, current_set, used_sums, current_size)

    backtrack(0, prescribed[:], used_sums, len(prescribed))
    
    return best_set
