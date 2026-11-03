import itertools
import time
from collections import deque

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134124132413214321"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute prefix maps
    # D[k] maps prefix tuple of length k to list of perms starting with it
    D = [None] * (n + 1)
    for k in range(1, n + 1):
        D[k] = {}
        for p in perms:
            prefix = p[:k]
            if prefix not in D[k]:
                D[k][prefix] = []
            D[k][prefix].append(p)
    
    # Greedy construction
    # Start with the first permutation
    current = perms[0]
    word = list(current)
    unused = set(perms[1:])
    
    # Track last occurrence of each symbol for LRU heuristic
    # last_pos[sym] = index in word where sym was last seen
    last_pos = {}
    for i, s in enumerate(word):
        last_pos[s] = i
    
    start_time = time.time()
    timeout = 115.0 # Safety margin (150s limit, leave 35s)
    
    for _ in range(N - 1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Try to find max overlap k
        # k can be n-1 down to 1 (k=n implies same perm, which is used)
        for k in range(n - 1, 0, -1):
            suffix = current[n - k:]
            candidates = D[k].get(suffix, [])
            
            if not candidates:
                continue
            
            # Among unused candidates, pick the one that minimizes LRU penalty
            # i.e., the one whose appended symbols are the "least recently used"
            # Or rather, pick the one that appends symbols that were seen longest ago
            # Actually, we want to maximize the "staleness" of the appended symbols
            # to spread out symbol usage evenly
            
            best_p = None
            best_score = -1
            
            for p in candidates:
                if p not in unused:
                    continue
                
                # Score: sum of (current_pos - last_pos[sym]) for appended symbols
                # Higher score = symbols were seen longer ago = better to use now
                score = 0
                for sym in p[k:]:
                    score += (len(word) - last_pos.get(sym, -1))
                
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is not None:
                # Use this permutation
                unused.remove(best_p)
                # Append the non-overlapping part
                word.extend(best_p[k:])
                # Update last_pos for appended symbols
                for i, sym in enumerate(best_p[k:]):
                    last_pos[sym] = len(word) - len(best_p[k:]) + i
                current = best_p
                found = True
                break
        
        if not found:
            # Fallback: pick any unused perm
            p = next(iter(unused))
            unused.remove(p)
            word.extend(p)
            # Update last_pos
            for i, sym in enumerate(p):
                last_pos[sym] = len(word) - len(p) + i
            current = p

    return "".join(map(str, word))
