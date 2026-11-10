import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    num_perms = len(perms)
    
    # Precompute prefix maps
    prefix_map = [{} for _ in range(n)]
    for p in perms:
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_map[k]:
                prefix_map[k][prefix] = set()
            prefix_map[k][prefix].add(p)
    
    # Precompute all perms as a set for k=0 case?
    # We can just use a set of all perms for k=0.
    all_perms_set = set(perms)
    
    best_word = None
    best_len = float('inf')
    
    while time.time() < deadline:
        start_idx = random.randint(0, num_perms - 1)
        current_perm = perms[start_idx]
        used = {current_perm}
        word = list(current_perm)
        
        for _ in range(num_perms - 1):
            if time.time() > deadline:
                break
            found = False
            # Try to find max overlap
            for k in range(n - 1, -1, -1):
                suffix = current_perm[n - k:]
                if k == 0:
                    candidates = all_perms_set - used
                else:
                    candidates = prefix_map[k].get(suffix, set()) - used
                
                if candidates:
                    # Pick a random candidate
                    # For small sets, list() is fine
                    next_perm = random.choice(list(candidates))
                    used.add(next_perm)
                    word.extend(next_perm[k:])
                    current_perm = next_perm
                    found = True
                    break
            
            if not found:
                break
        
        if len(word) < best_len:
            best_len = len(word)
            best_word = word[:]
            
        # Optional: stop if we hit a very good score?
        # 5905 is best known. 5884 is lower bound.
        # If best_len <= 5905, we might want to keep searching for 5884?
        # But 5884 is a bound, not necessarily achievable.
        # We'll just run until deadline.

    if best_word is None:
        # Fallback
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    return "".join(map(str, best_word))
