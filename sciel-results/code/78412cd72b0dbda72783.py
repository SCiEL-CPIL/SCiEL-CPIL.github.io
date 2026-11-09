"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations as _perms
from collections import Counter, defaultdict

# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Optimal is 33. Classic is 33.
        return _classic_construct(n)
    if n == 5:
        # Optimal is 153. Classic is 153.
        return _classic_construct(n)
        
    deadline = time.time() + 115.0
    
    # Get all permutations
    all_perms = list(_perms(range(1, n + 1)))
    N = len(all_perms)
    
    # Precompute prefix maps for fast lookup
    # prefix_map[k] is a dict: prefix_tuple -> list of perms starting with that prefix
    prefix_map = [defaultdict(list) for _ in range(n)]
    for p in all_perms:
        for k in range(1, n):
            prefix_map[k][p[:k]].append(p)
    
    best_word = _classic_construct(n)
    best_len = len(best_word)
    
    # Greedy search with multiple random starts
    # Time budget for greedy: ~60s
    greedy_deadline = time.time() + 60.0
    
    while time.time() < greedy_deadline:
        start_idx = random.randint(0, N - 1)
        current = all_perms[start_idx]
        visited = set([current])
        word = list(current)
        
        for _ in range(N - 1):
            best_next = None
            best_overlap = -1
            
            # Check overlaps from largest to smallest
            # k is the overlap length
            # We need p[:k] == current[n-k:]
            for k in range(n - 1, 0, -1):
                suffix = current[n - k:]
                candidates = prefix_map[k].get(suffix, [])
                for p in candidates:
                    if p not in visited:
                        best_next = p
                        best_overlap = k
                        break
                if best_next is not None:
                    break
            
            if best_next is None:
                # Fallback: pick any unvisited (should not happen if graph is connected)
                # Find any unvisited
                for p in all_perms:
                    if p not in visited:
                        best_next = p
                        best_overlap = 0
                        break
                
            if best_next is None:
                break # Stuck
                
            visited.add(best_next)
            if best_overlap > 0:
                word.extend(best_next[n - best_overlap:])
            else:
                word.extend(best_next)
            current = best_next
            
        current_len = len(word)
        if current_len < best_len:
            best_len = current_len
            best_word = "".join(map(str, word))
            
    # Local search on the best word found
    if time.time() < deadline - 5.0:
        best_word = _local_search(best_word, n, deadline - 2.0)
        
    return best_word

def _local_search(word_str, n, deadline):
    word = list(map(int, word_str))
    L = len(word)
    
    # Precompute windows
    perm_count = Counter()
    for i in range(L - n + 1):
        perm_count[tuple(word[i:i+n])] += 1
        
    removed = [False] * L
    
    # Single pass of removal
    for pos in range(L - 1, -1, -1):
        if time.time() > deadline:
            break
        if removed[pos]:
            continue
            
        valid = True
        min_i = max(0, pos - n + 1)
        max_i = min(pos, L - n)
        
        for wi in range(min_i, max_i + 1):
            if removed[wi] or removed[wi + n - 1]:
                continue
            w = tuple(word[wi:wi+n])
            if perm_count[w] <= 1:
                valid = False
                break
                
        if valid:
            removed[pos] = True
            for wi in range(min_i, max_i + 1):
                if removed[wi] or removed[wi + n - 1]:
                    continue
                w = tuple(word[wi:wi+n])
                perm_count[w] -= 1
                
    word = [word[i] for i in range(L) if not removed[i]]
    return "".join(map(str, word))

def _classic_construct(n):
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
