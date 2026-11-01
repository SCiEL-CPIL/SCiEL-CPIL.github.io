"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter

def _build_classic_superperm(n):
    """Build the classic recursive superpermutation of 1..n."""
    word = [1]
    for m in range(2, n + 1):
        # Find all (m-1)-element windows in word with no repeats
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
            # Max overlap
            max_ov = min(len(new), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    ov = o
                    break
            new += block[ov:]
        word = new
    return word

def _validate(word, n):
    """Check that all n! permutations of 1..n appear as windows."""
    needed = set(itertools.permutations(range(1, n + 1)))
    found = set()
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i + n])
        found.add(w)
    return found == needed

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s limit
    
    # Classic construction
    word = _build_classic_superperm(n)
    
    if n <= 4:
        return "".join(map(str, word))
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if w in all_perms:
            perm_counts[w] += 1
    
    # Greedy deletion with smarter strategy:
    # Multiple passes, in different orders, to escape local optima
    best_word = list(word)
    best_len = len(word)
    
    for pass_num in range(5):
        if time.time() - start_time > deadline - 3:
            break
        
        # Different scan order each pass
        positions = list(range(len(word)))
        if pass_num % 2 == 1:
            positions.reverse()
        elif pass_num == 2:
            # Random-ish order (deterministic)
            positions.sort(key=lambda x: (x * 7919) % len(word))
        
        changed = True
        while changed and (time.time() - start_time) < deadline - 3:
            changed = False
            for i in positions:
                if i >= len(word):
                    continue
                if time.time() - start_time > deadline - 3:
                    break
                
                s_min = max(0, i - n + 1)
                s_max = min(i, len(word) - n)
                
                safe = True
                for s in range(s_min, s_max + 1):
                    w = tuple(word[s:s + n])
                    if w in all_perms and perm_counts[w] == 1:
                        safe = False
                        break
                
                if safe:
                    word.pop(i)
                    changed = True
                    # Update perm_counts incrementally for affected region
                    # Only need to recompute windows in [max(0,i-n+1), min(i+1, len(word)-n+1)]
                    re_lo = max(0, i - n + 1)
                    re_hi = min(len(word) - n + 1, i + 2)
                    for s in range(re_lo, re_hi):
                        old_w = None
                        if s < i:
                            old_w = tuple(word[s:s + n])
                            if old_w in all_perms:
                                perm_counts[old_w] -= 1
                            if perm_counts[old_w] == 0:
                                del perm_counts[old_w]
                        else:
                            old_w = None
                    # After deletion, recompute all windows in the region
                    for s in range(max(0, i - n + 1), min(len(word) - n + 1, i + 2)):
                        new_w = tuple(word[s:s + n])
                        if new_w in all_perms:
                            perm_counts[new_w] += 1
                    
                    if len(word) < best_len:
                        best_word = list(word)
                        best_len = len(word)
                    
                    # Rebuild positions if length changed significantly
                    if len(word) != len(positions):
                        positions = list(range(len(word)))
                        if pass_num % 2 == 1:
                            positions.reverse()
                        elif pass_num == 2:
                            positions.sort(key=lambda x: (x * 7919) % len(word))
        
        # Verify best_word is still valid
        if _validate(best_word, n):
            pass
        else:
            # Fallback to current word
            best_word = list(word)
    
    if len(best_word) < len(word):
        word = best_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
