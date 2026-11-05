import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    
    # 1. Generate standard superpermutation
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
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        word = new_word
    
    S = list(word)
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    perm_counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    def is_valid_incremental(S, i):
        """
        Check if removing S[i] keeps all permutations.
        Returns (is_valid, deltas) if valid, else (False, None).
        """
        L = len(S)
        if L <= n:
            return False, None
        
        # Old windows that contain index i
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        # New windows that bridge the gap
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, L - 1 - n)
        
        # Collect deltas and check validity in one pass
        deltas = {}
        valid = True
        
        # Process old windows (decrements)
        for j in range(start_min, start_max + 1):
            w = tuple(S[j:j+n])
            deltas[w] = deltas.get(w, 0) - 1
            if w in all_perms:
                if perm_counts.get(w, 0) + deltas[w] < 1:
                    valid = False
                    break
        
        if not valid:
            return False, None
        
        # Process new windows (increments)
        for j in range(new_start_min, new_start_max + 1):
            part1 = S[j:i]
            part2 = S[i+1:j+n+1]
            w = tuple(part1 + part2)
            deltas[w] = deltas.get(w, 0) + 1
            if w in all_perms:
                if perm_counts.get(w, 0) + deltas[w] < 1:
                    valid = False
                    break
        
        if not valid:
            return False, None
            
        return True, deltas

    def apply_removal(S, perm_counts, i, deltas):
        """Apply a valid removal."""
        for p, delta in deltas.items():
            new_count = perm_counts.get(p, 0) + delta
            if new_count <= 0:
                perm_counts.pop(p, None)
            else:
                perm_counts[p] = new_count
        return S[:i] + S[i+1:]

    # Pruning Loop with multiple passes and alternating scan directions
    for _pass in range(200):
        if time.time() > deadline - 5:
            break
        removed_any = False
        L = len(S)
        if L <= n:
            break
        
        # Alternate scan direction each pass
        if _pass % 2 == 0:
            indices = range(L - 1, -1, -1)
        else:
            indices = range(L)
            
        for i in indices:
            if time.time() > deadline - 5:
                break
            if len(S) <= n:
                break
            valid, deltas = is_valid_incremental(S, i)
            if valid:
                S = apply_removal(S, perm_counts, i, deltas)
                removed_any = True
                break
        if not removed_any:
            break
            
    return "".join(map(str, S))
