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
        Returns (is_valid, new_S, deltas) if valid, else (False, None, None).
        """
        L = len(S)
        if L <= n:
            return False, None, None
        
        # Old windows that contain index i
        start_min = max(0, i - n + 1)
        start_max = min(i, len(S) - n)
        old_windows_to_remove = []
        for j in range(start_min, start_max + 1):
            old_windows_to_remove.append(tuple(S[j:j+n]))
            
        # New windows that bridge the gap
        new_windows_to_add = []
        for j in range(max(0, i - n + 1), i):
            if j > len(S) - 1 - n:
                continue
            part1 = S[j:i]
            part2 = S[i+1:j+n+1]
            new_win = tuple(part1 + part2)
            new_windows_to_add.append(new_win)
            
        # Collect deltas
        deltas = {}
        for w in old_windows_to_remove:
            deltas[w] = deltas.get(w, 0) - 1
        for w in new_windows_to_add:
            deltas[w] = deltas.get(w, 0) + 1
            
        # Check validity
        for p, delta in deltas.items():
            if p in all_perms:
                new_count = perm_counts.get(p, 0) + delta
                if new_count < 1:
                    return False, None, None
                    
        return True, S[:i] + S[i+1:], deltas

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
            valid, new_S, deltas = is_valid_incremental(S, i)
            if valid:
                S = apply_removal(S, perm_counts, i, deltas)
                removed_any = True
                # Continue scanning from the new position (don't break)
                # The next index in the loop will be checked against the updated S
                # We need to be careful: after removal, indices shift
                # Let's just restart the scan for simplicity and correctness
                break
        if not removed_any:
            break
            
    return "".join(map(str, S))
