import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Standard recursive construction
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 5:
        return "".join(map(str, word))

    # 2. Optimization Phase with efficient validity checking
    current_word = list(word)
    current_len = len(current_word)

    # Precompute permutation index map
    perm_map = {}
    idx = 0
    for p in permutations(range(1, n + 1)):
        perm_map[p] = idx
        idx += 1
    target_count = idx  # n!

    # Build window_idx array: window_idx[i] = perm_map of window starting at i, or -1
    def build_window_idx(w):
        L = len(w)
        widx = []
        for i in range(L - n + 1):
            window = tuple(w[i:i + n])
            widx.append(perm_map.get(window, -1))
        return widx

    # Build counts array and missing set
    def build_counts_and_missing(widx):
        counts = [0] * target_count
        missing = set(range(target_count))
        for v in widx:
            if v >= 0:
                if counts[v] == 0:
                    missing.discard(v)
                counts[v] += 1
        return counts, missing

    # Check if all permutations are present
    def all_present(missing):
        return len(missing) == 0

    # Initial setup
    widx = build_window_idx(current_word)
    counts, missing = build_counts_and_missing(widx)

    if not all_present(missing):
        return "".join(map(str, current_word))

    # Greedy deletion with efficient validity check
    # When deleting position pos, windows affected are those starting in [pos-n+1, pos]
    # We need to:
    # 1. Remove old windows containing pos
    # 2. Add new windows formed by the gap
    # 3. Check if missing set is empty

    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        pass_count += 1
        L = len(current_word)
        
        # Create position order: prioritize middle positions (more redundant)
        positions = list(range(L))
        random.shuffle(positions)
        # Sort by distance from center for priority, with random tie-breaking
        center = L / 2.0
        positions.sort(key=lambda p: (abs(p - center), random.random()))

        for pos in positions:
            if time.time() > deadline - 5:
                break
            
            L = len(current_word)
            if L <= n:
                break
            
            # Windows affected by removing position pos:
            # Old windows: start in [max(0, pos-n+1), min(pos, L-n)]
            old_start_min = max(0, pos - n + 1)
            old_start_max = min(pos, L - n)
            
            # Decrement counts for old windows, track what was removed
            removed_info = []  # list of (window_idx_value, was_count_zero_before)
            for s in range(old_start_min, old_start_max + 1):
                v = widx[s]
                if v >= 0:
                    if counts[v] == 1:
                        removed_info.append((v, True))  # will become 0
                    else:
                        removed_info.append((v, False))
                    counts[v] -= 1
            
            # Check if any permutation becomes missing
            becomes_missing = [v for v, was_zero in removed_info if was_zero]
            
            if len(becomes_missing) == 0:
                # Safe to delete - no permutation becomes missing
                # Perform deletion
                current_word = current_word[:pos] + current_word[pos + 1:]
                L = len(current_word)
                
                # Update widx: windows starting at [pos-n+1, pos] need to be recomputed
                # Windows before pos-n+1 are unchanged
                # Windows after pos are shifted left by 1
                
                # Rebuild affected windows
                new_widx = widx[:]  # Copy (this is O(L) but we do it only on success)
                
                # The windows that were removed are gone
                # New windows form in the gap
                # Windows starting at s in [pos-n+1, pos] (new indices)
                # In the new word, window starting at s covers new_word[s:s+n]
                # s ranges from max(0, pos-n+1) to min(pos-1, L-n) in new word
                
                # Actually, let's just rebuild the affected range
                # Old widx had length L-n+1
                # New widx has length L-n (one less)
                
                # Simpler: rebuild all windows from pos-n+1 to end
                # But that's O((L-n)·n) which is still expensive
                
                # Let's do incremental update:
                # The new word is shorter by 1
                # Windows starting at 0 to pos-n are unchanged (they don't include pos)
                # Windows starting at pos-n+1 to L-n-1 need recomputation
                # Windows starting at L-n+1 to L-1 no longer exist
                
                # Recompute windows in the affected range
                for s in range(max(0, pos - n + 1), L - n + 1):
                    if s < len(new_widx):
                        window = tuple(current_word[s:s + n])
                        new_widx[s] = perm_map.get(window, -1)
                    else:
                        window = tuple(current_word[s:s + n])
                        new_widx.append(perm_map.get(window, -1))
                
                # Truncate if needed
                expected_len = L - n + 1
                if len(new_widx) > expected_len:
                    new_widx = new_widx[:expected_len]
                
                widx = new_widx
                
                # Update missing set
                for v in becomes_missing:
                    missing.add(v)
                # Check if any newly formed windows fill gaps
                for s in range(max(0, pos - n + 1), len(widx)):
                    v = widx[s]
                    if v >= 0 and v in missing:
                        missing.discard(v)
                
                improved = True
                break  # Restart pass
            
            else:
                # Restore counts
                for v, was_zero in removed_info:
                    counts[v] += 1
    
    return "".join(map(str, current_word))
