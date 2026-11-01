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

    # 2. Optimization Phase with O(n) validity checking
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

    # Build counts array: counts[j] = number of windows with perm index j
    def build_counts(widx):
        counts = [0] * target_count
        for v in widx:
            if v >= 0:
                counts[v] += 1
        return counts

    # Check if all permutations are present
    def all_present(counts):
        for j in range(target_count):
            if counts[j] == 0:
                return False
        return True

    # Initial setup
    widx = build_window_idx(current_word)
    counts = build_counts(widx)

    if not all_present(counts):
        return "".join(map(str, current_word))

    # Greedy deletion with O(n) validity check
    # Strategy: iterate positions, try to delete, update counts in O(n)
    # A position pos affects windows starting at max(0, pos-n+1) to min(pos, L-n)
    # When we delete pos, windows that included pos are removed
    # New windows form in the gap: starting from pos-n+1 to pos (shifted)

    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        pass_count += 1
        L = len(current_word)
        
        # Create position order: prioritize middle positions (more redundant)
        positions = list(range(L))
        # Sort by distance from center - middle positions first
        center = L / 2.0
        positions.sort(key=lambda p: abs(p - center))
        
        # Shuffle within same distance to avoid bias
        random.shuffle(positions)
        # Re-sort by distance but with random tie-breaking
        # Actually, let's just shuffle and process in order, but bias toward middle
        # Simpler: random shuffle, but process from middle outward
        positions = list(range(L))
        random.shuffle(positions)
        # Sort by distance from center for priority
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
            # These windows contain pos and will be removed
            
            # Compute which old windows are removed
            old_start_min = max(0, pos - n + 1)
            old_start_max = min(pos, L - n)
            
            # Decrement counts for old windows
            removed_windows = []
            for s in range(old_start_min, old_start_max + 1):
                v = widx[s]
                if v >= 0:
                    counts[v] -= 1
                    removed_windows.append(v)
            
            # Check if all permutations still present
            valid = True
            for j in range(target_count):
                if counts[j] == 0:
                    valid = False
                    break
            
            if valid:
                # Perform deletion
                current_word = current_word[:pos] + current_word[pos + 1:]
                L = len(current_word)
                
                # Rebuild widx (expensive but only on successful deletion)
                widx = build_window_idx(current_word)
                # Rebuild counts from scratch (safe)
                counts = [0] * target_count
                for v in widx:
                    if v >= 0:
                        counts[v] += 1
                
                improved = True
                break  # Restart pass
            
            else:
                # Restore counts
                for v in removed_windows:
                    counts[v] += 1
    
    return "".join(map(str, current_word))
