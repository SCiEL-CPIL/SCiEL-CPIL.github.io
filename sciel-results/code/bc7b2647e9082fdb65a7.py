"""Rewritten for shortest superpermutation on n=7 with aggressive construction + local search."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        word = _construct_classic(4)
        return _local_search_shorten(word, n, deadline)

    # Build a valid superpermutation via classic construction
    word = _construct_classic(n)

    # Aggressive local search to shorten
    word = _local_search_shorten(word, n, deadline)

    return word

def _construct_classic(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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
            max_overlap = min(len(new), len(block))
            for k in range(max_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new[len(new) - k:] == block[:k]:
                    overlap = k
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _local_search_shorten(word, n, deadline):
    """Shorten the word by greedily removing characters while keeping validity.
    
    Uses an efficient incremental refcount over the n! permutations.
    Multiple passes with different random orders to escape local minima.
    Adds multi-character removal and smarter ordering heuristics.
    """
    w = [int(c) for c in word]
    L = len(w)
    target = _factorial(n)

    # Full mask for n symbols (bits 1..n)
    full_mask = (1 << (n + 1)) - (1 << 1)

    def is_perm_window(start):
        if start < 0 or start + n > L:
            return False
        mask = 0
        for j in range(start, start + n):
            mask |= 1 << w[j]
        return mask == full_mask

    # Build initial refcount and window_perm
    num_windows = L - n + 1
    window_perm = [None] * num_windows
    refcount = {}
    for i in range(num_windows):
        if is_perm_window(i):
            t = tuple(w[i:i + n])
            window_perm[i] = t
            refcount[t] = refcount.get(t, 0) + 1

    covered = sum(1 for v in refcount.values() if v > 0)
    if covered < target:
        return "".join(map(str, w))

    def try_remove(pos):
        """Try removing position pos. Returns True if valid."""
        nonlocal w, L, window_perm, num_windows
        if pos <= 0 or pos >= L - 1:
            return False
        
        # Windows affected: those starting in [pos-n+1, pos]
        lo = max(0, pos - n + 1)
        hi = min(pos, L - n)
        
        removed = []
        ok = True
        for i in range(lo, hi + 1):
            if i < len(window_perm) and window_perm[i] is not None:
                t = window_perm[i]
                removed.append((i, t))
                refcount[t] -= 1
                if refcount[t] == 0:
                    ok = False
                    break
        
        if not ok:
            for (i, t) in removed:
                refcount[t] += 1
            return False

        # Actually remove
        w.pop(pos)
        L -= 1
        num_windows = L - n + 1

        # Rebuild window_perm for affected region
        new_lo = max(0, lo)
        new_hi = min(hi, L - n)
        for i in range(new_lo, new_hi + 1):
            if i < len(window_perm):
                if is_perm_window(i):
                    t = tuple(w[i:i + n])
                    window_perm[i] = t
                    refcount[t] = refcount.get(t, 0) + 1
                else:
                    window_perm[i] = None

        return True

    def try_remove_multi(positions):
        """Try removing multiple positions (sorted descending). Returns True if valid."""
        nonlocal w, L, window_perm, num_windows
        positions = sorted(positions, reverse=True)
        
        # Check all positions are valid
        for pos in positions:
            if pos <= 0 or pos >= L - 1:
                return False
        
        # Determine the range of affected windows
        min_pos = min(positions)
        max_pos = max(positions)
        lo = max(0, min_pos - n + 1)
        hi = min(max_pos, L - n)
        
        # Check if any window would be lost
        removed_perms = []
        ok = True
        for i in range(lo, hi + 1):
            if i < len(window_perm) and window_perm[i] is not None:
                t = window_perm[i]
                removed_perms.append((i, t))
                if refcount[t] == 1:
                    ok = False
                    break
        
        if not ok:
            return False
        
        # Remove positions in descending order
        for pos in positions:
            w.pop(pos)
            L -= 1
        
        num_windows = L - n + 1
        
        # Rebuild window_perm for affected region
        new_lo = max(0, lo)
        new_hi = min(hi, L - n)
        for i in range(new_lo, new_hi + 1):
            if i < len(window_perm):
                if is_perm_window(i):
                    t = tuple(w[i:i + n])
                    window_perm[i] = t
                    refcount[t] = refcount.get(t, 0) + 1
                else:
                    window_perm[i] = None

        return True

    # Compute "essentiality" score for each position: how many unique windows depend on it
    def compute_essentiality():
        essentiality = [0] * L
        for i in range(num_windows):
            if window_perm[i] is not None and refcount[window_perm[i]] == 1:
                for j in range(i, i + n):
                    essentiality[j] += 1
        return essentiality

    # Multi-pass hill climbing with different random seeds
    best_w = list(w)
    best_L = L
    
    seeds = [12345, 54321, 99999, 42424, 77777, 11111, 22222, 33333]
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(seed)
        w = list(best_w)
        L = best_L
        num_windows = L - n + 1
        
        # Rebuild data structures
        window_perm = [None] * num_windows
        refcount = {}
        for i in range(num_windows):
            if is_perm_window(i):
                t = tuple(w[i:i + n])
                window_perm[i] = t
                refcount[t] = refcount.get(t, 0) + 1
        
        improved_any = True
        passes = 0
        while improved_any and time.time() < deadline - 3.0:
            improved_any = False
            passes += 1
            
            # Compute essentiality to prioritize removals
            ess = compute_essentiality()
            
            # Create position list weighted by low essentiality
            positions = list(range(1, L - 1))
            # Sort by essentiality (lowest first) with some randomness
            rng.shuffle(positions)
            positions.sort(key=lambda p: (ess[p], rng.random()))
            
            # First pass: try single removals
            for pos in positions:
                if time.time() > deadline - 2.0:
                    break
                if pos <= 0 or pos >= L - 1:
                    continue
                if try_remove(pos):
                    improved_any = True
                    if L < best_L:
                        best_L = L
                        best_w = list(w)
                    break
            
            # Second pass: try multi-removals (2-3 adjacent or nearby positions)
            if not improved_any and time.time() < deadline - 2.5:
                ess = compute_essentiality()
                # Try removing pairs of low-essentiality positions
                low_ess_positions = [p for p in range(1, L - 1) if ess[p] <= 1]
                if len(low_ess_positions) >= 2:
                    rng.shuffle(low_ess_positions)
                    found = False
                    for i in range(min(50, len(low_ess_positions))):
                        if time.time() > deadline - 2.0:
                            break
                        pos1 = low_ess_positions[i]
                        # Try with nearby positions
                        for offset in [-2, -1, 1, 2]:
                            pos2 = pos1 + offset
                            if 1 <= pos2 < L - 1 and abs(pos1 - pos2) <= n:
                                if try_remove_multi([pos1, pos2]):
                                    improved_any = True
                                    found = True
                                    if L < best_L:
                                        best_L = L
                                        best_w = list(w)
                                    break
                        if found:
                            break
        
        if L < best_L:
            best_L = L
            best_w = list(w)
    
    # Final polish: one more pass with fresh ordering
    if time.time() < deadline - 1.0:
        w = list(best_w)
        L = best_L
        num_windows = L - n + 1
        window_perm = [None] * num_windows
        refcount = {}
        for i in range(num_windows):
            if is_perm_window(i):
                t = tuple(w[i:i + n])
                window_perm[i] = t
                refcount[t] = refcount.get(t, 0) + 1
        
        # Greedy removal from left to right
        for pos in range(1, L - 1):
            if time.time() > deadline - 0.5:
                break
            if try_remove(pos):
                if L < best_L:
                    best_L = L
                    best_w = list(w)
                # After removal, continue from same index (shifted)
                if pos >= L - 1:
                    break
    
    return "".join(map(str, best_w))

# EVOLVE-BLOCK-END
