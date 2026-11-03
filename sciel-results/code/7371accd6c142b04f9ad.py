import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # 1. Generate standard superpermutation using classic recursive construction
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

    if n < 3:
        return "".join(map(str, word))

    # 2. Optimization via greedy deletion of non-essential characters
    # A position is "essential" if it's part of a permutation window that appears exactly once.
    # We iteratively remove non-essential positions.
    
    L = len(word)
    # Count occurrences of each permutation
    perm_counts = {}
    perm_positions = {}  # perm -> list of positions where it occurs
    
    for i in range(L - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            if window in perm_counts:
                perm_counts[window] += 1
                perm_positions[window].append(i)
            else:
                perm_counts[window] = 1
                perm_positions[window] = [i]
    
    # Mark essential indices: positions that are part of any unique permutation
    essential = [False] * L
    for perm, count in perm_counts.items():
        if count == 1:
            pos = perm_positions[perm][0]
            for k in range(n):
                essential[pos + k] = True
    
    # Iteratively remove non-essential characters
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(word)
        
        # Rebuild essential array for current word
        perm_counts = {}
        perm_positions = {}
        for i in range(L - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                if window in perm_counts:
                    perm_counts[window] += 1
                    perm_positions[window].append(i)
                else:
                    perm_counts[window] = 1
                    perm_positions[window] = [i]
        
        essential = [False] * L
        for perm, count in perm_counts.items():
            if count == 1:
                pos = perm_positions[perm][0]
                for k in range(n):
                    essential[pos + k] = True
        
        # Find first non-essential position and remove it
        for i in range(L):
            if not essential[i]:
                word = word[:i] + word[i + 1:]
                improved = True
                break
        
        if not improved:
            # All positions are essential; try a different heuristic
            # Try removing from the end if the last n-1 chars don't form a unique perm start
            if len(word) > n:
                # Check if removing last char is safe
                L = len(word)
                # The last window is word[L-n:L]
                last_window = tuple(word[L-n:L])
                if len(set(last_window)) == n:
                    # Check if this window appears elsewhere
                    count = 0
                    for i in range(L - n + 1):
                        if tuple(word[i:i+n]) == last_window:
                            count += 1
                            if count > 1:
                                break
                    if count > 1:
                        word = word[:-1]
                        improved = True
    
    return "".join(map(str, word))
