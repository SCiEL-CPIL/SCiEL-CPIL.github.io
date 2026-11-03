import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate standard superpermutation
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
            # Find max overlap
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

    # 2. Optimization via Greedy Deletion of Non-Essential Characters
    # A character is "essential" if it belongs to at least one window that is the
    # unique occurrence of its permutation. Removing non-essential chars preserves validity.
    
    # Precompute total permutations
    total_perms = 1
    for i in range(1, n + 1):
        total_perms *= i

    def is_valid_superperm(w):
        seen = set()
        L = len(w)
        for i in range(L - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == total_perms

    while time.time() < deadline:
        L = len(word)
        if L < n:
            break

        # Single pass: collect indices for each permutation
        perm_indices = {}
        for i in range(L - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:  # valid permutation
                if window in perm_indices:
                    perm_indices[window].append(i)
                else:
                    perm_indices[window] = [i]

        # Mark essential indices: those in windows that appear exactly once
        essential = [False] * L
        has_essential = False
        for perm, indices in perm_indices.items():
            if len(indices) == 1:
                has_essential = True
                start = indices[0]
                for k in range(n):
                    essential[start + k] = True

        # Find removable (non-essential) positions
        if not has_essential:
            # All windows appear >= 2 times; any single removal is safe in principle,
            # but removing from the middle minimizes disruption to adjacency structure.
            # Remove the middle character.
            mid = L // 2
            word = word[:mid] + word[mid+1:]
            continue

        # Collect all non-essential indices
        removable = [i for i in range(L) if not essential[i]]
        
        if not removable:
            # Every position is essential; try a local swap to potentially free up positions
            # Try swapping adjacent pairs and see if we can reduce length
            improved = False
            for _ in range(100): # Try 100 random swaps
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, L - 2)
                j = i + 1
                # Temporarily swap
                word[i], word[j] = word[j], word[i]
                # Quick check: does the word still contain all perms?
                if is_valid_superperm(word):
                    # Valid swap, keep it and restart deletion loop
                    improved = True
                    break
                else:
                    # Restore swap
                    word[i], word[j] = word[j], word[i]
            
            if not improved:
                # If no swap worked, we are stuck
                break

        # Score each removable position by how "redundant" it is
        # A position is more redundant if its neighboring windows are more duplicated
        scored_removable = []
        for idx in removable:
            # Count how many times the windows containing this position appear
            redundancy_score = 0
            for offset in range(-n+1, 1):
                w_start = idx - offset
                if 0 <= w_start <= L - n:
                    w = tuple(word[w_start:w_start+n])
                    if len(set(w)) == n and w in perm_indices:
                        redundancy_score += len(perm_indices[w])
            scored_removable.append((redundancy_score, idx))
        
        # Sort by redundancy (most redundant first) and pick the best
        scored_removable.sort(reverse=True)
        # Pick from top candidates with some randomness to avoid local optima
        top_k = min(5, len(scored_removable))
        rem_idx = scored_removable[random.randint(0, top_k-1)][1]
        word = word[:rem_idx] + word[rem_idx+1:]

    return "".join(map(str, word))
