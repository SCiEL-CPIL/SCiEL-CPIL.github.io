"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Classic recursive construction
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
            # Find maximum overlap
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    
    # Try to shorten the word
    if n >= 3 and time.time() < deadline:
        word = try_shorten_fast(word, n, deadline)
    
    return "".join(map(str, word))

def try_shorten_fast(word, n, deadline):
    """
    Optimized local search to remove redundant symbols.
    Uses incremental update of permutation counts.
    """
    word = list(word)
    L = len(word)
    
    # Initialize permutation counts
    perm_count = defaultdict(int)
    for k in range(L - n + 1):
        window = tuple(word[k:k + n])
        if len(set(window)) == n:
            perm_count[window] += 1
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate from right to left
        for i in range(L - 1, -1, -1):
            if time.time() >= deadline:
                return word
            
            # Identify affected windows
            # Window k is affected if i is in [k, k+n) => k <= i < k+n => i-n+1 <= k <= i
            start_k = max(0, i - n + 1)
            end_k = min(i, L - n)
            
            if start_k > end_k:
                continue

            # Check if deletion is safe
            safe = True
            # We need to check if any permutation count drops to 0
            # Collect old and new windows for affected indices
            old_windows = []
            new_windows = []
            
            for k in range(start_k, end_k + 1):
                p_old = tuple(word[k:k + n])
                # Construct new window: word[k:i] + word[i+1:k+n+1]
                # Note: indices in word are still original for this check
                p_new = tuple(word[k:k + i] + word[i + 1:k + n + 1])
                old_windows.append(p_old)
                new_windows.append(p_new)
                
            # Check counts
            # We only need to worry about permutations that are present
            # If a permutation P has count C, and we remove r instances and add a instances,
            # new count is C - r + a. We need C - r + a >= 1.
            
            # To be efficient, we can just iterate through the unique old windows
            # and check the balance.
            
            # Build a map of changes: P -> (removed, added)
            changes = defaultdict(lambda: [0, 0])
            for p_old, p_new in zip(old_windows, new_windows):
                if len(set(p_old)) == n:
                    changes[p_old][0] += 1
                if len(set(p_new)) == n:
                    changes[p_new][1] += 1
            
            for p, (rem, add) in changes.items():
                current_count = perm_count.get(p, 0)
                if current_count - rem + add < 1:
                    safe = False
                    break
            
            if safe:
                # Perform deletion
                # Update perm_count
                for p_old, p_new in zip(old_windows, new_windows):
                    if len(set(p_old)) == n:
                        perm_count[p_old] -= 1
                        if perm_count[p_old] == 0:
                            del perm_count[p_old]
                    if len(set(p_new)) == n:
                        perm_count[p_new] += 1
                
                word = word[:i] + word[i+1:]
                L -= 1
                improved = True
                # Continue loop, do not break, as we can try next index i-1
                # Note: indices > i have shifted, but we are iterating downwards,
                # and the logic for affected windows uses current L and i, so it remains valid.
    
    return word
# EVOLVE-BLOCK-END
