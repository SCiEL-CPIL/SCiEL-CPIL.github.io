import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate Standard Superpermutation
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # 2. Local Search: Remove non-essential characters
    # Precompute factorials for ranking
    factorials = [1] * (n + 1)
    for i in range(1, n + 1):
        factorials[i] = factorials[i - 1] * i
        
    def rank_perm(p):
        """Return rank of permutation p (list of 1..n) in lexicographic order, or -1 if invalid."""
        if len(p) != n:
            return -1
        if len(set(p)) != n:
            return -1
        if sorted(p) != list(range(1, n + 1)):
            return -1
            
        # Factoradic ranking
        # p = [p0, p1, ..., pn-1]
        # rank = p0 * (n-1)! + p1 * (n-2)! + ...
        # where pi is the index of the element in the remaining sorted list
        remaining = list(range(1, n + 1))
        rank = 0
        for i in range(n):
            val = p[i]
            idx = remaining.index(val)
            rank += idx * factorials[n - 1 - i]
            remaining.pop(idx)
        return rank

    def get_window_perm(w, start):
        """Get rank of window starting at 'start' in list w."""
        p = w[start:start + n]
        return rank_perm(p)

    L = len(word)
    
    # Initialize counts and window perms
    # window_perm[i] is the rank of the permutation at window i, or -1
    window_perm = [-1] * (L - n + 1)
    perm_counts = [0] * (factorials[n])
    
    for i in range(L - n + 1):
        r = get_window_perm(word, i)
        if r != -1:
            window_perm[i] = r
            perm_counts[r] += 1
            
    def is_essential(i):
        """Check if character at index i is essential."""
        # A character is essential if it belongs to any window with count 1
        # Windows containing i are [i-n+1, i]
        min_j = max(0, i - n + 1)
        max_j = min(i, L - n)
        for j in range(min_j, max_j + 1):
            if window_perm[j] != -1 and perm_counts[window_perm[j]] == 1:
                return True
        return False

    # Main loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Scan from left to right
        for i in range(L - n + 1):
            if time.time() > deadline:
                break
            if not is_essential(i):
                # Remove word[i]
                # Update word
                word.pop(i)
                L -= 1
                
                # Update window_perm and perm_counts
                # This requires re-evaluating windows affected by the shift
                # For simplicity and speed (since few removals), recalculate all
                # But we can be smarter:
                # Windows j < i: unchanged
                # Windows j >= i: shifted. New window j is old window j+1
                # However, the content of windows in [i-n+1, i] changes/invalidates.
                
                # Let's just recalculate the arrays. It's O(L*n) which is fast.
                window_perm = [-1] * (L - n + 1)
                perm_counts = [0] * (factorials[n])
                for k in range(L - n + 1):
                    r = get_window_perm(word, k)
                    if r != -1:
                        window_perm[k] = r
                        perm_counts[r] += 1
                        
                improved = True
                break # Restart scan with new length

    return "".join(map(str, word))
