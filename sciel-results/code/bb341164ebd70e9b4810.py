import time
import sys
from itertools import permutations

# Precompute permutation index for n=7 (and smaller)
# Map each permutation tuple to a unique integer index
_PERM_INDEX = {}
_PERM_LIST = []
_PERM_INV = []  # inverse: index -> tuple

def _build_perm_tables(n):
    """Build permutation index tables for n symbols."""
    global _PERM_INDEX, _PERM_LIST, _PERM_INV
    _PERM_INDEX = {}
    _PERM_LIST = []
    _PERM_INV = []
    idx = 0
    for p in permutations(range(1, n + 1)):
        _PERM_INDEX[p] = idx
        _PERM_LIST.append(p)
        _PERM_INV.append(p)
        idx += 1

def _classic_superperm(n):
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
            # Find max overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    total_perms = n * _factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == total_perms

def _fast_delete_pass(word, n, time_limit, start_time):
    """
    Efficient deletion pass.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    
    # Helper to compute window index
    def get_window_idx(w, start):
        if start < 0 or start + n > len(w):
            return -1
        window_tuple = tuple(w[start:start + n])
        if len(set(window_tuple)) != n:
            return -1
        return _PERM_INDEX.get(window_tuple, -1)

    # Initialize window_perm and perm_count
    L = len(word)
    window_perm = [-1] * (L - n + 1)
    perm_count = [0] * total_perms
    
    for s in range(L - n + 1):
        idx = get_window_idx(word, s)
        if idx >= 0:
            window_perm[s] = idx
            perm_count[idx] += 1
            
    # Helper to update window_perm and perm_count after deletion at pos i
    def update_after_delete(i):
        nonlocal window_perm, perm_count, L
        # Old length L, new length L-1
        # Deletion at i.
        # Affected old windows: s in [i-n+1, i]
        old_s_min = max(0, i - n + 1)
        old_s_max = min(i, L - n)
        
        for s in range(old_s_min, old_s_max + 1):
            if window_perm[s] >= 0:
                perm_count[window_perm[s]] -= 1
                window_perm[s] = -1
        
        # New length L-1
        L -= 1
        # Resize window_perm? 
        # New window starts range: [0, L-n]
        # We need to update window_perm for s in [i-n+1, L-n] (new indices)
        # Note: window_perm array is indexed by start s.
        # When we delete i, windows with s > i shift to s-1.
        # Windows with s < i-n+1 are unaffected.
        # Windows with s in [i-n+1, i] are destroyed/recomputed.
        
        # It is easier to just recompute the affected range in the new word.
        # New word length is L.
        # Affected new starts: s_new in [max(0, i-n+1), min(i, L-n)]
        # Wait, if we delete i, new position j corresponds to old j if j<i, old j+1 if j>=i.
        # New window at s_new covers [s_new, s_new+n-1].
        # This window is affected if it includes a position >= i (shifted) 
        # OR if it is a window that didn't exist before.
        
        # Let's just recompute all windows in the range [max(0, i-n+1), min(i, L-n)] 
        # AND also handle the shift for s > i.
        
        # Actually, the simplest robust way for small n and L~6000 is:
        # 1. Delete from word list.
        # 2. Rebuild window_perm and perm_count from scratch? 
        #    Cost: L*n = 42000 ops.
        #    With ~10-20 deletions, total cost 400k-800k ops.
        #    This is negligible (<0.1s).
        
        pass # We will implement the full rebuild for simplicity and safety, 
             # as the local update logic is prone to off-by-one errors with shifts.
        
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Scan from end to start? Or start to end?
        # Scanning from start to end is fine.
        for i in range(len(word)):
            if time.time() - start_time > time_limit:
                return word
            
            # Check if position i can be deleted
            # Windows containing i: s in [i-n+1, i]
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            can_delete = True
            for s in range(s_min, s_max + 1):
                if s < len(window_perm) and window_perm[s] >= 0:
                    if perm_count[window_perm[s]] == 1:
                        can_delete = False
                        break
            
            if can_delete:
                # Delete word[i]
                del word[i]
                
                # Rebuild window_perm and perm_count
                # This is O(L*n). 
                # Since we expect only ~8 deletions, this is fast.
                L_new = len(word)
                window_perm = [-1] * (L_new - n + 1)
                perm_count = [0] * total_perms
                for s in range(L_new - n + 1):
                    idx = get_window_idx(word, s)
                    if idx >= 0:
                        window_perm[s] = idx
                        perm_count[idx] += 1
                        
                improved = True
                break # Restart scan
    
    return word

def construct(n):
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin for 150s limit
    
    # Build permutation tables
    _build_perm_tables(n)
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Try to remove redundant characters using fast deletion
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _fast_delete_pass(word, n, remaining_time, start_time)
    
    # Final validation
    if not _is_valid_superperm(word, n):
        # Fallback to classic
        word = _classic_superperm(n)
    
    return "".join(map(str, word))
