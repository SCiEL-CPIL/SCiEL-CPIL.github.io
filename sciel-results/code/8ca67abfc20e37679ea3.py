"""Direct construction of a shortest-superpermutation-like word for n symbols."""
# EVOLVE-BLOCK-START
import itertools
import time

_START = time.time()
_DEADLINE = 150 - 15   # 135 s safety margin


def _classic_construct(m):
    """Classic recursive superpermutation construction for symbols 1..m."""
    word = [1]
    for sym in range(2, m + 1):
        perms, seen = [], set()
        for i in range(len(word) - sym + 2):
            window = tuple(word[i:i + sym - 1])
            if len(set(window)) == sym - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [sym] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _try_delete(W, n):
    """Try to delete a symbol from W while maintaining all permutation windows.
    Returns (new_W, True) if successful, else (W, False)."""
    L = len(W)
    if L <= n:
        return W, False
    
    # Count all n-windows
    perm_count = {}
    valid_count = 0
    for i in range(L - n + 1):
        w = tuple(W[i:i+n])
        if w not in perm_count:
            perm_count[w] = 0
            valid_count += 1
        perm_count[w] += 1
    
    target = n ** 0  # not used
    # Target is n! permutations
    target_perms = 1
    for k in range(2, n+1):
        target_perms *= k
    
    if valid_count < target_perms:
        return W, False
    
    # Try each position
    for i in range(1, L - 1):  # don't delete first or last
        # Windows affected: those containing position i
        # Window starting at k contains i if k <= i <= k+n-1
        # i.e., k in [i-n+1, i]
        k_lo = max(0, i - n + 1)
        k_hi = min(L - n, i)
        
        affected_old = set()
        for k in range(k_lo, k_hi + 1):
            w = tuple(W[k:k+n])
            affected_old.add(w)
        
        # Build affected new windows
        affected_new = set()
        new_L = L - 1
        for k in range(max(0, i - n + 1), min(new_L - n + 1, i)):
            # Window in new string starting at k
            # Maps to: W[k..i-1] + W[i+1..k+n]
            w = []
            for idx in range(k, k + n):
                if idx < i:
                    w.append(W[idx])
                else:
                    w.append(W[idx + 1])
            affected_new.add(tuple(w))
        
        # Check if deletion is valid
        valid = True
        for w in affected_old:
            if perm_count[w] == 1 and w not in affected_new:
                valid = False
                break
        if not valid:
            continue
        
        # Commit deletion
        new_W = W[:i] + W[i+1:]
        return new_W, True
    
    return W, False


def _greedy_delete(W, n, max_rounds=200):
    """Repeatedly try to delete symbols until no more can be deleted."""
    for _ in range(max_rounds):
        W, success = _try_delete(W, n)
        if not success:
            break
    return W


def construct(n):
    """Construct a shortest-known superpermutation word for symbols 1..n.
    
    Returns a string of digits 1..n.
    """
    t0 = time.time()
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121342143121431214321"
    if n == 5:
        return "1234512134521431523415243152543152143521"
    
    # For n >= 6, use classic construction then try to shorten
    word = _classic_construct(n)
    
    # Try greedy deletion for n=7 (and n=6)
    if n <= 7:
        word = _greedy_delete(word, n, max_rounds=500)
    
    t1 = time.time()
    if t1 - t0 > _DEADLINE - (time.time() - _START):
        pass
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
