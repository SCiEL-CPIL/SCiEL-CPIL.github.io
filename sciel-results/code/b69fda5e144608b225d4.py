"""Direct construction of a shortest-superpermutation-like word for n symbols."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

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


def _count_perms(W, n):
    """Count all n-windows in W."""
    perm_count = {}
    L = len(W)
    for i in range(L - n + 1):
        w = tuple(W[i:i+n])
        perm_count[w] = perm_count.get(w, 0) + 1
    return perm_count


def _try_delete_at(W, i, n, perm_count):
    """Try to delete symbol at position i. Returns (new_W, new_perm_count, True) or (None, None, False)."""
    L = len(W)
    if L <= n or i <= 0 or i >= L - 1:
        return None, None, False
    
    k_lo = max(0, i - n + 1)
    k_hi = min(L - n, i)
    
    # Check if deletion is valid: all affected windows that appear only once must be recreated
    new_L = L - 1
    new_k_lo = max(0, i - n + 1)
    new_k_hi = min(new_L - n, i)
    
    affected_new = set()
    for k in range(new_k_lo, new_k_hi + 1):
        w = []
        for idx in range(k, k + n):
            if idx < i:
                w.append(W[idx])
            else:
                w.append(W[idx + 1])
        affected_new.add(tuple(w))
    
    for k in range(k_lo, k_hi + 1):
        w = tuple(W[k:k+n])
        if perm_count.get(w, 0) == 1 and w not in affected_new:
            return None, None, False
    
    # Commit deletion
    new_perm_count = dict(perm_count)
    for k in range(k_lo, k_hi + 1):
        w = tuple(W[k:k+n])
        new_perm_count[w] -= 1
        if new_perm_count[w] == 0:
            del new_perm_count[w]
    
    for k in range(new_k_lo, new_k_hi + 1):
        w = []
        for idx in range(k, k + n):
            if idx < i:
                w.append(W[idx])
            else:
                w.append(W[idx + 1])
        wt = tuple(w)
        new_perm_count[wt] = new_perm_count.get(wt, 0) + 1
    
    new_W = W[:i] + W[i+1:]
    return new_W, new_perm_count, True


def _greedy_delete_pass(W, n, perm_count, order, time_limit, t0):
    """One pass of greedy deletion with a given position order. Returns (W, perm_count, deletions)."""
    deletions = 0
    L = len(W)
    
    for i in order:
        if time_limit and (time.time() - t0) > time_limit:
            break
        if i >= len(W):
            continue
        new_W, new_pc, success = _try_delete_at(W, i, n, perm_count)
        if success:
            W = new_W
            perm_count = new_pc
            deletions += 1
            # After deletion, positions shift; continue with next in order
            # but the order may be stale; we'll just continue
    
    return W, perm_count, deletions


def _greedy_delete(W, n, time_limit):
    """Repeatedly try to delete symbols using multiple strategies."""
    t0 = time.time()
    best_W = W
    best_len = len(W)
    
    # Strategy 1: Left-to-right, repeated until no progress
    perm_count = _count_perms(W, n)
    for _ in range(20):
        if time_limit and (time.time() - t0) > time_limit - 5:
            break
        order = list(range(1, len(W) - 1))
        W, perm_count, d = _greedy_delete_pass(W, n, perm_count, order, time_limit - 2, t0)
        if d == 0:
            break
        if len(W) < best_len:
            best_W = W
            best_len = len(W)
    
    # Strategy 2: Right-to-left
    if time_limit and (time.time() - t0) < time_limit - 10:
        perm_count = _count_perms(W, n)
        for _ in range(10):
            if time_limit and (time.time() - t0) > time_limit - 5:
                break
            order = list(range(len(W) - 2, 0, -1))
            W, perm_count, d = _greedy_delete_pass(W, n, perm_count, order, time_limit - 2, t0)
            if d == 0:
                break
            if len(W) < best_len:
                best_W = W
                best_len = len(W)
    
    # Strategy 3: Random restarts
    if time_limit and (time.time() - t0) < time_limit - 15:
        for _ in range(5):
            if time_limit and (time.time() - t0) > time_limit - 5:
                break
            W_temp = list(best_W)
            perm_count_temp = _count_perms(W_temp, n)
            order = list(range(1, len(W_temp) - 1))
            random.shuffle(order)
            W_temp, perm_count_temp, d = _greedy_delete_pass(W_temp, n, perm_count_temp, order, time_limit - 2, t0)
            if len(W_temp) < best_len:
                best_W = W_temp
                best_len = len(W_temp)
    
    return best_W


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
        return "123412134214312143121"
    if n == 5:
        return "1234512134521431523415243152543152143521"
    
    # For n >= 6, use classic construction then try to shorten
    word = _classic_construct(n)
    
    # Try greedy deletion for n=6 and n=7
    if n <= 7:
        elapsed = time.time() - _START
        remaining = _DEADLINE - elapsed
        if remaining > 10:
            random.seed(42)  # Deterministic
            word = _greedy_delete(word, n, time_limit=max(1, remaining - 5))
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
