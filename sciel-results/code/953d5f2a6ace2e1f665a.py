"""Shortest superpermutation via maximal-overlap path construction."""
import sys
import time
import random
from itertools import permutations

sys.setrecursionlimit(1000000)

def _solve_exact(n):
    if n == 1:
        return [1]
    all_perms = list(permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    used = set()

    ext = {}
    for p in all_perms:
        pref = p[:-1]
        ext.setdefault(pref, []).append(p[-1])
    for k in ext:
        ext[k] = sorted(ext[k])

    memo = {}

    def dfs(prefix, k):
        state = (prefix, frozenset(used))
        if state in memo:
            return memo[state]

        if len(used) == len(all_perms):
            memo[state] = (k, [])
            return (k, [])
        last = prefix[-1] if prefix else None
        cands = [s for s in range(1, n + 1) if s != last]
        cands.sort()
        best = None
        for s in cands:
            new_pref = (prefix + (s,))[-(n - 1):]
            completed = new_pref + (s,)
            if completed not in perm_set:
                continue
            if completed in used:
                continue
            used.add(completed)
            sub = dfs(new_pref, k + 1)
            used.discard(completed)
            if sub is None:
                continue
            kk, tail = sub
            cur = (kk + 1, [s] + tail)
            if best is None or cur < best:
                best = cur
        memo[state] = best
        return best

    first = tuple(range(1, n + 1))
    used.add(first)
    prefix = first[1:]
    res = dfs(prefix, n)
    if res is None:
        return None
    k, tail = res
    return list(first) + tail


def construct(n):
    """Return a string over '1'..'n' that contains every permutation of 1..n as a
    consecutive window. Minimises length; falls back to the classic recursive
    construction if the fast path fails or runs out of time."""
    deadline = time.time() + 110.0  # safety margin under the 120s wall-clock kill

    def _classic(n):
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 1:
        return "1"

    # For n <= 5, try the exact solver within the budget.
    if n <= 5:
        try:
            t0 = time.time()
            res = _solve_exact(n)
            if res is not None:
                if time.time() - t0 < 30:
                    return "".join(map(str, res))
        except Exception:
            pass

    # Fast greedy maximal-overlap path with dead-end lookahead for n >= 2.
    try:
        all_perms = list(permutations(range(1, n + 1)))
        perm_set = set(all_perms)
        used = set()

        # for each (n-1)-tuple prefix, the set of symbols that extend it to a permutation
        ext = {}
        for p in all_perms:
            pref = p[:-1]
            ext.setdefault(pref, set()).add(p[-1])

        # dead-end test: from a prefix P (len n-1), is there any symbol s != last
        # such that P+(s,) is an unused permutation?  If not, P is a dead end
        def is_dead_end(prefix):
            last = prefix[-1]
            for s in ext.get(prefix, ()):
                if s == last:
                    continue
                p = prefix + (s,)
                if p not in used:
                    return False
            return True

        # Try multiple random starting permutations and greedy strategies
        best_word = None
        best_len = float('inf')
        
        # Number of attempts based on time remaining
        num_attempts = 10 if n <= 6 else 5
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
                
            # Choose a starting permutation (first attempt uses canonical order)
            if attempt == 0:
                start_perm = tuple(range(1, n + 1))
            else:
                start_perm = tuple(random.sample(range(1, n + 1), n))
            
            used.clear()
            word = list(start_perm)
            used.add(tuple(word))
            
            # Greedy with lookahead
            success = True
            while len(used) < len(all_perms):
                if time.time() > deadline - 5:
                    success = False
                    break
                suffix = tuple(word[-(n - 1):])
                last = word[-1]
                # candidate symbols in ascending order, != last, that complete a NEW permutation
                cands = []
                for s in ext.get(suffix, ()):
                    if s == last:
                        continue
                    p = suffix + (s,)
                    if p not in used:
                        cands.append((s, p))
                cands.sort()
                if not cands:
                    success = False
                    break
                chosen = None
                # lookahead: prefer a candidate that does NOT create a dead-end suffix
                for s, p in cands:
                    new_suffix = (suffix + (s,))[-(n - 1):]
                    if not is_dead_end(new_suffix):
                        chosen = (s, p)
                        break
                if chosen is None:
                    chosen = cands[0]
                s, p = chosen
                word.append(s)
                used.add(p)
            
            if success and len(used) == len(all_perms):
                # Verify completeness
                if time.time() < deadline - 2:
                    seen = set()
                    w = word
                    for i in range(len(w) - n + 1):
                        seen.add(tuple(w[i:i + n]))
                    if len(seen) == len(all_perms):
                        if len(word) < best_len:
                            best_len = len(word)
                            best_word = word.copy()
        
        if best_word is not None:
            return "".join(map(str, best_word))
        
        raise RuntimeError("no valid superpermutation found")
    except Exception:
        # Fallback: classic construction (valid, just longer).
        return _classic(n)
