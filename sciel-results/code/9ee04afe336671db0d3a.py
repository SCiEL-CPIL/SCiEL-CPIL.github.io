"""Shortest superpermutation via maximal-overlap path construction."""
import sys
import time
import random
from itertools import permutations

sys.setrecursionlimit(1000000)

# ---- Exact memoized solver for small n ----
def _solve_exact(n):
    if n == 1:
        return [1]
    all_perms = list(permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    used = set()

    # precompute: for each symbol x, which permutations end in x
    perms_end = [[] for _ in range(n + 1)]
    for p in all_perms:
        perms_end[p[-1]].append(p)

    # for each prefix (last n-1 symbols), which symbol(s) extend it to a permutation
    # ext[prefix] = list of symbols s such that prefix + (s,) in perm_set
    ext = {}
    for p in all_perms:
        pref = p[:-1]
        ext.setdefault(pref, []).append(p[-1])
    for k in ext:
        ext[k] = sorted(ext[k])

    memo = {}

    def dfs(prefix, k):
        # prefix: tuple of last n-1 symbols (or shorter); k: count appended so far
        # returns best (k, suffix-list) for completing from here
        state = (prefix, frozenset(used))
        if state in memo:
            return memo[state]

        # we've placed k symbols; need all n! permutations covered
        # A permutation is "covered" once it has appeared as a window.
        # We track used permutations.
        if len(used) == len(all_perms):
            memo[state] = (k, [])
            return (k, [])
        last = prefix[-1] if prefix else None
        # candidate symbols: smallest first, skip == last
        cands = [s for s in range(1, n + 1) if s != last]
        cands.sort()
        best = None
        for s in cands:
            new_pref = (prefix + (s,))[-(n - 1):]
            # which permutations does appending s complete?
            # any permutation p with p[:-1] == new_pref and p[-1] == s, i.e. p == new_pref + (s,)
            completed = new_pref + (s,)
            if completed not in perm_set:
                # appending s forms no new permutation window that's valid;
                # but window may still be non-permutation; only valid permutations count.
                # If completed not in perm_set, no permutation is newly covered -> skip
                continue
            if completed in used:
                # already covered; appending s gives no new permutation
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

    # start: seed with first permutation
    # choose starting permutation = (1,2,...,n); mark it used, prefix = last n-1
    first = tuple(range(1, n + 1))
    used.add(first)
    prefix = first[1:]
    res = dfs(prefix, n)  # we've already placed n symbols for the first permutation
    if res is None:
        return None
    k, tail = res
    return list(first) + tail


def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
                    break
            new += block[overlap:]
        word = new
    return word

def _build_level_perms(n):
    """
    Build the list of (n-1)-length permutations in the order they appear
    in the classic superpermutation of level n-1.
    """
    if n == 1:
        return [[1]]
    word = _classic_superperm(n - 1)
    perms, seen = [], set()
    m = n - 1
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    return [list(p) for p in perms]

def _overlap(a, b):
    """Maximum overlap: suffix of a matches prefix of b."""
    max_ov = min(len(a), len(b))
    for o in range(max_ov, 0, -1):
        if a[-o:] == b[:o]:
            return o
    return 0

def _block_for_perm(p, m):
    """The block for permutation p with new symbol m: p + [m] + p."""
    return list(p) + [m] + list(p)

def _optimize_order_tsp(perm_list, m, time_limit=10.0):
    """
    Try to find a better ordering of perm_list to minimize total length.
    Uses a greedy + local search approach.
    """
    start_time = time.time()
    n_perms = len(perm_list)
    
    # Initial order: the classic order
    order = list(range(n_perms))
    
    # Compute all-pairs overlap matrix
    blocks = [_block_for_perm(p, m) for p in perm_list]
    n = n_perms
    ov_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _overlap(blocks[i], blocks[j])
    
    # Cost of transition i->j is len(blocks[j]) - ov_matrix[i][j]
    # Total cost = len(blocks[0]) + sum of transition costs
    
    def current_cost(ord):
        cost = len(blocks[ord[0]])
        for k in range(1, len(ord)):
            i, j = ord[k-1], ord[k]
            cost += len(blocks[j]) - ov_matrix[i][j]
        return cost
    
    best_order = list(order)
    best_cost = current_cost(best_order)
    
    # 2-opt local search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n - 1):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                # Reverse segment [i+1..j]
                new_order = best_order[:i+1] + best_order[i+1:j+1][::-1] + best_order[j+1:]
                new_cost = current_cost(new_order)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_order = new_order
                    improved = True
    
    # Also try some random restarts if time permits
    if (time.time() - start_time) < time_limit - 2.0:
        random.seed(42)
        for _ in range(5):
            if (time.time() - start_time) > time_limit - 1.0:
                break
            rand_order = list(range(n))
            random.shuffle(rand_order)
            # Local search from random start
            cur_order = list(rand_order)
            cur_cost = current_cost(cur_order)
            imp = True
            while imp and (time.time() - start_time) < time_limit - 0.5:
                imp = False
                for i in range(n - 1):
                    if (time.time() - start_time) > time_limit - 0.5:
                        break
                    for j in range(i + 1, n):
                        new_order = cur_order[:i+1] + cur_order[i+1:j+1][::-1] + cur_order[j+1:]
                        new_cost = current_cost(new_order)
                        if new_cost < cur_cost:
                            cur_cost = new_cost
                            cur_order = new_order
                            imp = True
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_order = cur_order
    
    return best_order, best_cost


def construct(n):
    """Return a string over '1'..'n' that contains every permutation of 1..n as a
    consecutive window. Minimises length; falls back to the classic recursive
    construction if the fast path fails or runs out of time."""
    deadline = time.time() + 140.0  # safety margin under the 150s wall-clock kill

    if n == 1:
        return "1"

    if n == 2:
        return "121"
    
    if n == 3:
        return "123121321"

    # For n <= 5, try the exact solver within the budget.
    if n <= 5:
        try:
            t0 = time.time()
            res = _solve_exact(n)
            if res is not None:
                # verify
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
        # (we can't extend without repeating a symbol or covering nothing new).
        def is_dead_end(prefix):
            last = prefix[-1]
            for s in ext.get(prefix, ()):
                if s == last:
                    continue
                p = prefix + (s,)
                if p not in used:
                    return False
            return True

        # Start with the smallest permutation.
        word = list(range(1, n + 1))
        used.add(tuple(word))
        # We need to make sure that from the current suffix we can always continue.
        # Greedily append the smallest usable symbol.
        while len(used) < len(all_perms):
            if time.time() > deadline:
                raise RuntimeError("time")
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
                raise RuntimeError("stuck")
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

        # Verify completeness.
        if time.time() < deadline - 1:
            seen = set()
            w = word
            for i in range(len(w) - n + 1):
                seen.add(tuple(w[i:i + n]))
            if len(seen) == len(all_perms):
                return "".join(map(str, word))
        raise RuntimeError("verify failed")
    except Exception:
        pass
    
    # Fallback: classic construction with TSP optimization
    try:
        classic_word = _classic_superperm(n)
        classic_len = len(classic_word)
        
        # Try to optimize the ordering at the top level
        if n <= 7 and (time.time() < deadline - 10.0):
            perm_list = _build_level_perms(n)
            m = n
            n_perms = len(perm_list)
            
            if n_perms <= 720:
                time_limit_for_tsp = min(10.0, deadline - (time.time()) - 5.0)
                if time_limit_for_tsp > 1.0:
                    best_order, best_cost = _optimize_order_tsp(perm_list, m, time_limit=time_limit_for_tsp)
                    
                    # Build the word using the optimized order
                    blocks = [_block_for_perm(perm_list[i], m) for i in best_order]
                    opt_word = list(blocks[0])
                    for k in range(1, len(blocks)):
                        b = blocks[k]
                        ov = _overlap(opt_word, b)
                        opt_word += b[ov:]
                    
                    if len(opt_word) < classic_len:
                        return "".join(map(str, opt_word))
        
        return "".join(map(str, classic_word))
    except Exception:
        # Ultimate fallback
        return "".join(map(str, _classic_superperm(n)))
