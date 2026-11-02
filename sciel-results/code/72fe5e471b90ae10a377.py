"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import math
from collections import Counter

def _build_classic_superperm(n):
    """Build the classic recursive superpermutation of 1..n."""
    word = [1]
    for m in range(2, n + 1):
        seen = set()
        order = []
        for i in range(len(word) - m + 2):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                order.append(w)
        new = []
        for p in order:
            block = list(p) + [m] + list(p)
            max_ov = min(len(new), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    ov = o
                    break
            new += block[ov:]
        word = new
    return word

def _validate(word, n):
    """Check that all n! permutations of 1..n appear as windows."""
    needed = set(itertools.permutations(range(1, n + 1)))
    found = set()
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i + n])
        found.add(w)
    return found == needed

def _init_counts(word, n, all_perms):
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if w in all_perms:
            perm_counts[w] += 1
    return perm_counts

def _freq_balanced_order(word, n, all_perms, perm_counts, salt):
    """
    Build a deletion-scan order weighted by:
      - positions that sit inside *unique* (count==1) permutations first,
      - positions of over-represented symbols first (frequency balancing),
      - a deterministic scramble for diversity.
    Returns a list of indices to try deleting.
    """
    L = len(word)
    # symbol frequency
    sym_freq = Counter(word)
    maxf = max(sym_freq.values()) if sym_freq else 1

    # For each position, compute a priority:
    #   +1 if it lies in any unique perm
    #   + (symbol over-representation)
    #   - small deterministic noise
    pos_unique = [0] * L
    for s in range(L - n + 1):
        w = tuple(word[s:s + n])
        if w in all_perms and perm_counts.get(w, 0) == 1:
            for i in range(s, s + n):
                pos_unique[i] += 1

    keys = []
    for i in range(L):
        over = (sym_freq.get(word[i], 0) / maxf)  # 0..1
        noise = ((i * 7919 + salt * 104729) % 1000) / 1000.0
        key = -pos_unique[i] * 10.0 - over * 5.0 + noise
        keys.append((key, i))
    keys.sort()
    return [i for _, i in keys]

def _greedy_delete(word, n, all_perms, perm_counts, deadline, start_time, max_passes=10):
    """
    Repeatedly delete positions that are not the sole occurrence of any permutation.
    Uses frequency-balanced scan order and incremental count updates.
    """
    best_word = list(word)
    best_len = len(word)

    for pass_num in range(max_passes):
        if time.time() - start_time > deadline - 3:
            break
        L = len(word)
        if L <= n:
            break
        order = _freq_balanced_order(word, n, all_perms, perm_counts, salt=pass_num)

        changed = True
        while changed and (time.time() - start_time) < deadline - 3:
            changed = False
            for i in order:
                if i >= len(word):
                    continue
                if time.time() - start_time > deadline - 3:
                    break

                s_min = max(0, i - n + 1)
                s_max = min(i, len(word) - n)

                safe = True
                for s in range(s_min, s_max + 1):
                    w = tuple(word[s:s + n])
                    if w in all_perms and perm_counts.get(w, 0) == 1:
                        safe = False
                        break

                if safe:
                    old_windows = []
                    for s in range(max(0, i - n + 1), min(len(word) - n + 1, i + 1)):
                        old_windows.append(tuple(word[s:s + n]))

                    word.pop(i)
                    changed = True

                    for w in old_windows:
                        if w in all_perms:
                            perm_counts[w] -= 1
                            if perm_counts[w] <= 0:
                                del perm_counts[w]

                    new_lo = max(0, i - n + 1)
                    new_hi = min(len(word) - n + 1, i + 1)
                    for s in range(new_lo, new_hi):
                        new_w = tuple(word[s:s + n])
                        if new_w in all_perms:
                            perm_counts[new_w] = perm_counts.get(new_w, 0) + 1

                    if len(word) < best_len:
                        best_word = list(word)
                        best_len = len(word)

                    if len(word) != len(order):
                        order = _freq_balanced_order(word, n, all_perms, perm_counts, salt=pass_num + 100)

    return best_word

def _pair_swap_polish(word, n, all_perms, perm_counts, deadline, start_time):
    """
    Final high-precision polish: try local swaps of adjacent symbols to
    enable further deletions. Accept a swap only if it doesn't invalidate
    the superpermutation property (all perms still present) and the
    resulting word (after a quick delete pass) is shorter.
    """
    best = list(word)
    best_len = len(word)
    L = len(word)

    # quick delete pass to get a baseline
    base = _greedy_delete(list(word), n, all_perms,
                          _init_counts(word, n, all_perms),
                          deadline, start_time, max_passes=3)
    if len(base) < best_len:
        best = base
        best_len = len(base)

    # Try a few random-ish adjacent swaps, each followed by a delete pass
    for attempt in range(40):
        if time.time() - start_time > deadline - 2:
            break
        cand = list(best)
        Lc = len(cand)
        if Lc <= n + 1:
            break
        # deterministic pseudo-random pair
        r = (attempt * 2654435761) % (Lc - 1)
        a = r
        b = r + 1
        cand[a], cand[b] = cand[b], cand[a]

        # quick validity check: does cand still contain all perms?
        # (cheap: count windows)
        cc = _init_counts(cand, n, all_perms)
        if len(cc) < len(all_perms):
            continue

        polished = _greedy_delete(cand, n, all_perms, cc,
                                  deadline, start_time, max_passes=3)
        if len(polished) < best_len:
            best = polished
            best_len = len(polished)

    return best

def construct(n):
    if n == 1:
        return "1"

    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    word = _build_classic_superperm(n)

    if n <= 4:
        return "".join(map(str, word))

    all_perms = set(itertools.permutations(range(1, n + 1)))
    perm_counts = _init_counts(word, n, all_perms)

    # Main greedy deletion with frequency balancing
    word = _greedy_delete(word, n, all_perms, perm_counts, deadline, start_time, max_passes=12)

    # Final polish pass (swap + delete) if time remains
    if time.time() - start_time < deadline - 10:
        perm_counts = _init_counts(word, n, all_perms)
        word = _pair_swap_polish(word, n, all_perms, perm_counts, deadline, start_time)

    # Final safety validation; fall back to classic if broken
    if not _validate(word, n):
        word = _build_classic_superperm(n)

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
