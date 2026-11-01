"""Fresh design: fast overlap-packing greedy with deadline guard."""
import time
import random
from itertools import permutations

_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _deadline_ok():
    return _now() < 0.80 * 120.0


def construct(n):
    _now()
    if n <= 1:
        return "".join(map(str, range(1, n + 1)))

    all_perms = list(permutations(range(1, n + 1)))
    total = len(all_perms)
    covered = set()
    word = list(all_perms[0])
    covered.add(all_perms[0])

    while len(covered) < total and _deadline_ok():
        wl = len(word)
        max_ov = min(wl, n - 1)
        suffix = tuple(word[wl - max_ov:]) if max_ov > 0 else ()
        best_ov = -1
        best_perm = None
        for perm in all_perms:
            if perm in covered:
                continue
            ov = 0
            for o in range(max_ov, 0, -1):
                if suffix[wl - max_ov - o + 1:] == perm[:o]:
                    ov = o
                    break
            if ov > best_ov:
                best_ov = ov
                best_perm = perm
            elif ov == best_ov and best_perm is not None and random.random() < 0.05:
                best_perm = perm
        if best_perm is None:
            for perm in all_perms:
                if perm not in covered:
                    best_perm = perm
                    break
        if best_perm is None:
            break
        new_syms = list(best_perm[best_ov:])
        word.extend(new_syms)
        start = len(word) - n
        for i in range(len(new_syms) - n + 1):
            covered.add(tuple(word[start + i:start + i + n]))

    return "".join(map(str, word))
