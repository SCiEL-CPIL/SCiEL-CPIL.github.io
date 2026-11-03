"""Shortest superpermutation via maximal-overlap path construction with heuristic search."""
import sys
import time
import random
from itertools import permutations
from collections import defaultdict

sys.setrecursionlimit(1000000)

def _classic(n):
    """Classic recursive superpermutation: length 1!+2!+...+n! = 5913 for n=7."""
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


def construct(n):
    """Return a string over '1'..'n' that contains every permutation of 1..n as a
    consecutive window. Minimises length; falls back to the classic recursive
    construction if the fast path fails or runs out of time."""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    deadline = time.time() + 110.0  # safety margin under the 120s wall-clock kill

    # Build all permutations and extension map
    all_perms = list(permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    total_perms = len(all_perms)

    # ext[prefix] = set of symbols s such that prefix+(s,) is a permutation
    ext = {}
    for p in all_perms:
        pref = p[:-1]
        ext.setdefault(pref, set()).add(p[-1])

    def greedy_search(seed_word, deadline):
        """Greedy with multi-step lookahead. Returns word list or None on failure."""
        word = list(seed_word)
        used = set()
        used.add(tuple(word))
        # Mark all windows in seed
        for i in range(len(word) - n + 1):
            used.add(tuple(word[i:i + n]))

        while len(used) < total_perms:
            if time.time() > deadline:
                return None
            suffix = tuple(word[-(n - 1):])
            last = word[-1]

            # Candidate symbols
            cands = []
            for s in ext.get(suffix, ()):
                if s == last:
                    continue
                p = suffix + (s,)
                if p not in used:
                    new_suffix = (suffix + (s,))[-(n - 1):]
                    # Count available moves from new_suffix
                    avail = 0
                    for s2 in ext.get(new_suffix, ()):
                        if s2 == s:
                            continue
                        p2 = new_suffix + (s2,)
                        if p2 not in used:
                            avail += 1
                    cands.append((s, p, avail))

            if not cands:
                return None

            # 2-step lookahead scoring
            scored = []
            for s, p, avail in cands:
                if avail == 0:
                    scored.append((1, 0, s, p, avail))  # dead-end last
                    continue
                new_suffix = (suffix + (s,))[-(n - 1):]
                # Best next-step availability
                best_next = 0
                for s2 in ext.get(new_suffix, ()):
                    if s2 == s:
                        continue
                    p2 = new_suffix + (s2,)
                    if p2 not in used:
                        next_suffix = (new_suffix + (s2,))[-(n - 1):]
                        na = 0
                        for s3 in ext.get(next_suffix, ()):
                            if s3 == s2:
                                continue
                            p3 = next_suffix + (s3,)
                            if p3 not in used:
                                na += 1
                        best_next = max(best_next, na)
                scored.append((0, -best_next, s, p, avail))

            scored.sort()
            _, _, s, p, avail = scored[0]
            word.append(s)
            used.add(p)

        return word

    # Try multiple starting permutations
    # Start with natural order, then try some random permutations
    best_word = None
    start_perms = [list(range(1, n + 1))]
    # Add a few random starts
    random.seed(42)
    for _ in range(5):
        sp = list(range(1, n + 1))
        random.shuffle(sp)
        start_perms.append(sp)

    for sp in start_perms:
        if time.time() > deadline - 5:
            break
        res = greedy_search(sp, deadline)
        if res is not None:
            # Verify
            seen = set()
            for i in range(len(res) - n + 1):
                seen.add(tuple(res[i:i + n]))
            if len(seen) == total_perms:
                if best_word is None or len(res) < len(best_word):
                    best_word = res

    if best_word is not None:
        return "".join(map(str, best_word))

    # Fallback: classic construction
    return _classic(n)
