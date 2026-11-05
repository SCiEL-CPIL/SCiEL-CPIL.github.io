import time
import random
from itertools import permutations

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
    return word

def construct(n):
    start = time.time()
    DEADLINE = 110.0  # safety margin before 120/150s
    word = _classic(n)
    best = list(word)
    if n < 2:
        return "".join(map(str, best))
    # Precompute permutation index map
    perm_to_idx = {p: i for i, p in enumerate(permutations(range(1, n + 1)))}
    P = len(perm_to_idx)
    nlen = n
    def recompute(w):
        L = len(w)
        win_rank = [None] * L
        count = [0] * P
        for i in range(L - nlen + 1):
            r = perm_to_idx.get(tuple(w[i:i + nlen]))
            if r is not None:
                win_rank[i] = r
                count[r] += 1
        zero = 0
        for c in count:
            if c == 0:
                zero += 1
        return win_rank, count, zero
    # Greedy deletion with restarts
    rng = random.Random(12345)
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() - start > DEADLINE:
            break
        w = list(best)
        L = len(w)
        win_rank, count, zero = recompute(w)
        if zero != 0:
            # not valid, shouldn't happen
            continue
        improved = True
        while improved:
            if time.time() - start > DEADLINE:
                break
            improved = False
            valid_positions = []
            for p in range(L):
                # test deletion at p
                zb = zero
                changes = []
                ok = True
                # old affected windows containing p
                lo = max(0, p - nlen + 1)
                hi = min(p, L - nlen)
                for i in range(lo, hi + 1):
                    r = win_rank[i]
                    if r is not None:
                        count[r] -= 1
                        changes.append((r, -1))
                        if count[r] == 0:
                            zero += 1
                # new affected windows
                lo2 = max(0, p - nlen + 1)
                hi2 = min(p, L - nlen - 1)
                for i2 in range(lo2, hi2 + 1):
                    nw = w[i2:p] + w[p + 1:i2 + nlen + 1]
                    r2 = perm_to_idx.get(tuple(nw))
                    if r2 is not None:
                        count[r2] += 1
                        changes.append((r2, 1))
                        if count[r2] == 1:
                            zero -= 1
                if zero == 0:
                    valid_positions.append(p)
                # revert
                for r, d in changes:
                    count[r] -= d
                zero = zb
            if valid_positions:
                p = rng.choice(valid_positions)
                # actually delete p
                w = w[:p] + w[p + 1:]
                L = len(w)
                win_rank, count, zero = recompute(w)
                improved = True
        if len(w) < len(best):
            best = list(w)
    return "".join(map(str, best))
