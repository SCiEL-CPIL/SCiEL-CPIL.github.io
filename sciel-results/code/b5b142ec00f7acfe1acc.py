import time
import random

def construct():
    start = time.time()
    deadline = start + 135.0

    # ---------- Phase 1: greedy seed ----------
    chosen = []
    present = set()
    cnt = [0] * 2001  # index s-2, sums 2..2000

    for x in range(1, 1001):
        ok = True
        for y in chosen:
            if cnt[x + y - 2] >= 2:
                ok = False
                break
        if ok and cnt[2 * x - 2] >= 2:
            ok = False
        if ok:
            for y in chosen:
                cnt[x + y - 2] += 1
            cnt[2 * x - 2] += 1
            chosen.append(x)
            present.add(x)

    best = list(chosen)
    best_len = len(chosen)

    # ---------- Local search helpers (operate on current chosen/present/cnt) ----------
    def try_add(x):
        if x in present:
            return False
        for y in chosen:
            if cnt[x + y - 2] >= 2:
                return False
        if cnt[2 * x - 2] >= 2:
            return False
        for y in chosen:
            cnt[x + y - 2] += 1
        cnt[2 * x - 2] += 1
        chosen.append(x)
        present.add(x)
        return True

    def try_remove(x):
        if x not in present:
            return False
        for y in chosen:
            if y != x:
                cnt[x + y - 2] -= 1
        cnt[2 * x - 2] -= 1
        chosen.remove(x)
        present.discard(x)
        return True

    rng = random.Random(12345)

    # ---------- Phase 2: randomized hill-climbing ----------
    while time.time() < deadline - 5:
        x = rng.randint(1, 1000)
        if x not in present:
            if try_add(x):
                if len(chosen) > best_len:
                    best_len = len(chosen)
                    best = list(chosen)
                continue
            if len(chosen) > 1:
                y = rng.choice(chosen)
                if try_remove(y):
                    if try_add(x):
                        if try_add(y):
                            if len(chosen) > best_len:
                                best_len = len(chosen)
                                best = list(chosen)
                    else:
                        try_add(y)

    # ---------- Phase 3: polish - try all additions ----------
    for x in range(1, 1001):
        if time.time() > deadline - 2:
            break
        if x not in present:
            try_add(x)
    if len(chosen) > best_len:
        best = list(chosen)
        best_len = len(chosen)

    # ---------- Phase 4: final local search if time ----------
    while time.time() < deadline - 1:
        x = rng.randint(1, 1000)
        if x not in present:
            if try_add(x):
                if len(chosen) > best_len:
                    best_len = len(chosen)
                    best = list(chosen)
                continue
            if len(chosen) > 1:
                y = rng.choice(chosen)
                if try_remove(y):
                    if try_add(x):
                        if try_add(y):
                            if len(chosen) > best_len:
                                best_len = len(chosen)
                                best = list(chosen)
                    else:
                        try_add(y)

    return best
