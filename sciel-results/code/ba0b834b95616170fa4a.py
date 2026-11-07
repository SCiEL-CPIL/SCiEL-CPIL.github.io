"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 256, 'maximum_size': 30}

def _verify(s):
    """Check if set s has unique subset sums and no divisibility among them."""
    n = len(s)
    if n == 0:
        return True
    if n > 14:
        return False
    sums = {0}
    for x in s:
        sums = sums | {c + x for c in sums}
    nonempty = sorted(sums - {0})
    if len(nonempty) != (1 << n) - 1:
        return False
    max_sum = nonempty[-1]
    nonempty_set = set(nonempty)
    for v in nonempty:
        if v > max_sum // 2:
            break
        m = 2 * v
        while m <= max_sum:
            if m in nonempty_set:
                return False
            m += v
    return True

def construct():
    deadline = time.time() + 80
    best_set = [31, 32, 34]
    best_size = 3
    random.seed(42)
    
    def try_grow(start_set):
        current = list(start_set)
        cur_set = set(current)
        stall = 0
        while time.time() < deadline and stall < 80:
            added = False
            candidates = [x for x in range(256, 0, -1) if x not in cur_set]
            random.shuffle(candidates[:120])
            for cand in candidates[:100]:
                new_set = current + [cand]
                if _verify(new_set):
                    current = new_set
                    cur_set.add(cand)
                    added = True
                    stall = 0
                    break
            if not added:
                stall += 1
                if len(current) > 2:
                    idx = random.randint(0, len(current) - 1)
                    removed = current.pop(idx)
                    cur_set.discard(removed)
                    cands2 = [x for x in range(256, 0, -1) if x not in cur_set]
                    random.shuffle(cands2[:120])
                    for cand in cands2[:80]:
                        new_set = current + [cand]
                        if _verify(new_set):
                            current.append(cand)
                            cur_set.add(cand)
                            added = True
                            break
                    if not added:
                        current.append(removed)
                        cur_set.add(removed)
        return current
    
    starts = [
        [128, 192, 224, 240],
        [100, 150, 200, 240],
        [80, 120, 160, 200, 240],
        [128, 160, 192, 224, 240],
        [64, 96, 128, 160, 192, 224],
        [128, 144, 160, 176, 192, 208],
        [100, 130, 160, 190, 220, 250],
    ]
    
    for st in starts:
        if time.time() > deadline:
            break
        if _verify(st):
            grown = try_grow(st)
            if len(grown) > best_size and _verify(grown):
                best_set = grown
                best_size = len(grown)
    
    for _ in range(25):
        if time.time() > deadline:
            break
        size = random.randint(3, 10)
        pool = random.sample(range(60, 257), min(size + 5, 197))
        random.shuffle(pool)
        current = []
        for cand in pool:
            if len(current) >= 14:
                break
            if _verify(current + [cand]):
                current.append(cand)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
        if len(current) >= 2:
            grown = try_grow(current)
            if len(grown) > best_size and _verify(grown):
                best_set = grown
                best_size = len(grown)
    
    for lo in range(90, 200, 10):
        if time.time() > deadline:
            break
        pool = list(range(lo, min(lo + 45, 257)))
        random.shuffle(pool)
        current = []
        for cand in pool:
            if len(current) >= 14:
                break
            if _verify(current + [cand]):
                current.append(cand)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
        if len(current) >= 2:
            grown = try_grow(current)
            if len(grown) > best_size and _verify(grown):
                best_set = grown
                best_size = len(grown)
    
    if _verify(best_set):
        return list(best_set)
    return [31, 32, 34]

# EVOLVE-BLOCK-END

