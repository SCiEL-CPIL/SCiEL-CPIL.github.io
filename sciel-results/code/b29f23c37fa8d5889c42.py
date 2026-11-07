import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 50.0
    N = 40
    pool = list(range(2, N + 1))
    rng = random.Random(987654321)

    def timeup():
        return (time.time() - t0) > DEADLINE

    def is_valid_add(prods, x):
        for q in prods:
            if q % x == 0 and (q // x) in prods:
                return False
        return True

    def rebuild_prods(current):
        prods = {1}
        for x in current:
            prods |= {p * x for p in prods}
        return prods

    def local_search(order):
        current = []
        cur_set = set()
        prods = {1}
        # initial greedy
        for x in order:
            if x in cur_set:
                continue
            if is_valid_add(prods, x):
                prods |= {p * x for p in prods}
                current.append(x)
                cur_set.add(x)
        # improvement loop with kicks
        for _ in range(40):
            if timeup():
                break
            added = False
            for x in order:
                if x in cur_set:
                    continue
                if is_valid_add(prods, x):
                    prods |= {p * x for p in prods}
                    current.append(x)
                    cur_set.add(x)
                    added = True
                    break
            if not added:
                if not current:
                    break
                e = rng.choice(current)
                current.remove(e)
                cur_set.discard(e)
                prods = rebuild_prods(current)
                for x in order:
                    if x in cur_set:
                        continue
                    if is_valid_add(prods, x):
                        prods |= {p * x for p in prods}
                        current.append(x)
                        cur_set.add(x)
        return current

    best = []
    best_len = 0
    for r in range(5000):
        if timeup():
            break
        order = pool[:]
        rng.shuffle(order)
        cur = local_search(order)
        if len(cur) > best_len:
            best_len = len(cur)
            best = cur[:]
        if best_len >= 18:
            break

    result = [int(x) for x in best]
    prods = {1}
    for x in result:
        prods |= {p * x for p in prods}
    if len(prods) != (1 << len(result)):
        prods = {1}
        safe = []
        for x in sorted(result):
            if is_valid_add(prods, x):
                prods |= {p * x for p in prods}
                safe.append(x)
        result = safe
    return result
