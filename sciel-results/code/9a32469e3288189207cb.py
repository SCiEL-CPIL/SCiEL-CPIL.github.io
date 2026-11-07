import time
import random

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    N = 256
    MAX_SIZE = 60
    universe = list(range(1, N + 1))

    best = []
    best_size = 0

    def try_greedy(pool, rng, hard_deadline):
        # layers[k] = set of distinct sums using exactly k elements
        layers = [set() for _ in range(MAX_SIZE + 2)]
        layers[0].add(0)
        val_to_cards = {0: {0}}
        all_vals = {0}
        chosen = []
        used = set()
        cur = 0

        for x in pool:
            if time.time() > hard_deadline:
                break
            if cur >= MAX_SIZE:
                break
            if x in used:
                continue
            # validity check: adding x must not create cross-cardinality collision
            ok = True
            for v in all_vals:
                w = v + x
                Cw = val_to_cards.get(w)
                if Cw is not None:
                    Cv = val_to_cards[v]
                    # new placement of w is at card c+1 for each c in Cv
                    allowed = False
                    for c in Cv:
                        if (c + 1) in Cw:
                            allowed = True
                    # collision if Cw has a card that is NOT (c+1) for any c in Cv
                    # i.e. exists cw in Cw such that for all c in Cv, cw != c+1
                    collide = False
                    for cw in Cw:
                        if not any(cw == c + 1 for c in Cv):
                            collide = True
                            break
                    if collide:
                        ok = False
                        break
            if not ok:
                continue
            # commit x
            used.add(x)
            chosen.append(x)
            # compute new sums from old layers
            new_by_layer = [set() for _ in range(cur + 2)]
            for k in range(cur, -1, -1):
                for s in layers[k]:
                    new_by_layer[k + 1].add(s + x)
            # merge into layers and bookkeeping
            for k in range(1, cur + 2):
                if new_by_layer[k]:
                    layers[k] = layers[k] | new_by_layer[k]
                    for v in new_by_layer[k]:
                        all_vals.add(v)
                        if v not in val_to_cards:
                            val_to_cards[v] = set()
                        val_to_cards[v].add(k)
            cur += 1
        return chosen

    # deterministic ascending baseline first
    base = try_greedy(universe, None, deadline)
    if len(base) > best_size:
        best_size = len(base)
        best = base[:]

    rng = random.Random(20240517)
    restart = 0
    while time.time() < deadline - 2.0:
        restart += 1
        # build a pool: shuffle, but bias toward smaller elements by
        # mixing ascending prefix with random tail
        pool = universe[:]
        rng.shuffle(pool)
        res = try_greedy(pool, rng, deadline)
        if len(res) > best_size:
            best_size = len(res)
            best = res[:]
            # if we hit a great size, keep searching but with fresh randomness
            rng = random.Random(rng.randrange(1, 10**9))

    # final: ensure types are plain ints in a list
    out = [int(v) for v in best]
    return out
