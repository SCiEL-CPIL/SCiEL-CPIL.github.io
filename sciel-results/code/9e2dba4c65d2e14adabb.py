import time

# EVOLVE-BLOCK-START
def construct():
    DEADLINE = 85.0  # safety margin before the 90 s kill
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    # ---------- Greedy baseline ----------
    def greedy(candidates):
        values, sums = [], set()
        for x in candidates:
            if x <= 0 or x > 256:
                continue
            if values and x <= values[-1]:
                continue
            total = x
            if total in sums:
                continue
            tails = [total]
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    base_vals, _ = greedy(range(1, 257))
    best = list(base_vals)
    best_len = len(base_vals)

    # ---------- Beam search (breadth-first, time-guarded) ----------
    K = 1500        # beam width
    CAND_LIMIT = 8  # max next candidates per node

    def feasible_next(x, values, sums):
        if x < 1 or x > 256:
            return False
        if values and x <= values[-1]:
            return False
        total = x
        if total in sums:
            return False
        for a in reversed(values):
            total += a
            if total in sums:
                return False
        return True

    beam = [([], frozenset())]
    for depth in range(best_len + 12):
        if elapsed() > DEADLINE:
            break
        new_beam = []
        for values, sums in beam:
            if elapsed() > DEADLINE:
                break
            lo = (values[-1] + 1) if values else 1
            cands = []
            for x in range(lo, 257):
                if feasible_next(x, values, sums):
                    cands.append(x)
                    if len(cands) >= CAND_LIMIT:
                        break
            for x in cands:
                total = x
                new_sums = [total]
                for a in reversed(values):
                    total += a
                    new_sums.append(total)
                new_beam.append((values + (x,), sums | frozenset(new_sums)))
        if not new_beam:
            break
        # Keep the K prefixes with the smallest last value (most headroom above).
        new_beam.sort(key=lambda e: e[0][-1])
        beam = new_beam[:K]
        L = len(beam[0][0])
        if L > best_len:
            best = list(beam[0][0])
            best_len = L

    # ---------- Randomized greedy diversification (fills spare time) ----------
    import random
    rng = random.Random(98765)
    while elapsed() < DEADLINE - 2.0:
        cand = list(range(1, 257))
        for _ in range(rng.randint(0, 4)):
            if len(cand) > 10:
                cand.pop(rng.randrange(0, min(50, len(cand))))
        vals, _ = greedy(cand)
        if len(vals) > best_len:
            best = list(vals)
            best_len = len(vals)

    return best
# EVOLVE-BLOCK-END
