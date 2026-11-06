import time
import random

N = 18
ALL = 1 << N

def _apply(net, m):
    for i, j in net:
        hi = m & (1 << j)
        lo = m & (1 << i)
        if hi and not lo:
            m ^= (1 << i) | (1 << j)
    return m

def _count_violations(net):
    """Number of masks (out of 2^N) not sorted after net. 0 => valid."""
    bad = 0
    for m in range(ALL):
        r = _apply(net, m)
        # r is sorted iff it is of form 111...000, i.e., r & (r+1) == 0
        if r & (r + 1) != 0:
            bad += 1
    return bad

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0
    random.seed(12345)

    def oe_net():
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best = oe_net()
    best_bad = _count_violations(best)
    best_len = len(best)

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    temp = 1.0
    it = 0
    while time.time() < deadline:
        it += 1
        cur = best
        cur_bad = best_bad
        if cur_bad == 0:
            op = random.random()
            if op < 0.5 and len(cur) > 1:
                idx = random.randrange(len(cur))
                cand = cur[:idx] + cur[idx + 1:]
                cand_bad = _count_violations(cand)
                if cand_bad == 0:
                    best = cand
                    best_bad = 0
                    best_len = len(best)
                    temp = max(0.05, temp * 0.97)
                else:
                    temp = max(0.05, temp * 0.99)
            elif op < 0.8:
                if len(cur) < 2:
                    continue
                idx = random.randrange(len(cur) - 1)
                cand = list(cur)
                cand[idx], cand[idx + 1] = cand[idx + 1], cand[idx]
                cand_bad = _count_violations(cand)
                if cand_bad == 0:
                    best = cand
                    best_bad = 0
                    best_len = len(best)
            else:
                if len(cur) < 1:
                    continue
                idx = random.randrange(len(cur))
                pair = cur[idx]
                cand = cur[:idx] + cur[idx + 1:]
                pos = random.randrange(len(cand) + 1)
                cand.insert(pos, pair)
                cand_bad = _count_violations(cand)
                if cand_bad == 0:
                    best = cand
                    best_bad = 0
                    best_len = len(best)
        else:
            op = random.random()
            if op < 0.4:
                pair = random.choice(all_pairs)
                pos = random.randrange(len(cur) + 1)
                cand = list(cur)
                cand.insert(pos, pair)
                cand_bad = _count_violations(cand)
                if cand_bad < cur_bad:
                    best = cand
                    best_bad = cand_bad
                    best_len = len(best)
            elif op < 0.7:
                if len(cur) < 1:
                    continue
                idx = random.randrange(len(cur))
                cand = cur[:idx] + cur[idx + 1:]
                cand_bad = _count_violations(cand)
                if cand_bad < cur_bad:
                    best = cand
                    best_bad = cand_bad
                    best_len = len(best)
            else:
                if len(cur) < 1:
                    continue
                idx = random.randrange(len(cur))
                cand = list(cur)
                cand[idx] = random.choice(all_pairs)
                cand_bad = _count_violations(cand)
                if cand_bad < cur_bad:
                    best = cand
                    best_bad = cand_bad
                    best_len = len(best)
                elif cand_bad == cur_bad and random.random() < temp:
                    best = cand
                    best_len = len(best)
            temp = max(0.05, temp * 0.999)

        if best_bad > 0 and it % 2000 == 0:
            base = oe_net()
            random.shuffle(base)
            cand = base
            cand_bad = _count_violations(cand)
            if cand_bad < best_bad:
                best = cand
                best_bad = cand_bad
                best_len = len(best)

    if best_bad == 0:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(best) - 1, -1, -1):
                cand = best[:idx] + best[idx + 1:]
                if _count_violations(cand) == 0:
                    best = cand
                    best_len = len(best)
                    changed = True

    return best
# EVOLVE-BLOCK-END
