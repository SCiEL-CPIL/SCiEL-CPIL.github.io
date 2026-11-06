"""Bitset-accelerated optimizer for 32 integers in 1..512 maximizing distinct 3-AP differences."""
import time
import random

N = 32
U = 512
DEADLINE = time.time() + 140.0  # safety margin under the 150s wall-clock kill


def _score_from_bits(bits):
    """Count distinct d>0 with a 3-term AP, given a bitset (int) of chosen positions (1-indexed)."""
    cnt = 0
    d = 1
    while d <= (U - 1) // 2:
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            cnt += 1
        d += 1
    return cnt


def _score_from_set(s):
    b = 0
    for x in s:
        b |= 1 << x
    return _score_from_bits(b)


def _delta_score(bits, add, rem):
    """Score change of removing rem and adding add (both single positions, both in 1..512)."""
    old = _score_from_bits(bits)
    bits2 = (bits & ~(1 << rem)) | (1 << add)
    return _score_from_bits(bits2) - old


def _local_search(bits, budget):
    """Exhaustive 1-swap local search to a fixed point."""
    s = set()
    for p in range(1, U + 1):
        if (bits >> p) & 1:
            s.add(p)
    best = bits
    best_score = _score_from_bits(bits)
    improved = True
    while improved and time.time() < budget:
        improved = False
        for rem in list(s):
            rbit = 1 << rem
            for add in range(1, U + 1):
                if add in s:
                    continue
                d = _delta_score(best, add, rem)
                if d > 0:
                    best = (best & ~rbit) | (1 << add)
                    s.remove(rem)
                    s.add(add)
                    best_score += d
                    improved = True
                    break
            else:
                continue
            break
    return best, best_score


def _anneal(bits, t0, t1, iters, budget):
    """Simulated annealing with 1-swap moves."""
    s = set()
    for p in range(1, U + 1):
        if (bits >> p) & 1:
            s.add(p)
    cur = bits
    cur_score = _score_from_bits(bits)
    best = cur
    best_score = cur_score
    step = (t1 - t0) / max(1, iters)
    for i in range(iters):
        if time.time() >= budget:
            break
        T = t0 + i * step
        rem = random.choice(tuple(s))
        add = random.randrange(1, U + 1)
        while add in s:
            add = random.randrange(1, U + 1)
        d = _delta_score(cur, add, rem)
        if d >= 0 or random.random() < min(1.0, d / (T + 1e-9)):
            cur = (cur & ~(1 << rem)) | (1 << add)
            s.remove(rem)
            s.add(add)
            cur_score += d
            if cur_score > best_score:
                best = cur
                best_score = cur_score
    return best, best_score


def _initial_bits():
    chosen = random.sample(range(1, U + 1), N)
    b = 0
    for x in chosen:
        b |= 1 << x
    return b


def _bits_to_list(bits):
    out = []
    for p in range(1, U + 1):
        if (bits >> p) & 1:
            out.append(p)
    return out


def construct():
    random.seed(20240607)
    global DEADLINE
    DEADLINE = time.time() + 140.0
    budget = DEADLINE - 3.0  # reserve margin

    best_bits = None
    best_score = -1
    restart = 0
    while time.time() < budget:
        restart += 1
        bits = _initial_bits()
        bits, sc = _local_search(bits, budget)
        if sc > best_score:
            best_score = sc
            best_bits = bits
        if time.time() >= budget:
            break
        # simulated annealing phase, scaled to remaining time
        remaining = budget - time.time()
        if remaining < 2.0:
            break
        iters = max(200, int(remaining * 600))
        t0, t1 = 12.0, 0.5
        a_bits, a_sc = _anneal(bits, t0, t1, iters, budget)
        if a_sc > best_score:
            best_score = a_sc
            best_bits = a_bits
        # polish the best-so-far
        p_bits, p_sc = _local_search(best_bits, budget)
        if p_sc > best_score:
            best_score = p_sc
            best_bits = p_bits

    if best_bits is None:
        best_bits = _initial_bits()
    return _bits_to_list(best_bits)
