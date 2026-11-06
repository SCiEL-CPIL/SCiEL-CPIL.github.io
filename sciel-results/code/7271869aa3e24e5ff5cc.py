import random
import time

# EVOLVE-BLOCK-START

def _score_set(s):
    # s: frozenset of ints in 1..512
    mask = 0
    for x in s:
        mask |= (1 << x)
    count = 0
    for d in range(1, 257):
        # need a, a+d, a+2d all in set, a in 1..512-2d
        m = mask
        # shift by d: bit i set iff (i+d) in set
        m2 = mask >> d
        m4 = mask >> (2 * d)
        # a in set AND a+d in set AND a+2d in set  <=>  bit a set in mask & (mask>>d) & (mask>>2d)
        combined = m & m2 & m4
        if combined:
            count += 1
    return count

def _swap(s, i, j):
    # swap elements at positions i and j in the list representation
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return frozenset(lst)

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin under 150s
    rng = random.Random(12345)

    best = None
    best_score = -1

    # initial: 1..32
    start = frozenset(range(1, 33))
    best = start
    best_score = _score_set(start)

    def anneal(start_set, t_start, t_end):
        nonlocal best, best_score
        cur = start_set
        cur_score = _score_set(cur)
        cur_lst = sorted(cur)
        # temperature schedule
        T = 1.0
        T_end = 0.02
        # We'll do steps until time budget
        while time.time() < t_end:
            i = rng.randrange(32)
            j = rng.randrange(32)
            if i == j:
                continue
            # swap
            a = cur_lst[i]
            b = cur_lst[j]
            cur_lst[i] = b
            cur_lst[j] = a
            new_set = frozenset(cur_lst)
            new_score = _score_set(new_set)
            delta = new_score - cur_score
            if delta > 0 or (T > 0 and rng.random() < (2.718281828459045 ** (delta / max(T, 1e-9)))):
                cur = new_set
                cur_score = new_score
            else:
                # revert
                cur_lst[i] = a
                cur_lst[j] = b
            # cool
            T = max(T_end, T * 0.999)
            if cur_score > best_score:
                best_score = cur_score
                best = cur

    # Run annealing in chunks with restarts, tracking deadline
    restart = 0
    while time.time() < DEADLINE:
        # each chunk uses a portion of remaining time
        remaining = DEADLINE - time.time()
        chunk = min(remaining, 6.0)
        if chunk < 0.5:
            break
        t_start = time.time()
        t_end = t_start + chunk
        # pick a starting set: best found so far, or a random one
        if restart == 0:
            start_set = best
        elif rng.random() < 0.5:
            start_set = best
        else:
            # random 32-subset of 1..512
            start_set = frozenset(rng.sample(range(1, 513), 32))
        anneal(start_set, t_start, t_end)
        restart += 1

    return sorted(best)

# EVOLVE-BLOCK-END
