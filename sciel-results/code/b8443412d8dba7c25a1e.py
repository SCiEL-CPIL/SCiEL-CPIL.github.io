import time
import random
import sys
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 82.0  # safety margin before 90s kill


def _build_subset_trans(trans):
    """Precompute subset transitions for one letter using dynamic programming."""
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset


def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by
    trans_a and trans_b. Returns 0 if not synchronizing.
    Uses BFS on the subset automaton with precomputed transitions.
    """
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)

    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])

    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]

        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)

        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)

    return 0


def _hardcoded_best():
    """Return the standard Ping-Pong automaton for N=12."""
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    return a, b


def _structural_starts():
    """Generate diverse, structure-aware starting automata."""
    starts = []
    a_cycle = [(i + 1) % N for i in range(N)]
    a_rev = [(i - 1) % N for i in range(N)]

    # Ping-pong family (a cycle, b shifts down with variable sink)
    for base in (a_cycle, a_rev):
        for k in range(N):
            b = [k] + [i - 1 for i in range(1, N)]
            starts.append((base[:], b[:]))

    # Pinwheel family: a is a cycle, b is a different rotation
    for shift in range(1, N):
        b = [(i + shift) % N for i in range(N)]
        starts.append((a_cycle[:], b[:]))

    # Defect family: a is a cycle, b is a cycle with one defect
    for dp in range(N):
        for dd in range(N):
            b = [(i + 1) % N for i in range(N)]
            if dd != (dp + 1) % N:
                b[dp] = dd
                starts.append((a_cycle[:], b[:]))

    return starts


def _random_start():
    """Generate a random automaton biased toward synchronizability."""
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    if random.random() < 0.7:
        perm = list(range(N))
        random.shuffle(perm)
        a = perm[:]
    return a, b


def _mutate(cur_a, cur_b):
    """
    Apply one structure-aware mutation, returning (new_a, new_b).
    Chooses among targeted strategies plus random single-transition edits.
    """
    r = random.random()
    if r < 0.30:
        # cycle_defect: break a cycle edge by retargeting a state
        state = random.randint(0, N - 1)
        letter = random.choice([0, 1])
        arr = cur_a if letter == 0 else cur_b
        new_dest = random.randint(0, N - 1)
        if new_dest == arr[state]:
            new_dest = (new_dest + 1) % N
        if letter == 0:
            cur_a[state] = new_dest
        else:
            cur_b[state] = new_dest
    elif r < 0.55:
        # reset_adjust: retarget a random state in the "reset" letter
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        if new_dest == cur_b[state]:
            new_dest = (new_dest + 1) % N
        cur_b[state] = new_dest
    elif r < 0.75:
        # asymmetry: swap destinations of two states within one letter
        i, j = random.sample(range(N), 2)
        letter = random.choice([0, 1])
        if letter == 0:
            cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
        else:
            cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
    else:
        # random single-transition edit
        state = random.randint(0, N - 1)
        letter = random.choice([0, 1])
        new_dest = random.randint(0, N - 1)
        if letter == 0:
            cur_a[state] = new_dest
        else:
            cur_b[state] = new_dest


def _anneal(start_a, start_b, deadline):
    """
    Simulated annealing to maximize the reset threshold.
    Accepts worse solutions with temperature-controlled probability to escape
    local optima. Returns (best_a, best_b, best_score).
    """
    cur_a = start_a[:]
    cur_b = start_b[:]
    cur_score = compute_threshold(cur_a, cur_b)
    if cur_score == 0:
        return None, None, 0

    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score

    T0 = 12.0
    Tend = 0.02
    n_iter = 600
    # Geometric cooling schedule
    cool = (Tend / T0) ** (1.0 / n_iter) if T0 > 0 else 1.0

    for it in range(n_iter):
        if time.time() > deadline:
            break
        T = T0 * (cool ** it)

        new_a = cur_a[:]
        new_b = cur_b[:]
        _mutate(new_a, new_b)
        new_score = compute_threshold(new_a, new_b)

        if new_score > cur_score:
            cur_a, cur_b, cur_score = new_a, new_b, new_score
            if cur_score > best_score:
                best_score = cur_score
                best_a, best_b = cur_a[:], cur_b[:]
        else:
            # Accept worse with probability exp((new - cur)/T)
            if new_score == 0:
                # Never accept non-synchronizing
                pass
            else:
                delta = new_score - cur_score  # negative
                if random.random() < math.exp(delta / max(T, 1e-9)):
                    cur_a, cur_b, cur_score = new_a, new_b, new_score

    return best_a, best_b, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # 1) Hardcoded strong start (Ping-Pong)
    ha, hb = _hardcoded_best()
    hs = compute_threshold(ha, hb)
    if hs > best_score:
        best_score = hs
        best_a, best_b = ha[:], hb[:]

    # 2) Evaluate diverse structural starts, keep the best
    for sa, sb in _structural_starts():
        if time.time() > deadline - 25.0:
            break
        s = compute_threshold(sa, sb)
        if s > best_score:
            best_score = s
            best_a, best_b = sa[:], sb[:]

    # 3) Simulated annealing from the best structural start (the key structural
    #    change: temperature-controlled acceptance escapes the greedy local optima
    #    that capped the previous program at 71).
    if best_a is not None and time.time() < deadline - 15.0:
        ba, bb, bs = _anneal(best_a, best_b, deadline - 15.0)
        if bs > best_score:
            best_score = bs
            best_a, best_b = ba, bb

    # 4) A second anneal from a random synchronizing start for diversity,
    #    using any remaining time.
    if time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            ra, rb = _random_start()
            rs = compute_threshold(ra, rb)
            if rs == 0:
                continue
            if rs > best_score:
                best_score = rs
                best_a, best_b = ra[:], rb[:]
            ba, bb, bs = _anneal(ra, rb, deadline - 10.0)
            if bs > best_score:
                best_score = bs
                best_a, best_b = ba, bb
            if time.time() > deadline - 10.0:
                break

    # Ensure valid return
    if best_a is None or best_score == 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]

    return [best_a, best_b]
