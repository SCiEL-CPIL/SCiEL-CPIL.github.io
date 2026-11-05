"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word."""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _apply_mask(a, mask):
    """Apply transformation a to bitmask, return new bitmask."""
    result = 0
    m = mask
    while m:
        low = m & (-m)
        i = low.bit_length() - 1
        result |= (1 << a[i])
        m ^= low
    return result

def _build_tables(a, b):
    """Precompute transition tables for both letters on all subsets (bitmasks)."""
    # We only need to compute for masks that are reachable, but for speed we can
    # precompute for all 2^N masks. That's 4096 entries, very fast.
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    for mask in range(1 << N):
        trans_a[mask] = _apply_mask(a, mask)
        trans_b[mask] = _apply_mask(b, mask)
    return trans_a, trans_b

def _threshold_fast(a, b):
    """BFS on subset automaton using precomputed transition tables. Returns -1 if not synchronizing."""
    trans_a, trans_b = _build_tables(a, b)
    start = FULL_MASK
    if start & (start - 1) == 0:
        return 0
    visited = [-1] * (1 << N)
    visited[start] = 0
    q = deque([start])
    while q:
        curr = q.popleft()
        d = visited[curr]
        if curr & (curr - 1) == 0:
            return d
        na = trans_a[curr]
        if visited[na] == -1:
            visited[na] = d + 1
            q.append(na)
        nb = trans_b[curr]
        if visited[nb] == -1:
            visited[nb] = d + 1
            q.append(nb)
    return -1

def _threshold(a, b):
    """BFS on subset automaton. Returns -1 if not synchronizing."""
    return _threshold_fast(a, b)

def construct():
    """Construct a 12-state binary deterministic synchronizing automaton with long reset word."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin below 90s wall-clock limit

    best_a = None
    best_b = None
    best_thr = 0

    def try_automaton(a, b):
        nonlocal best_a, best_b, best_thr
        thr = _threshold_fast(a, b)
        if thr > 0 and thr > best_thr:
            best_a, best_b, best_thr = a[:], b[:], thr

    # Construction A: a = cycle, b = [0,1,2,...,10,10] (pinwheel)
    a_cycle = [(i + 1) % N for i in range(N)]
    a_rev = [(i - 1) % N for i in range(N)]

    b_mono = list(range(N-1)) + [N-2]
    try_automaton(a_cycle, b_mono)
    try_automaton(a_rev, b_mono)

    if time.time() > deadline:
        return best_a, best_b

    # Construction B: a = cycle, b = identity with one defect
    for defect_pos in range(N):
        for defect_target in range(N):
            if defect_target == defect_pos:
                continue
            b_id = list(range(N))
            b_id[defect_pos] = defect_target
            try_automaton(a_cycle, b_id)
            try_automaton(a_rev, b_id)
            if time.time() > deadline:
                return best_a, best_b

    if time.time() > deadline:
        return best_a, best_b

    # Construction C: Systematic two-defect search on identity base
    for p1 in range(N):
        for p2 in range(p1+1, N):
            for t1 in range(N):
                if t1 == p1:
                    continue
                for t2 in range(N):
                    if t2 == p2:
                        continue
                    if time.time() > deadline:
                        return best_a, best_b
                    b_two_def = list(range(N))
                    b_two_def[p1] = t1
                    b_two_def[p2] = t2
                    try_automaton(a_cycle, b_two_def)

    if time.time() > deadline:
        return best_a, best_b

    # Construction D: One-defect perturbations of monotone base
    base_b = list(range(N-1)) + [N-2]
    for i in range(N):
        for new_val in range(N):
            if new_val == base_b[i]:
                continue
            if time.time() > deadline:
                return best_a, best_b
            b_test = base_b[:]
            b_test[i] = new_val
            try_automaton(a_cycle, b_test)
            try_automaton(a_rev, b_test)

    if time.time() > deadline:
        return best_a, best_b

    # Construction E: Structured "shift with gap" family
    for k in range(1, N):
        for defect_pos in range(N):
            for defect_target in range(N):
                if time.time() > deadline:
                    return best_a, best_b
                b_shift = [((i - k) % N) for i in range(N)]
                b_shift[defect_pos] = defect_target
                try_automaton(a_cycle, b_shift)

    if time.time() > deadline:
        return best_a, best_b

    # Construction F: Random search with a = cycle (more trials)
    rng = random.Random(12345)
    for trial in range(2000):
        if time.time() > deadline:
            break
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        try_automaton(a_cycle, b_rand)

    if time.time() > deadline:
        return best_a, best_b

    # Construction G: Random permutations for a, random b
    for trial in range(1000):
        if time.time() > deadline:
            break
        a_perm = list(range(N))
        rng.shuffle(a_perm)
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        try_automaton(a_perm, b_rand)

    if time.time() > deadline:
        return best_a, best_b

    # Construction H: Local improvement - mutate best found
    if best_a is not None and best_b is not None and best_thr > 0:
        for _ in range(2000):
            if time.time() > deadline:
                break
            b_mut = best_b[:]
            pos = rng.randint(0, N-1)
            b_mut[pos] = rng.randint(0, N-1)
            try_automaton(best_a, b_mut)

    if time.time() > deadline:
        return best_a, best_b

    # Construction I: Simulated annealing-style search on both a and b
    if best_a is not None and best_b is not None and best_thr > 0:
        cur_a = best_a[:]
        cur_b = best_b[:]
        cur_thr = best_thr
        temp = 50.0
        for step in range(5000):
            if time.time() > deadline:
                break
            # Mutate either a or b
            if rng.random() < 0.5:
                pos = rng.randint(0, N-1)
                old_val = cur_a[pos]
                cur_a[pos] = rng.randint(0, N-1)
                while cur_a[pos] == old_val:
                    cur_a[pos] = rng.randint(0, N-1)
            else:
                pos = rng.randint(0, N-1)
                old_val = cur_b[pos]
                cur_b[pos] = rng.randint(0, N-1)
                while cur_b[pos] == old_val:
                    cur_b[pos] = rng.randint(0, N-1)
            
            thr = _threshold_fast(cur_a, cur_b)
            if thr > 0:
                if thr >= cur_thr or rng.random() < pow(2.718281828, (thr - cur_thr) / max(temp, 1)):
                    cur_thr = thr
                    if thr > best_thr:
                        best_a, best_b, best_thr = cur_a[:], cur_b[:], thr
                else:
                    # Revert
                    if rng.random() < 0.5:
                        cur_a[pos] = old_val
                    else:
                        cur_b[pos] = old_val
            else:
                # Revert
                if rng.random() < 0.5:
                    cur_a[pos] = old_val
                else:
                    cur_b[pos] = old_val
            
            temp *= 0.999

    if time.time() > deadline:
        return best_a, best_b

    # Fallback: ensure we have a valid synchronizing automaton
    if best_a is None or best_b is None or best_thr <= 0:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_thr = _threshold_fast(best_a, best_b)

    return best_a, best_b
