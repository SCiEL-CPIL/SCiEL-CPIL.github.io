"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word."""
import time
from collections import deque

N = 12
_DEADLINE = 80.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _build_transition_table(a, b):
    """Precompute transition for all 2^N subsets under letters a and b."""
    size = 1 << N
    trans_a = [0] * size
    trans_b = [0] * size
    for mask in range(size):
        na = 0
        nb = 0
        for i in range(N):
            if mask & (1 << i):
                na |= (1 << a[i])
                nb |= (1 << b[i])
        trans_a[mask] = na
        trans_b[mask] = nb
    return trans_a, trans_b


def _threshold_fast(a, b):
    """BFS on subset automaton using precomputed transition tables.
    Returns -1 if not synchronizing."""
    start_mask = (1 << N) - 1
    if start_mask & (start_mask - 1) == 0:
        return 0

    trans_a, trans_b = _build_transition_table(a, b)

    visited = [-1] * (1 << N)
    visited[start_mask] = 0
    q = deque([start_mask])

    while q:
        cur = q.popleft()
        d = visited[cur]
        if cur & (cur - 1) == 0:
            return d

        next_a = trans_a[cur]
        if visited[next_a] == -1:
            visited[next_a] = d + 1
            q.append(next_a)

        next_b = trans_b[cur]
        if visited[next_b] == -1:
            visited[next_b] = d + 1
            q.append(next_b)

    return -1


def _threshold(a, b):
    return _threshold_fast(a, b)


def _steepest_ascent(a, b, max_steps=200):
    """Systematic steepest-ascent hill climbing: try all single-transition changes,
    pick the best improvement each step. Returns (best_a, best_b, best_thresh)."""
    cur_a = a[:]
    cur_b = b[:]
    cur_thr = _threshold(cur_a, cur_b)
    if cur_thr < 0:
        return cur_a, cur_b, cur_thr

    for _ in range(max_steps):
        if _elapsed() > _DEADLINE:
            break

        best_improvement = cur_thr
        best_new_a = None
        best_new_b = None

        # Try all changes to a[i]
        for i in range(N):
            for new_val in range(N):
                if new_val == cur_a[i]:
                    continue
                test_a = cur_a[:]
                test_a[i] = new_val
                thr = _threshold(test_a, cur_b)
                if thr > best_improvement:
                    best_improvement = thr
                    best_new_a = test_a
                    best_new_b = cur_b[:]

        # Try all changes to b[i]
        for i in range(N):
            for new_val in range(N):
                if new_val == cur_b[i]:
                    continue
                test_b = cur_b[:]
                test_b[i] = new_val
                thr = _threshold(cur_a, test_b)
                if thr > best_improvement:
                    best_improvement = thr
                    best_new_a = cur_a[:]
                    best_new_b = test_b

        if best_new_a is None:
            # No improvement found — local optimum
            break

        cur_a = best_new_a
        cur_b = best_new_b
        cur_thr = best_improvement

    return cur_a, cur_b, cur_thr


def construct():
    """Return two length-N lists of destination states in 0..N-1.

    Must define a complete binary deterministic automaton that synchronizes.
    Score is the exact shortest reset-word length, maximized.
    """
    # Primary: Černý automaton construction — known to achieve (N-1)^2 = 121
    # a = cycle: 0->1->2->...->11->0
    # b = identity except b(N-1) = N-2 (merges last two states)
    a_cerny = [(i + 1) % N for i in range(N)]
    b_cerny = list(range(N))
    b_cerny[N - 1] = N - 2  # b(11) = 10

    thr_cerny = _threshold(a_cerny, b_cerny)
    best_a = a_cerny[:]
    best_b = b_cerny[:]
    best_thresh = thr_cerny

    # Verify: if Černý construction gives the theoretical max, we're done
    theoretical_max = (N - 1) ** 2  # 121
    if best_thresh >= theoretical_max:
        return best_a, best_b

    # If not at theoretical max, attempt steepest ascent from the Černý starting point
    if _elapsed() < _DEADLINE - 10:
        sa_a, sa_b, sa_thr = _steepest_ascent(a_cerny, b_cerny, max_steps=50)
        if sa_thr > best_thresh:
            best_a = sa_a
            best_b = sa_b
            best_thresh = sa_thr

    # Additional structured candidates as fallback/verification
    candidates = []

    # Variant: b maps N-1 to 0 instead of N-2
    b_var1 = list(range(N))
    b_var1[N - 1] = 0
    candidates.append((a_cerny[:], b_var1[:]))

    # Variant: a = reverse cycle
    a_rev = [(i - 1) % N for i in range(N)]
    candidates.append((a_rev[:], b_cerny[:]))
    candidates.append((a_rev[:], b_var1[:]))

    # Variant: b = [0,1,...,9,10,10] (same as b_cerny for N=12)
    # Variant: b = [0,0,1,2,...,10]
    b_shift = [0, 0] + list(range(1, N - 1))
    candidates.append((a_cerny[:], b_shift[:]))

    # Evaluate all candidates
    for ca, cb in candidates:
        if _elapsed() > _DEADLINE - 5:
            break
        thr = _threshold(ca, cb)
        if thr > best_thresh:
            best_thresh = thr
            best_a = ca[:]
            best_b = cb[:]

    # Final steepest ascent from the overall best
    if _elapsed() < _DEADLINE - 5 and best_thresh < theoretical_max:
        sa_a, sa_b, sa_thr = _steepest_ascent(best_a, best_b, max_steps=30)
        if sa_thr > best_thresh:
            best_a = sa_a
            best_b = sa_b
            best_thresh = sa_thr

    # Safety fallback
    if best_a is None or best_thresh < 0:
        best_a = a_cerny
        best_b = b_cerny
        best_thresh = _threshold(best_a, best_b)

    return best_a, best_b
