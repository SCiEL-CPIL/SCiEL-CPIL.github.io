"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # seconds, safety margin for 60s/90s limits


def _compute_states(A, B):
    """Compute sums and diffs for A and B."""
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()

    for i in range(len(A)):
        a = A[i]
        sums_A.add(2 * a)
        for j in range(i + 1, len(A)):
            b = A[j]
            sums_A.add(a + b)
            diffs_A.add(b - a)
            diffs_A.add(a - b)

    for i in range(len(B)):
        b = B[i]
        sums_B.add(2 * b)
        for j in range(i + 1, len(B)):
            c = B[j]
            sums_B.add(b + c)
            diffs_B.add(c - b)
            diffs_B.add(b - c)

    return sums_A, sums_B, diffs_A, diffs_B


def _forbidden_mask_A(A, sums_A, diffs_B):
    """Return a set of x in [1,N] that CANNOT be added to A.

    x is forbidden if:
      - 2*x in sums_A, or
      - exists a in A with (a+x) in sums_A, or
      - exists a in A with (x-a) in diffs_B or (a-x) in diffs_B.
    """
    forbidden = set()
    for a in A:
        # x such that a + x in sums_A  =>  x = s - a for s in sums_A
        for s in sums_A:
            x = s - a
            if 1 <= x <= N:
                forbidden.add(x)
        # x such that 2*x in sums_A  =>  x = s/2
        # handled below globally
        # x such that x - a in diffs_B  =>  x = d + a
        for d in diffs_B:
            x = d + a
            if 1 <= x <= N:
                forbidden.add(x)
            x = a - d
            if 1 <= x <= N:
                forbidden.add(x)
    # 2*x in sums_A
    for s in sums_A:
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N:
                forbidden.add(x)
    return forbidden


def _forbidden_mask_B(B, sums_B, diffs_A):
    forbidden = set()
    for b in B:
        for s in sums_B:
            x = s - b
            if 1 <= x <= N:
                forbidden.add(x)
        for d in diffs_A:
            x = d + b
            if 1 <= x <= N:
                forbidden.add(x)
            x = b - d
            if 1 <= x <= N:
                forbidden.add(x)
    for s in sums_B:
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N:
                forbidden.add(x)
    return forbidden


def _add_to_A(x, A, sums_A, diffs_A):
    A.append(x)
    for a in A:
        if a != x:
            sums_A.add(a + x)
            diffs_A.add(x - a)
            diffs_A.add(a - x)
    sums_A.add(2 * x)


def _add_to_B(x, B, sums_B, diffs_B):
    B.append(x)
    for b in B:
        if b != x:
            sums_B.add(b + x)
            diffs_B.add(x - b)
            diffs_B.add(b - x)
    sums_B.add(2 * x)


def _greedy_fill(order, A, B, sums_A, sums_B, diffs_A, diffs_B, deadline):
    """Greedily fill A and B from order using forbidden masks."""
    start = time.time()
    forb_A = _forbidden_mask_A(A, sums_A, diffs_B)
    forb_B = _forbidden_mask_B(B, sums_B, diffs_A)

    for x in order:
        if time.time() - start > deadline:
            break
        # Try A
        if x not in forb_A and x not in A:
            _add_to_A(x, A, sums_A, diffs_A)
            forb_A = _forbidden_mask_A(A, sums_A, diffs_B)
            forb_B = _forbidden_mask_B(B, sums_B, diffs_A)
            continue
        # Try B
        if x not in forb_B and x not in B:
            _add_to_B(x, B, sums_B, diffs_B)
            forb_A = _forbidden_mask_A(A, sums_A, diffs_B)
            forb_B = _forbidden_mask_B(B, sums_B, diffs_A)
            continue

    return A, B, sums_A, sums_B, diffs_A, diffs_B


def _polish(A, B, sums_A, sums_B, diffs_A, diffs_B, deadline):
    """Final high-precision polish: try to insert every unused element in
    sorted order into the smaller set first, then the other."""
    start = time.time()
    used = set(A) | set(B)
    candidates = [x for x in range(1, N + 1) if x not in used]

    for x in candidates:
        if time.time() - start > deadline:
            break
        # Try the smaller set first to balance
        if len(A) <= len(B):
            first, second = 'A', 'B'
        else:
            first, second = 'B', 'A'

        if first == 'A':
            if x not in A:
                forb_A = _forbidden_mask_A(A, sums_A, diffs_B)
                if x not in forb_A:
                    _add_to_A(x, A, sums_A, diffs_A)
                    continue
        if second == 'B':
            if x not in B:
                forb_B = _forbidden_mask_B(B, sums_B, diffs_A)
                if x not in forb_B:
                    _add_to_B(x, B, sums_B, diffs_B)
                    continue

    return A, B, sums_A, sums_B, diffs_A, diffs_B


def construct():
    start = time.time()
    best_A = []
    best_B = []
    best_score = 0
    rng = random.Random(42)

    # Initial Greedy
    order = list(range(1, N + 1))
    rng.shuffle(order)
    A, B = [], []
    sums_A, sums_B, diffs_A, diffs_B = set(), set(), set(), set()

    A, B, sums_A, sums_B, diffs_A, diffs_B = _greedy_fill(
        order, A, B, sums_A, sums_B, diffs_A, diffs_B, DEADLINE - (time.time() - start)
    )

    # Polish pass
    A, B, sums_A, sums_B, diffs_A, diffs_B = _polish(
        A, B, sums_A, sums_B, diffs_A, diffs_B, DEADLINE - (time.time() - start)
    )

    best_score = len(A) + len(B)
    best_A = A[:]
    best_B = B[:]

    # Iterated Local Search
    while time.time() - start < DEADLINE - 3.0:
        # Perturb: remove k random elements
        k = rng.randint(1, 5)
        A_curr = best_A[:]
        B_curr = best_B[:]

        for _ in range(k):
            if A_curr:
                idx = rng.randrange(len(A_curr))
                A_curr.pop(idx)
            elif B_curr:
                idx = rng.randrange(len(B_curr))
                B_curr.pop(idx)
            else:
                break

        sums_A, sums_B, diffs_A, diffs_B = _compute_states(A_curr, B_curr)

        used = set(A_curr) | set(B_curr)
        candidates = [x for x in range(1, N + 1) if x not in used]
        rng.shuffle(candidates)

        A_new, B_new, _, _, _, _ = _greedy_fill(
            candidates, A_curr, B_curr, sums_A, sums_B, diffs_A, diffs_B,
            DEADLINE - (time.time() - start)
        )

        # Polish each candidate solution
        A_new, B_new, sA, sB, dA, dB = _polish(
            A_new, B_new,
            _compute_states(A_new, B_new)[0], _compute_states(A_new, B_new)[1],
            _compute_states(A_new, B_new)[2], _compute_states(A_new, B_new)[3],
            DEADLINE - (time.time() - start)
        )

        score = len(A_new) + len(B_new)
        if score > best_score:
            best_score = score
            best_A = A_new[:]
            best_B = B_new[:]
            if best_score >= 2 * MAX_EACH:
                break

    # Final polish on best
    sA, sB, dA, dB = _compute_states(best_A, best_B)
    best_A, best_B, _, _, _, _ = _polish(
        best_A, best_B, sA, sB, dA, dB, DEADLINE - (time.time() - start)
    )

    # Ensure nonempty
    if not best_A and not best_B:
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
