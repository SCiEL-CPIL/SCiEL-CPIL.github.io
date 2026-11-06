import time
import random

def _valid(S):
    """Check the property for sorted set S. Returns True if valid."""
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            b = lst[j]
            for k in range(j, n):
                c = lst[k]
                if (b + c) % a == 0:
                    return False
    return True


def _conflict(a, S):
    """Check if adding a to S creates a conflict. S must not contain a."""
    lst = sorted(S)
    m = len(lst)
    for j in range(m):
        b = lst[j]
        for k in range(j, m):
            c = lst[k]
            if (b + c) % a == 0:
                return True
    return False


def _greedy_add(S, candidates, deadline):
    """Try to add each candidate in order if conflict-free."""
    S = set(S)
    for a in candidates:
        if time.time() > deadline:
            break
        if a not in S and not _conflict(a, S):
            S.add(a)
    return S


def _local_search(S, deadline, max_iter=200):
    """Local search: try to swap elements to increase set size."""
    S = set(S)
    best = len(S)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        # Try adding a new element
        for a in range(1, 257):
            if a not in S and not _conflict(a, S):
                S.add(a)
                improved = True
                break
        if improved:
            best = len(S)
            continue
        # Try removing an element and adding a different one
        for a in list(S):
            if time.time() > deadline:
                break
            S.remove(a)
            for b in range(1, 257):
                if b not in S and not _conflict(b, S):
                    S.add(b)
                    if len(S) >= best:
                        improved = True
                    break
            if not improved:
                S.remove(b)
                S.add(a)
            if time.time() > deadline:
                break
        if not improved:
            break
    return S


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin
    n = 256

    # Base: [171, 256]
    best_set = set(range(171, n + 1))
    best_size = len(best_set)

    # Strategy 1: Descending greedy from 170 down to 129
    S = set(range(171, n + 1))
    S = _greedy_add(S, range(170, 128, -1), deadline)
    if _valid(S) and len(S) > best_size:
        best_set = set(S)
        best_size = len(S)

    # Strategy 2: Ascending greedy from 129 to 170
    if time.time() < deadline:
        S = set(range(171, n + 1))
        S = _greedy_add(S, range(129, 171), deadline)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Strategy 3: Different base thresholds
    for base in [165, 168, 172, 175, 180, 185]:
        if time.time() > deadline:
            break
        S = set(range(base, n + 1))
        below = list(range(base - 1, 128, -1))
        S = _greedy_add(S, below, deadline)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Strategy 4: Randomized greedy orders
    random.seed(42)
    for trial in range(10):
        if time.time() > deadline:
            break
        S = set(range(171, n + 1))
        candidates = list(range(129, 171))
        random.shuffle(candidates)
        S = _greedy_add(S, candidates, deadline)
        if _valid(S) and len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Local search on best found
    if time.time() < deadline:
        improved = _local_search(set(best_set), deadline, max_iter=150)
        if _valid(improved) and len(improved) > best_size:
            best_set = improved
            best_size = len(improved)

    # Final verification
    if not _valid(best_set):
        best_set = set(range(171, n + 1))

    return list(best_set)
