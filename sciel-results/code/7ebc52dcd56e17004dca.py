import time
import random
import bisect

def _is_valid_candidate(candidate):
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _greedy_build(order):
    selected = []
    for x in order:
        candidate = selected + [x]
        candidate.sort()
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected


def _try_insert(candidate, x):
    """Try to insert x into candidate and return the new candidate if valid."""
    n = len(candidate)
    for i in range(n + 1):
        new_candidate = candidate[:i] + [x] + candidate[i:]
        if _is_valid_candidate(new_candidate):
            return new_candidate
    return None


def _local_search(candidate, unused, deadline, rng):
    """Improve candidate by trying to insert unused elements."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(unused)):
            if time.time() >= deadline:
                break
            x = unused[i]
            new_candidate = _try_insert(candidate, x)
            if new_candidate is not None:
                candidate = new_candidate
                unused = unused[:i] + unused[i+1:]
                improved = True
    return candidate, unused


def construct():
    start_time = time.time()
    # Safety margin: 55s limit, 60s hard limit, 90s kill
    deadline = start_time + 52.0

    # 1. Deterministic Baselines
    best_seq = []

    # Ascending
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq

    # Descending
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq

    # 2. Random Restarts with Local Search
    rng = random.Random(42)
    base_range = list(range(1, 257))

    while time.time() < deadline:
        rng.shuffle(base_range)
        seq = _greedy_build(base_range)
        
        # Try local search to improve
        unused = [x for x in base_range if x not in seq]
        seq, _ = _local_search(seq, unused, deadline, rng)
        
        if len(seq) > len(best_seq):
            best_seq = seq

    return best_seq
