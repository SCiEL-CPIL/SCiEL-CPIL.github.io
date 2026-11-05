"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word."""
import time
import random
from collections import deque

N = 12

def _threshold(a, b):
    """BFS on subset automaton to find shortest reset word length. Returns -1 if not synchronizing."""
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    visited = {start: 0}
    q = deque([start])
    while q:
        curr = q.popleft()
        d = visited[curr]
        if len(curr) == 1:
            return d
        na = frozenset(a[i] for i in curr)
        if na not in visited:
            visited[na] = d + 1
            q.append(na)
        nb = frozenset(b[i] for i in curr)
        if nb not in visited:
            visited[nb] = d + 1
            q.append(nb)
    return -1

def _is_synchronizing(a, b):
    """Quick check if automaton synchronizes. Returns threshold or -1."""
    return _threshold(a, b)

def construct():
    """Construct a 12-state binary deterministic synchronizing automaton with long reset word."""
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin below 60s wall-clock limit
    
    best_a = None
    best_b = None
    best_thr = 0
    
    # Known good construction: "pinwheel" or "Černý" type
    # a is a cycle, b maps most states to i+1 but creates a slow contraction
    # 
    # The best known for N=12 should be around 55 (from (N-1)(N-2)/2)
    # But the actual worst case might be different. Let's use a known construction.
    
    # Strategy 1: Use a known long-reset-word construction
    # For N states, one family: a = (0 1 2 ... N-1), b maps i -> i+1 for i < N-1, b(N-1) -> 0
    # But that's just a permutation, doesn't synchronize.
    
    # Better: a = cycle, b = identity except one state maps elsewhere
    # This gives threshold related to the distance in the cycle.
    
    # Let's try multiple constructions:
    
    # Construction A: a = cycle (0->1->2->...->11->0)
    # b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]  (last two map to 10)
    a_cycle = [(i + 1) % N for i in range(N)]
    
    b_mono = list(range(N-1)) + [N-2]  # b = [0,1,2,...,10,10]
    thr = _threshold(a_cycle, b_mono)
    if thr > 0 and thr > best_thr:
        best_a, best_b, best_thr = a_cycle[:], b_mono[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction B: a = cycle, b = identity except b(0) = 1
    # This should give a long reset word.
    b_id = list(range(N))
    b_id[0] = 1  # b(0) = 1 instead of 0
    thr = _threshold(a_cycle, b_id)
    if thr > 0 and thr > best_thr:
        best_a, best_b, best_thr = a_cycle[:], b_id[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction C: a = cycle, b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # (b(0)=0, b(1)=0, b(i)=i-1 for i>=2)
    b_shift = [0, 0] + list(range(1, N-1))
    thr = _threshold(a_cycle, b_shift)
    if thr > 0 and thr > best_thr:
        best_a, best_b, best_thr = a_cycle[:], b_shift[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction D: Try all "one defect" perturbations of the monotone b
    # a = cycle, b = [0,1,2,...,10,10] with one element changed
    base_b = list(range(N-1)) + [N-2]
    for i in range(N):
        for new_val in range(N):
            if new_val == base_b[i]:
                continue
            if time.time() > deadline:
                break
            b_test = base_b[:]
            b_test[i] = new_val
            thr = _threshold(a_cycle, b_test)
            if thr > 0 and thr > best_thr:
                best_a, best_b, best_thr = a_cycle[:], b_test[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction E: Random search with a = cycle
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > deadline:
            break
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        thr = _threshold(a_cycle, b_rand)
        if thr > 0 and thr > best_thr:
            best_a, best_b, best_thr = a_cycle[:], b_rand[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction F: Random permutations for a, random b
    for trial in range(100):
        if time.time() > deadline:
            break
        a_perm = list(range(N))
        rng.shuffle(a_perm)
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        thr = _threshold(a_perm, b_rand)
        if thr > 0 and thr > best_thr:
            best_a, best_b, best_thr = a_perm[:], b_rand[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction G: a = cycle, b = [0,1,2,...,9,10,10] with a = cycle in different direction
    a_rev = [(i - 1) % N for i in range(N)]
    b_mono2 = list(range(N-1)) + [N-2]
    thr = _threshold(a_rev, b_mono2)
    if thr > 0 and thr > best_thr:
        best_a, best_b, best_thr = a_rev[:], b_mono2[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction H: Try a "bottleneck" construction
    # a = cycle, b maps most states forward but one state maps to a "sink"
    for sink in range(N):
        if time.time() > deadline:
            break
        b_sink = [(i + 1) % N for i in range(N)]
        b_sink[sink] = sink  # one state is fixed
        thr = _threshold(a_cycle, b_sink)
        if thr > 0 and thr > best_thr:
            best_a, best_b, best_thr = a_cycle[:], b_sink[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction I: Two-step bottleneck
    # a = cycle, b maps two adjacent states to the same value
    for i in range(N):
        if time.time() > deadline:
            break
        b_two = [(i + 1) % N for i in range(N)]
        b_two[i] = (i + 2) % N  # two states map to same
        thr = _threshold(a_cycle, b_two)
        if thr > 0 and thr > best_thr:
            best_a, best_b, best_thr = a_cycle[:], b_two[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction J: Known "Černý" construction for N=12
    # a = (0 1 2 ... 11)
    # b: 0->0, 1->1, 2->2, ..., 10->10, 11->10
    # This is the same as Construction A.
    
    # Construction K: a = cycle, b = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b_three = [0, 0, 0] + list(range(1, N-1))
    thr = _threshold(a_cycle, b_three)
    if thr > 0 and thr > best_thr:
        best_a, best_b, best_thr = a_cycle[:], b_three[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Construction L: a = cycle, b = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_dup = [0, 1, 1] + list(range(2, N-1))
    thr = _threshold(a_cycle, b_dup)
    if thr > 0 and thr > best_thr:
        best_a, best_b, best_thr = a_cycle[:], b_dup[:], thr
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Fallback: ensure we have a valid synchronizing automaton
    if best_a is None or best_b is None or best_thr <= 0:
        # Simple construction that definitely synchronizes
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N  # all states go to 0
        best_thr = _threshold(best_a, best_b)
    
    return best_a, best_b
