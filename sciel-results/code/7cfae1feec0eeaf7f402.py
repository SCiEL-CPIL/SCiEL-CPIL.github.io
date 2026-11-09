import time
import random
import numpy as np

N = 512
DEADLINE_MARGIN = 5  # seconds before the 60s limit to stop work

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_MARGIN

    def elapsed_ok():
        return (time.time() - t0) < deadline

    # --- Fast greedy: iterate 1..N, add x if it doesn't create a 4-AP ---
    # We maintain a boolean array `in_set` of size N+1.
    # To check if adding x creates a 4-AP, we check all d where the 4-AP
    # involving x stays in [1, N].
    
    def check_add(x, in_set):
        # x as 1st term: x, x+d, x+2d, x+3d in set
        #   need x+d, x+2d, x+3d all in set
        # x as 2nd term: x-d, x, x+d, x+2d
        #   need x-d, x+d, x+2d all in set
        # x as 3rd term: x-2d, x-d, x, x+d
        #   need x-2d, x-d, x+d all in set
        # x as 4th term: x-3d, x-2d, x-d, x
        #   need x-3d, x-2d, x-d all in set
        
        # As 1st: d from 1 to (N-x)//3
        for d in range(1, (N - x) // 3 + 1):
            if in_set[x + d] and in_set[x + 2*d] and in_set[x + 3*d]:
                return False
        # As 2nd: d from 1 to min(x-1, (N-x)//2)
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if in_set[x - d] and in_set[x + d] and in_set[x + 2*d]:
                return False
        # As 3rd: d from 1 to min((x-1)//2, N-x)
        for d in range(1, min((x - 1) // 2, N - x) + 1):
            if in_set[x - 2*d] and in_set[x - d] and in_set[x + d]:
                return False
        # As 4th: d from 1 to (x-1)//3
        for d in range(1, (x - 1) // 3 + 1):
            if in_set[x - 3*d] and in_set[x - 2*d] and in_set[x - d]:
                return False
        return True

    # Run multiple greedy passes with different orderings to get diverse sets
    best_size = 0
    best_set = set()

    # Pass 1: standard ascending order
    in_set = np.zeros(N + 1, dtype=bool)
    for x in range(1, N + 1):
        if not elapsed_ok():
            break
        if check_add(x, in_set):
            in_set[x] = True
    s = {x for x in range(1, N + 1) if in_set[x]}
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    if not elapsed_ok():
        return sorted(best_set)

    # Pass 2: descending order
    in_set = np.zeros(N + 1, dtype=bool)
    for x in range(N, 0, -1):
        if not elapsed_ok():
            break
        if check_add(x, in_set):
            in_set[x] = True
    s = {x for x in range(1, N + 1) if in_set[x]}
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    if not elapsed_ok():
        return sorted(best_set)

    # Pass 3: random shuffles
    rng = random.Random(42)
    for trial in range(20):
        if not elapsed_ok():
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        in_set = np.zeros(N + 1, dtype=bool)
        for x in order:
            if check_add(x, in_set):
                in_set[x] = True
        s = {x for x in range(1, N + 1) if in_set[x]}
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if not elapsed_ok():
        return sorted(best_set)

    # Local search: try to swap out elements to add more
    # For each element in the set, try removing it and see if we can add 2+ new elements
    improved = True
    while improved and elapsed_ok():
        improved = False
        S = set(best_set)
        # Try removing each element
        for x in list(S):
            if not elapsed_ok():
                break
            S.remove(x)
            in_set = np.zeros(N + 1, dtype=bool)
            for y in S:
                in_set[y] = True
            # Try adding all elements not in S in ascending order
            added = 0
            for z in range(1, N + 1):
                if z in S:
                    continue
                if check_add(z, in_set):
                    S.add(z)
                    in_set[z] = True
                    added += 1
            if len(S) > len(best_set):
                best_set = S
                best_size = len(S)
                improved = True
                break

    return sorted(best_set)
