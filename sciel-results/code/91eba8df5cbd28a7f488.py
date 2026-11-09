"""Rewritten program: randomized greedy with multiple restarts for two Sidon sets."""
import time
import random

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 55  # safety margin before 60s kill

    def greedy_random(seed=None):
        if seed is not None:
            random.seed(seed)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()  # positive differences
        diffs_B = set()  # positive differences
        all_diffs = set()  # union of both difference sets

        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        for x in candidates:
            # Try to add x to A
            new_diffs_A = set()
            valid = True
            for a in A:
                d = abs(x - a)
                if d == 0:
                    continue
                new_diffs_A.add(d)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                # Check Sidon property for A
                new_sums = set()
                for a in A:
                    s = x + a
                    if s in sums_A:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * x
                    if s in sums_A:
                        valid = False
                    else:
                        new_sums.add(s)
                if valid:
                    # Accept x into A
                    for a in A:
                        all_diffs.add(abs(x - a))
                    sums_A |= new_sums
                    diffs_A |= new_diffs_A
                    A.append(x)
                    continue
            # Try to add x to B
            new_diffs_B = set()
            valid = True
            for b in B:
                d = abs(x - b)
                if d == 0:
                    continue
                new_diffs_B.add(d)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                # Check Sidon property for B
                new_sums = set()
                for b in B:
                    s = x + b
                    if s in sums_B:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * x
                    if s in sums_B:
                        valid = False
                    else:
                        new_sums.add(s)
                if valid:
                    # Accept x into B
                    for b in B:
                        all_diffs.add(abs(x - b))
                    sums_B |= new_sums
                    diffs_B |= new_diffs_B
                    B.append(x)

        return A, B

    best_A = []
    best_B = []
    best_size = 0
    seed_counter = 0

    while time.time() < deadline:
        A, B = greedy_random(seed=seed_counter)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    return [best_A, best_B]
