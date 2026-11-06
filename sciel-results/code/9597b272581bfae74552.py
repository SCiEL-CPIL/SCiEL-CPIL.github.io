"""Two Sidon sets with disjoint difference sets – Bitset-based fast search.
Uses integer bitmasks for sum and difference collisions to enable high-throughput
local search within the time budget."""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 55.0  # Safety margin before 60s limit

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # State: lists of elements, and bitmasks for sums and diffs
    # Max sum is 256+256=512. Max diff is 255.
    # We use Python ints as bitsets.
    
    A = []
    B = []
    sums_A = 0
    sums_B = 0
    diffs_A = 0
    diffs_B = 0
    
    def rebuild_A():
        nonlocal sums_A, diffs_A
        sums_A = 0
        diffs_A = 0
        for i in range(len(A)):
            a1 = A[i]
            sums_A |= (1 << (a1 + a1))
            for j in range(i + 1, len(A)):
                a2 = A[j]
                s = a1 + a2
                d = abs(a1 - a2)
                sums_A |= (1 << s)
                diffs_A |= (1 << d)

    def rebuild_B():
        nonlocal sums_B, diffs_B
        sums_B = 0
        diffs_B = 0
        for i in range(len(B)):
            b1 = B[i]
            sums_B |= (1 << (b1 + b1))
            for j in range(i + 1, len(B)):
                b2 = B[j]
                s = b1 + b2
                d = abs(b1 - b2)
                sums_B |= (1 << s)
                diffs_B |= (1 << d)

    def can_add_A(x):
        # Check if x can be added to A
        # 1. Check x+x against sums_A
        if (sums_A >> (x + x)) & 1:
            return False
        # 2. Check x+a against sums_A and diffs against diffs_A and diffs_B
        for a in A:
            s = x + a
            d = abs(x - a)
            if (sums_A >> s) & 1:
                return False
            if (diffs_A >> d) & 1:
                return False
            if (diffs_B >> d) & 1:
                return False
        return True

    def add_A(x):
        nonlocal sums_A, diffs_A
        sums_A |= (1 << (x + x))
        for a in A:
            s = x + a
            d = abs(x - a)
            sums_A |= (1 << s)
            diffs_A |= (1 << d)
        A.append(x)

    def can_add_B(x):
        # Check if x can be added to B
        if (sums_B >> (x + x)) & 1:
            return False
        for b in B:
            s = x + b
            d = abs(x - b)
            if (sums_B >> s) & 1:
                return False
            if (diffs_B >> d) & 1:
                return False
            if (diffs_A >> d) & 1:
                return False
        return True

    def add_B(x):
        nonlocal sums_B, diffs_B
        sums_B |= (1 << (x + x))
        for b in B:
            s = x + b
            d = abs(x - b)
            sums_B |= (1 << s)
            diffs_B |= (1 << d)
        B.append(x)

    def remove_A(x):
        nonlocal sums_A, diffs_A
        A.remove(x)
        rebuild_A()

    def remove_B(x):
        nonlocal sums_B, diffs_B
        B.remove(x)
        rebuild_B()

    # Phase 1: Greedy
    for x in range(1, N + 1):
        if time.time() > deadline - 5.0:
            break
        if len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
        elif len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)

    # Phase 2: Local Search
    rng = random.Random(42)
    best_score = len(A) + len(B)
    best_A = list(A)
    best_B = list(B)
    
    while time.time() < deadline - 2.0:
        # 1. Try to add a random element
        if len(A) < MAX_SIZE or len(B) < MAX_SIZE:
            for _ in range(10):
                x = rng.randint(1, N)
                if x in A or x in B:
                    continue
                if len(A) < MAX_SIZE and can_add_A(x):
                    add_A(x)
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    break
                elif len(B) < MAX_SIZE and can_add_B(x):
                    add_B(x)
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    break

        # 2. Swap in A
        if A:
            idx = rng.randint(0, len(A) - 1)
            a_old = A[idx]
            for _ in range(20):
                x = rng.randint(1, N)
                if x in A or x in B:
                    continue
                # Remove old, try add new
                remove_A(a_old)
                if can_add_A(x):
                    add_A(x)
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    break
                else:
                    # Restore old
                    add_A(a_old)
                    break

        # 3. Swap in B
        if B:
            idx = rng.randint(0, len(B) - 1)
            b_old = B[idx]
            for _ in range(20):
                x = rng.randint(1, N)
                if x in A or x in B:
                    continue
                # Remove old, try add new
                remove_B(b_old)
                if can_add_B(x):
                    add_B(x)
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    break
                else:
                    # Restore old
                    add_B(b_old)
                    break

    # Restore best
    A = best_A
    B = best_B
    
    if not A: A = [1]
    if not B: B = [2]
    
    return [A, B]
