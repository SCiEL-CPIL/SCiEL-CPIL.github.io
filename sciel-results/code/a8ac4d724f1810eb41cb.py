import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_total = 0
    best_A = []
    best_B = []

    for seed in range(200):
        if time.time() > deadline - 1.0:
            break

        random.seed(seed)

        # Greedy construction with bitsets for O(1) checks
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0

        order = list(range(1, N + 1))
        random.shuffle(order)

        for x in order:
            if time.time() > deadline - 1.0:
                break

            # Try adding to A
            can_add_A = True
            if (sums_A >> (x + x)) & 1:
                can_add_A = False
            else:
                for a in A:
                    if (sums_A >> (x + a)) & 1:
                        can_add_A = False
                        break
                    d = abs(x - a)
                    if (diffs_A >> d) & 1 or (diffs_B >> d) & 1:
                        can_add_A = False
                        break

            if can_add_A:
                A.append(x)
                sums_A |= (1 << (x + x))
                for a in A[:-1]:
                    sums_A |= (1 << (x + a))
                    d = abs(x - a)
                    diffs_A |= (1 << d)
            else:
                # Try adding to B
                can_add_B = True
                if (sums_B >> (x + x)) & 1:
                    can_add_B = False
                else:
                    for b in B:
                        if (sums_B >> (x + b)) & 1:
                            can_add_B = False
                            break
                        d = abs(x - b)
                        if (diffs_B >> d) & 1 or (diffs_A >> d) & 1:
                            can_add_B = False
                            break

                if can_add_B:
                    B.append(x)
                    sums_B |= (1 << (x + x))
                    for b in B[:-1]:
                        sums_B |= (1 << (x + b))
                        d = abs(x - b)
                        diffs_B |= (1 << d)

        total = len(A) + len(B)

        # Fast local search: try adding elements not yet in either set
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False

            # Try to add a new element to A
            for y in range(1, N + 1):
                if time.time() > deadline - 0.5:
                    break
                if y in A or y in B:
                    continue
                ok = True
                if (sums_A >> (y + y)) & 1:
                    ok = False
                else:
                    for a in A:
                        if (sums_A >> (y + a)) & 1:
                            ok = False
                            break
                        d = abs(y - a)
                        if (diffs_A >> d) & 1 or (diffs_B >> d) & 1:
                            ok = False
                            break
                if ok:
                    A.append(y)
                    sums_A |= (1 << (y + y))
                    for a in A[:-1]:
                        sums_A |= (1 << (y + a))
                        d = abs(y - a)
                        diffs_A |= (1 << d)
                    improved = True
                    break

            if improved:
                continue

            # Try to add a new element to B
            for y in range(1, N + 1):
                if time.time() > deadline - 0.5:
                    break
                if y in A or y in B:
                    continue
                ok = True
                if (sums_B >> (y + y)) & 1:
                    ok = False
                else:
                    for b in B:
                        if (sums_B >> (y + b)) & 1:
                            ok = False
                            break
                        d = abs(y - b)
                        if (diffs_B >> d) & 1 or (diffs_A >> d) & 1:
                            ok = False
                            break
                if ok:
                    B.append(y)
                    sums_B |= (1 << (y + y))
                    for b in B[:-1]:
                        sums_B |= (1 << (y + b))
                        d = abs(y - b)
                        diffs_B |= (1 << d)
                    improved = True
                    break

        if len(A) + len(B) > best_total:
            best_total = len(A) + len(B)
            best_A = A[:]
            best_B = B[:]

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
