"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit

    n = 256
    max_size = 64

    def validate(A, B):
        """Validate that A and B are valid Sidon sets with disjoint nonzero diffs."""
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                val = A[i] + A[j]
                if val in sums_A:
                    return False
                sums_A.add(val)
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                val = B[i] + B[j]
                if val in sums_B:
                    return False
                sums_B.add(val)
        # Check disjoint nonzero differences
        dA = set()
        for a in A:
            for b in A:
                if a != b:
                    dA.add(a - b)
        dB = set()
        for a in B:
            for b in B:
                if a != b:
                    dB.add(a - b)
        if dA & dB:
            return False
        return True

    def fast_greedy(order, rng):
        """
        Fast greedy that builds A and B incrementally with O(1) amortized checks.
        Returns (A, B).
        """
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()  # all nonzero diffs a-b for a,b in A
        diffs_B = set()  # all nonzero diffs a-b for a,b in B

        for x in order:
            placed = False

            # Try adding x to A
            ok = True
            if (2 * x) in sums_A:
                ok = False
            else:
                for a in A:
                    if (x + a) in sums_A:
                        ok = False
                        break
            if ok:
                for a in A:
                    d1 = x - a
                    d2 = a - x
                    if d1 in diffs_A or d1 in diffs_B or d2 in diffs_A or d2 in diffs_B:
                        ok = False
                        break
            if ok:
                A.append(x)
                sums_A.add(2 * x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
                placed = True

            if not placed:
                # Try adding x to B
                ok = True
                if (2 * x) in sums_B:
                    ok = False
                else:
                    for b in B:
                        if (x + b) in sums_B:
                            ok = False
                            break
                if ok:
                    for b in B:
                        d1 = x - b
                        d2 = b - x
                        if d1 in diffs_A or d1 in diffs_B or d2 in diffs_A or d2 in diffs_B:
                            ok = False
                            break
                if ok:
                    B.append(x)
                    sums_B.add(2 * x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    placed = True

        return A, B

    def local_search(A, B, rng, max_iters=200):
        """
        Local search: try moving elements between A and B, or adding unplaced elements.
        Returns improved (A, B).
        """
        A = list(A)
        B = list(B)
        current_total = len(A) + len(B)

        # Precompute current state
        def compute_state(A, B):
            sums_A = set()
            for i in range(len(A)):
                for j in range(i, len(A)):
                    sums_A.add(A[i] + A[j])
            sums_B = set()
            for i in range(len(B)):
                for j in range(i, len(B)):
                    sums_B.add(B[i] + B[j])
            dA = set()
            for a in A:
                for b in A:
                    if a != b:
                        dA.add(a - b)
            dB = set()
            for a in B:
                for b in B:
                    if a != b:
                        dB.add(a - b)
            return sums_A, sums_B, dA, dB

        # Try adding unplaced elements
        placed = set(A) | set(B)
        unplaced = [x for x in range(1, n + 1) if x not in placed]
        rng.shuffle(unplaced)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Try adding an unplaced element to A or B
            for x in unplaced:
                if time.time() > deadline:
                    break
                # Try A
                ok = True
                if (2 * x) in sums_A:
                    ok = False
                else:
                    for a in A:
                        if (x + a) in sums_A:
                            ok = False
                            break
                if ok:
                    for a in A:
                        d1 = x - a
                        d2 = a - x
                        if d1 in dA or d1 in dB or d2 in dA or d2 in dB:
                            ok = False
                            break
                if ok:
                    A.append(x)
                    sums_A.add(2 * x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        dA.add(x - a)
                        dA.add(a - x)
                    unplaced.remove(x)
                    current_total += 1
                    improved = True
                    break

                # Try B
                ok = True
                if (2 * x) in sums_B:
                    ok = False
                else:
                    for b in B:
                        if (x + b) in sums_B:
                            ok = False
                            break
                if ok:
                    for b in B:
                        d1 = x - b
                        d2 = b - x
                        if d1 in dA or d1 in dB or d2 in dA or d2 in dB:
                            ok = False
                            break
                if ok:
                    B.append(x)
                    sums_B.add(2 * x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        dB.add(x - b)
                        dB.add(b - x)
                    unplaced.remove(x)
                    current_total += 1
                    improved = True
                    break

            if not improved:
                # Try moving an element from A to B or B to A
                moved = False
                if A:
                    idx = rng.randrange(len(A))
                    x = A[idx]
                    # Remove x from A
                    A.pop(idx)
                    sums_A = set()
                    for i in range(len(A)):
                        for j in range(i, len(A)):
                            sums_A.add(A[i] + A[j])
                    dA = set()
                    for a in A:
                        for b in A:
                            if a != b:
                                dA.add(a - b)
                    # Try adding x to B
                    ok = True
                    if (2 * x) in sums_B:
                        ok = False
                    else:
                        for b in B:
                            if (x + b) in sums_B:
                                ok = False
                                break
                    if ok:
                        for b in B:
                            d1 = x - b
                            d2 = b - x
                            if d1 in dA or d1 in dB or d2 in dA or d2 in dB:
                                ok = False
                                break
                    if ok:
                        B.append(x)
                        sums_B.add(2 * x)
                        for b in B[:-1]:
                            sums_B.add(x + b)
                            dB.add(x - b)
                            dB.add(b - x)
                        moved = True
                    if not moved:
                        # Put x back in A
                        A.append(x)
                        sums_A = set()
                        for i in range(len(A)):
                            for j in range(i, len(A)):
                                sums_A.add(A[i] + A[j])
                        dA = set()
                        for a in A:
                            for b in A:
                                if a != b:
                                    dA.add(a - b)
                if not moved and B:
                    idx = rng.randrange(len(B))
                    x = B[idx]
                    B.pop(idx)
                    sums_B = set()
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            sums_B.add(B[i] + B[j])
                    dB = set()
                    for a in B:
                        for b in B:
                            if a != b:
                                dB.add(a - b)
                    # Try adding x to A
                    ok = True
                    if (2 * x) in sums_A:
                        ok = False
                    else:
                        for a in A:
                            if (x + a) in sums_A:
                                ok = False
                                break
                    if ok:
                        for a in A:
                            d1 = x - a
                            d2 = a - x
                            if d1 in dA or d1 in dB or d2 in dA or d2 in dB:
                                ok = False
                                break
                    if ok:
                        A.append(x)
                        sums_A.add(2 * x)
                        for a in A[:-1]:
                            sums_A.add(x + a)
                            dA.add(x - a)
                            dA.add(a - x)
                        moved = True
                    if not moved:
                        B.append(x)
                        sums_B = set()
                        for i in range(len(B)):
                            for j in range(i, len(B)):
                                sums_B.add(B[i] + B[j])
                        dB = set()
                        for a in B:
                            for b in B:
                                if a != b:
                                    dB.add(a - b)

                if not improved and not moved:
                    break

        return A, B

    best_total = 0
    best_result = ([1], [2])  # fallback valid result

    rng = random.Random(42)

    # Phase 1: Multiple greedy restarts with different orderings
    for restart in range(300):
        if time.time() > deadline:
            break

        order = list(range(1, n + 1))
        rng.shuffle(order)

        A, B = fast_greedy(order, rng)

        if A and B:
            total = len(A) + len(B)
            if total > best_total:
                if validate(A, B):
                    best_total = total
                    best_result = (A, B)
                    if best_total >= 30:
                        break

    # Phase 2: Local search on best result
    if best_total > 0 and time.time() < deadline:
        A, B = local_search(list(best_result[0]), list(best_result[1]), rng, max_iters=500)
        if A and B:
            total = len(A) + len(B)
            if total > best_total:
                if validate(A, B):
                    best_total = total
                    best_result = (A, B)

    # Phase 3: More restarts with local search
    for restart in range(150):
        if time.time() > deadline:
            break

        order = list(range(1, n + 1))
        rng.shuffle(order)

        A, B = fast_greedy(order, rng)

        if A and B:
            total = len(A) + len(B)
            if total >= best_total:
                A2, B2 = local_search(A, B, rng, max_iters=300)
                total2 = len(A2) + len(B2)
                if total2 > best_total:
                    if validate(A2, B2):
                        best_total = total2
                        best_result = (A2, B2)
                        if best_total >= 32:
                            break

    return list(best_result[0]), list(best_result[1])
# EVOLVE-BLOCK-END
