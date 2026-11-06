"""Two Sidon sets with disjoint difference sets – bitset local search."""

import time
import random

N = 256
MAX_EL = 64
DEADLINE = 55.0  # safety margin before 90s kill

def _sum_bitset(arr):
    """Return a bitmask where bit s is set if s can be written as a+b (a<=b) from arr."""
    bs = 0
    for i in range(len(arr)):
        ai = arr[i]
        for j in range(i, len(arr)):
            bs |= 1 << (ai + arr[j])
    return bs

def _diff_bitset(arr):
    """Return a bitmask where bit d is set if d = |a-b| (a!=b) occurs in arr."""
    bs = 0
    for i in range(len(arr)):
        ai = arr[i]
        for j in range(len(arr)):
            if i != j:
                d = abs(ai - arr[j])
                bs |= 1 << d
    return bs

def _valid(A, B, sA, sB, dA, dB):
    """Check: A and B are Sidon (sums unique) and nonzero differences disjoint."""
    # Sidon: sum bitset popcount == number of pair-sums with repetition
    nA, nB = len(A), len(B)
    if sA.bit_count() != nA * (nA + 1) // 2:
        return False
    if sB.bit_count() != nB * (nB + 1) // 2:
        return False
    # Differences disjoint
    if dA & dB:
        return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Greedy initialization: build A and B alternately
    A = []
    B = []
    sA = 0
    sB = 0
    dA = 0
    dB = 0

    # Precompute for speed: for a candidate x and current set S, compute new sum/diff contributions
    # We'll just do the greedy loop with bitsets
    for x in range(1, N + 1):
        if time.time() > deadline - 10:
            break
        # Try adding x to A
        if len(A) < MAX_EL:
            # Check Sidon for A + x
            ok = True
            for a in A:
                if (1 << (a + x)) & sA:
                    ok = False
                    break
            if ok and (1 << (2 * x)) & sA:
                ok = False
            # Check differences
            if ok:
                for a in A:
                    d = abs(a - x)
                    if (1 << d) & dA:
                        ok = False
                        break
                    if (1 << d) & dB:
                        ok = False
                        break
            if ok:
                A.append(x)
                sA |= 1 << (2 * x)
                for a in A[:-1]:
                    sA |= 1 << (a + x)
                    d = abs(a - x)
                    dA |= 1 << d
        # Try adding x to B
        if len(B) < MAX_EL:
            ok = True
            for b in B:
                if (1 << (b + x)) & sB:
                    ok = False
                    break
            if ok and (1 << (2 * x)) & sB:
                ok = False
            if ok:
                for b in B:
                    d = abs(b - x)
                    if (1 << d) & dB:
                        ok = False
                        break
                    if (1 << d) & dA:
                        ok = False
                        break
            if ok:
                B.append(x)
                sB |= 1 << (2 * x)
                for b in B[:-1]:
                    sB |= 1 << (b + x)
                    d = abs(b - x)
                    dB |= 1 << d

    best_total = len(A) + len(B)
    best_A, best_B = list(A), list(B)

    # Local search: try to add elements, then try swaps to free up space
    rng = random.Random(42)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)

    # Rebuild bitsets for current state
    sA = _sum_bitset(A)
    sB = _sum_bitset(B)
    dA = _diff_bitset(A)
    dB = _diff_bitset(B)

    while time.time() < deadline:
        # Phase 1: try to add a candidate to A or B
        added = False
        for x in candidates:
            if time.time() > deadline:
                break
            # Try add to A
            if len(A) < MAX_EL and x not in A:
                # Check Sidon
                ok = True
                for a in A:
                    if (1 << (a + x)) & sA:
                        ok = False
                        break
                if ok and (1 << (2 * x)) & sA:
                    ok = False
                if ok:
                    for a in A:
                        d = abs(a - x)
                        if (1 << d) & dA or (1 << d) & dB:
                            ok = False
                            break
                if ok:
                    A.append(x)
                    sA |= 1 << (2 * x)
                    for a in A[:-1]:
                        sA |= 1 << (a + x)
                        dA |= 1 << abs(a - x)
                    added = True
                    if len(A) + len(B) > best_total:
                        best_total = len(A) + len(B)
                        best_A, best_B = list(A), list(B)
                    break

            # Try add to B
            if len(B) < MAX_EL and x not in B:
                ok = True
                for b in B:
                    if (1 << (b + x)) & sB:
                        ok = False
                        break
                if ok and (1 << (2 * x)) & sB:
                    ok = False
                if ok:
                    for b in B:
                        d = abs(b - x)
                        if (1 << d) & dB or (1 << d) & dA:
                            ok = False
                            break
                if ok:
                    B.append(x)
                    sB |= 1 << (2 * x)
                    for b in B[:-1]:
                        sB |= 1 << (b + x)
                        dB |= 1 << abs(b - x)
                    added = True
                    if len(A) + len(B) > best_total:
                        best_total = len(A) + len(B)
                        best_A, best_B = list(A), list(B)
                    break

        if not added:
            # Phase 2: try swaps – remove an element from A (or B), then add a new one
            # Try swapping in A
            swap_done = False
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                old = A[i]
                # Tentatively remove old from A
                A2 = A[:i] + A[i+1:]
                sA2 = _sum_bitset(A2)
                dA2 = _diff_bitset(A2)
                # Try to add a new element
                for x in candidates:
                    if x in A2:
                        continue
                    ok = True
                    for a in A2:
                        if (1 << (a + x)) & sA2:
                            ok = False
                            break
                    if ok and (1 << (2 * x)) & sA2:
                        ok = False
                    if ok:
                        for a in A2:
                            d = abs(a - x)
                            if (1 << d) & dA2 or (1 << d) & dB:
                                ok = False
                                break
                    if ok:
                        # Do the swap
                        A = A2 + [x]
                        sA = sA2 | (1 << (2 * x))
                        for a in A2:
                            sA |= 1 << (a + x)
                            dA |= 1 << abs(a - x)
                        dA = dA2
                        swap_done = True
                        break
                if swap_done:
                    break

            if not swap_done:
                # Try swapping in B
                for i in range(len(B)):
                    if time.time() > deadline:
                        break
                    old = B[i]
                    B2 = B[:i] + B[i+1:]
                    sB2 = _sum_bitset(B2)
                    dB2 = _diff_bitset(B2)
                    for x in candidates:
                        if x in B2:
                            continue
                        ok = True
                        for b in B2:
                            if (1 << (b + x)) & sB2:
                                ok = False
                                break
                        if ok and (1 << (2 * x)) & sB2:
                            ok = False
                        if ok:
                            for b in B2:
                                d = abs(b - x)
                                if (1 << d) & dB2 or (1 << d) & dA:
                                    ok = False
                                    break
                        if ok:
                            B = B2 + [x]
                            sB = sB2 | (1 << (2 * x))
                            for b in B2:
                                sB |= 1 << (b + x)
                                dB |= 1 << abs(b - x)
                            dB = dB2
                            swap_done = True
                            break
                    if swap_done:
                        break

            if not swap_done:
                # No improvement possible with current candidates; break
                break

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
