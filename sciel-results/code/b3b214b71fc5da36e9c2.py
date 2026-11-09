"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 50.0  # seconds (Safe margin for 60s/90s limits)


def _pair_sums_ok(lst):
    """Check if all pair sums (including a+a) in lst are unique."""
    if len(lst) <= 1:
        return True
    sums = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i, len(lst)):
            b = lst[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs(lst):
    """Return set of nonzero differences b-a for a,b in lst, a != b."""
    d = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(len(lst)):
            if i != j:
                d.add(lst[j] - a)
    return d


def _can_add(lst, x, sums_used, diffs_used):
    """
    Check if x can be added to lst maintaining Sidon property
    (pair sums unique) and that new differences don't conflict with diffs_used.
    sums_used: set of all pair sums currently in lst
    diffs_used: set of all nonzero differences currently in lst
    Returns (new_sums, new_diffs) if valid, else (None, None).
    """
    # Check new pair sums
    # We can check against sums_used directly first
    for a in lst:
        if (a + x) in sums_used:
            return None, None
    if (x + x) in sums_used:
        return None, None
    
    # Check new differences
    for a in lst:
        d1 = x - a
        d2 = a - x
        if d1 in diffs_used or d2 in diffs_used:
            return None, None
        # Also check if the new differences conflict with each other? 
        # No, d1 = -d2. Since diffs_used contains both positive and negative diffs 
        # (because we add a-j and j-a), checking d1 and d2 against diffs_used is sufficient 
        # for conflicts with existing diffs.
        # However, we must ensure that adding d1 and d2 doesn't create a conflict 
        # between d1 and d2 themselves? No, they are just two new elements.
        # But wait, if we add d1, and later in the loop we encounter another a' such that 
        # x-a' is in the *original* diffs_used, we return.
        # The new diffs d1 and d2 are distinct (unless x==a, which is not possible as x not in lst).
        # So we just need to check against existing diffs.
        
    # If we pass all checks, we need to return the updated sets
    new_sums = set(sums_used)
    for a in lst:
        new_sums.add(a + x)
    new_sums.add(x + x)

    new_diffs = set(diffs_used)
    for a in lst:
        new_diffs.add(x - a)
        new_diffs.add(a - x)

    return new_sums, new_diffs


def _fast_greedy(t0, rng, seed_offset=0):
    """
    Fast greedy builder with randomization.
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()

    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # To add variation, we can bias the preference
    prefer_A = rng.random() < 0.5

    for x in candidates:
        if time.time() - t0 > DEADLINE - 2:
            break
        
        added = False
        # Try preferred side first
        if prefer_A:
            sides = [0, 1]
        else:
            sides = [1, 0]
            
        for side in sides:
            if side == 0:
                ns, nd = _can_add(A, x, sumsA, diffsA)
                if ns is not None:
                    # Check difference disjointness
                    if not (nd & diffsB):
                        A.append(x)
                        sumsA = ns
                        diffsA = nd
                        added = True
                        break
            else:
                ns, nd = _can_add(B, x, sumsB, diffsB)
                if ns is not None:
                    # Check difference disjointness
                    if not (nd & diffsA):
                        B.append(x)
                        sumsB = ns
                        diffsB = nd
                        added = True
                        break
        # If not added, skip x

    return A, B


def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    # Check Sidon property for A
    sumsA = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
    # Check Sidon property for B
    sumsB = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
    # Check difference disjointness
    diffsA = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diffsA.add(A[j] - A[i])
    diffsB = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                diffsB.add(B[j] - B[i])
    if diffsA & diffsB:
        return False
    # Check all elements in [1, N]
    for x in A:
        if x < 1 or x > N:
            return False
    for x in B:
        if x < 1 or x > N:
            return False
    return True


def construct():
    t0 = time.time()
    rng = random.Random(42)

    best_A, best_B = [], []
    best_size = 0

    # Run multiple fast greedy restarts
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() - t0 > DEADLINE - 2:
            break
        
        # Use a different seed for each restart to vary the shuffle
        r = random.Random(42 + i)
        A, B = _fast_greedy(t0, r)
        
        if not A or not B:
            continue
            
        # Simple local improvement: try to add remaining numbers
        # We can try to add any number from 1..N that wasn't picked
        used = set(A) | set(B)
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()
        
        # Recompute state for local improvement
        for x in A:
            for y in A:
                if x <= y:
                    sumsA.add(x+y)
                if x != y:
                    diffsA.add(y-x)
        for x in B:
            for y in B:
                if x <= y:
                    sumsB.add(x+y)
                if x != y:
                    diffsB.add(y-x)
                    
        # Try to add more
        for x in range(1, N+1):
            if time.time() - t0 > DEADLINE - 2:
                break
            if x in used:
                continue
            # Try A
            ns, nd = _can_add(A, x, sumsA, diffsA)
            if ns is not None and not (nd & diffsB):
                A.append(x)
                used.add(x)
                sumsA = ns
                diffsA = nd
                continue
            # Try B
            ns, nd = _can_add(B, x, sumsB, diffsB)
            if ns is not None and not (nd & diffsA):
                B.append(x)
                used.add(x)
                sumsB = ns
                diffsB = nd
                continue

        if _validate(A, B) and len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A, best_B = list(A), list(B)
            
            # If we hit the max possible, we can stop early? Max is 128.
            if best_size == 128:
                break

    # Ensure nonempty and valid
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    if not _validate(best_A, best_B):
        # Fallback
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
