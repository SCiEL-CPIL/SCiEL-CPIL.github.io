"""Rewritten program: structural change to a 3-base (base-3) digit-based construction,
with a local-search improvement pass under a strict wall-clock deadline."""
import time

# EVOLVE-BLOCK-START

N = 512
K = 4  # length of forbidden arithmetic progression


def has_4ap_in_set(S):
    """Check if set S contains a nonconstant 4-term arithmetic progression."""
    S_list = sorted(S)
    S_set = set(S_list)
    n = len(S_list)
    # For each pair (a, b) with a < b, check if a+2d, a+3d are in S where d = b-a
    # But we need 4-term: a, a+d, a+2d, a+3d all in S.
    # We can iterate over a and d.
    for a in S_list:
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if b in S_set and c in S_set and e in S_set:
                return True
    return False


def base3_no_zero_digits():
    """Construct a set using numbers whose base-3 representation has no digit 0."""
    S = set()
    for x in range(1, N + 1):
        n = x
        ok = True
        while n > 0:
            if n % 3 == 0:
                ok = False
                break
            n //= 3
        if ok:
            S.add(x)
    return S


def greedy_construct():
    """Greedy: add x if it doesn't create a 4-AP."""
    selected = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        # x can be the first, second, third, or fourth term
        # Case 1: x is first term: need x+d, x+2d, x+3d in selected
        conflict = False
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
                conflict = True
                break
        if not conflict:
            # Case 2: x is second term: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d in selected
            for d in range(1, x):
                if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
                    conflict = True
                    break
        if not conflict:
            # Case 3: x is third term: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d in selected
            for d in range(1, (x + 1) // 2):
                if (x - 2 * d) in selected and (x - d) in selected and (x + d) in selected:
                    conflict = True
                    break
        if not conflict:
            # Case 4: x is fourth term: x-3d, x-2d, x-d, x -> need x-3d, x-2d, x-d in selected
            for d in range(1, (x + 2) // 3 + 1):
                if (x - 3 * d) in selected and (x - 2 * d) in selected and (x - d) in selected:
                    conflict = True
                    break
        if not conflict:
            selected.add(x)
    return selected


def local_improve(S, deadline):
    """Try to add elements that don't create 4-APs, in random order, until deadline."""
    S = set(S)
    candidates = [x for x in range(1, N + 1) if x not in S]
    import random
    random.seed(42)
    random.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        # Check if adding x creates a 4-AP
        conflict = False
        # Case 1: x is first
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                conflict = True
                break
        if not conflict:
            # Case 2: x is second
            for d in range(1, x):
                if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                    conflict = True
                    break
        if not conflict:
            # Case 3: x is third
            for d in range(1, (x + 1) // 2):
                if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                    conflict = True
                    break
        if not conflict:
            # Case 4: x is fourth
            for d in range(1, (x + 2) // 3 + 1):
                if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                    conflict = True
                    break
        if not conflict:
            S.add(x)
    return S


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin: 50s out of 60s
    
    # Start with the base-3 no-zero-digit construction
    S = base3_no_zero_digits()
    
    # Try to improve with local search (add elements in random order)
    if time.time() < deadline:
        S = local_improve(S, deadline)
    
    # Verify no 4-AP
    if has_4ap_in_set(S):
        # Fallback: use greedy
        S = greedy_construct()
    
    return sorted(S)

# EVOLVE-BLOCK-END
