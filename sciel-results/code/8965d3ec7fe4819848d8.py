import time
from itertools import product

# EVOLVE-BLOCK-START

def _ternary_digits(x):
    """Return list of ternary digits of x (least significant first)."""
    if x == 0:
        return [0]
    digits = []
    while x > 0:
        digits.append(x % 3)
        x //= 3
    return digits

def _base3_restricted_set(n):
    """
    Construct a 4-AP-free set using base-3 digit restriction.
    Select numbers in [1, n] whose ternary representation contains no digit '2'.
    This is known to be 3-AP-free. For 4-AP-free, we verify and may need to
    filter further.
    """
    selected = set()
    for x in range(1, n + 1):
        digits = _ternary_digits(x)
        if all(d != 2 for d in digits):
            selected.add(x)
    return selected

def _has_4ap_in_set(S):
    """Check if set S contains a nonconstant 4-term arithmetic progression."""
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        # Check if there exists d > 0 such that a, a+d, a+2d, a+3d are all in S
        # Maximum d such that a+3d <= max(S)
        max_d = (max(S) - a) // 3 if max(S) >= a else 0
        for d in range(1, max_d + 1):
            if (a + d in S) and (a + 2*d in S) and (a + 3*d in S):
                return True
    return False

def _remove_4aps_from_set(S, n):
    """
    Remove elements from S to break all 4-term APs.
    Greedily remove the element involved in the most APs.
    """
    S = set(S)
    changed = True
    while changed:
        changed = False
        # Find all 4-APs
        ap_to_remove = None
        max_count = 0
        # For each element, count how many 4-APs it's part of
        count = {x: 0 for x in S}
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            max_d = (max(S) - a) // 3 if max(S) >= a else 0
            for d in range(1, max_d + 1):
                if (a + d in S) and (a + 2*d in S) and (a + 3*d in S):
                    for elem in (a, a+d, a+2*d, a+3*d):
                        count[elem] += 1
                    if count[a + 2*d] > max_count:
                        max_count = count[a + 2*d]
                        ap_to_remove = a + 2*d
        if ap_to_remove is not None:
            S.remove(ap_to_remove)
            changed = True
    return S

def _two_exchange_improve(S, n, time_limit=55.0):
    """
    Try 2-exchange local search: remove two elements, add two others to break APs
    and increase size.
    """
    start_time = time.time()
    S = set(S)
    best_size = len(S)
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        S_list = sorted(S)
        # Try removing each pair of elements
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if (time.time() - start_time) >= time_limit:
                    return S
                S_trial = set(S)
                S_trial.discard(S_list[i])
                S_trial.discard(S_list[j])
                # Try adding two elements not in S_trial
                candidates = [x for x in range(1, n + 1) if x not in S_trial]
                # Sort candidates by "desirability" - prefer those that don't create APs
                # For efficiency, just try a few
                for k in range(len(candidates)):
                    if (time.time() - start_time) >= time_limit:
                        return S
                    c1 = candidates[k]
                    S_trial2 = set(S_trial)
                    S_trial2.add(c1)
                    # Check if S_trial2 has no 4-AP
                    # Quick check: verify no new AP involving c1
                    valid_c1 = True
                    for d in range(1, (n - c1) // 3 + 1):
                        if (c1 + d in S_trial2) and (c1 + 2*d in S_trial2) and (c1 + 3*d in S_trial2):
                            valid_c1 = False
                            break
                        if (c1 - d in S_trial2) and (c1 - 2*d in S_trial2) and (c1 - 3*d in S_trial2):
                            valid_c1 = False
                            break
                    if not valid_c1:
                        continue
                    # Try adding a second element
                    for l in range(k + 1, len(candidates)):
                        if (time.time() - start_time) >= time_limit:
                            return S
                        c2 = candidates[l]
                        if c2 in S_trial2:
                            continue
                        S_trial3 = set(S_trial2)
                        S_trial3.add(c2)
                        # Check c2 doesn't create AP
                        valid_c2 = True
                        for d in range(1, (n - c2) // 3 + 1):
                            if (c2 + d in S_trial3) and (c2 + 2*d in S_trial3) and (c2 + 3*d in S_trial3):
                                valid_c2 = False
                                break
                            if (c2 - d in S_trial3) and (c2 - 2*d in S_trial3) and (c2 - 3*d in S_trial3):
                                valid_c2 = False
                                break
                        if not valid_c2:
                            continue
                        # Final verification
                        if not _has_4ap_in_set(S_trial3):
                            if len(S_trial3) > len(S):
                                S = S_trial3
                                best_size = len(S)
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
    return S

def _verify_4ap_free(S, n):
    """Verify S contains no nonconstant 4-term arithmetic progression."""
    S_list = sorted(S)
    for i in range(len(S_list)):
        a = S_list[i]
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            if (a + d in S) and (a + 2*d in S) and (a + 3*d in S):
                return False
    return True

def construct():
    n = 512
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit

    # Step 1: Base-3 restricted set (no digit '2' in ternary)
    S = _base3_restricted_set(n)

    # Step 2: Remove elements to break any 4-APs
    S = _remove_4aps_from_set(S, n)

    # Step 3: Try to add more elements
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for x in range(1, n + 1):
            if (time.time() - start_time) >= deadline:
                break
            if x in S:
                continue
            S_trial = set(S)
            S_trial.add(x)
            # Check if adding x creates a 4-AP
            valid = True
            # Check all 4-APs that could include x
            # x could be at position 0, 1, 2, or 3 in the AP
            # Position 0: x, x+d, x+2d, x+3d
            for d in range(1, (n - x) // 3 + 1):
                if (x + d in S_trial) and (x + 2*d in S_trial) and (x + 3*d in S_trial):
                    valid = False
                    break
            if valid:
                # Position 1: x-3d, x-2d, x-d, x
                for d in range(1, x // 3 + 1):
                    if (x - 3*d in S_trial) and (x - 2*d in S_trial) and (x - d in S_trial):
                        valid = False
                        break
            if valid:
                # Position 2: x-2d, x-d, x, x+d
                for d in range(1, min(x, n - x) // 1 + 1):
                    if d * 2 > x or x + d > n:
                        continue
                    if (x - 2*d in S_trial) and (x - d in S_trial) and (x + d in S_trial):
                        valid = False
                        break
            if valid:
                # Position 3: x-3d, x-2d, x-d, x
                for d in range(1, x // 3 + 1):
                    if (x - 3*d in S_trial) and (x - 2*d in S_trial) and (x - d in S_trial):
                        valid = False
                        break
            if valid:
                S = S_trial
                improved = True

    # Step 4: 2-exchange local search if time permits
    if (time.time() - start_time) < deadline - 10:
        S = _two_exchange_improve(S, n, time_limit=deadline - (time.time() - start_time) - 5)

    # Final verification
    if not _verify_4ap_free(S, n):
        # Fallback: use greedy construction
        S_fallback = set()
        for x in range(1, n + 1):
            if all(not all(x - j*d in S_fallback for j in (1, 2, 3)) for d in range(1, (x - 1) // 3 + 1)):
                S_fallback.add(x)
        S = S_fallback

    return sorted(S)

# EVOLVE-BLOCK-END
