"""Rewritten program: diameter-bounded DFS with bitmask pruning for Sidon set containing 0,
covering differences 1..30, minimizing the largest element."""
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s wall-clock limit

    # We need a Sidon set A containing 0, |A| <= 32,
    # all positive differences distinct, and every d in 1..30 appears as a difference.
    # Minimize max(A).

    # Key insight: if A has n elements, it has n*(n-1)/2 positive differences.
    # We need at least 30 differences, so n >= 8 (since 7*6/2=21 < 30, 8*7/2=28 < 30,
    # 9*8/2=36 >= 30). So minimum size is 9.
    # For a Sidon set of size n in [0, M], we need M >= roughly n^2 (since all n*(n-1)/2
    # differences must be distinct and <= M).

    # Strategy: DFS over sorted elements a_0=0 < a_1 < ... < a_{k-1} <= M,
    # tracking which differences 1..30 are covered and ensuring no difference repeats.
    # We search for the smallest M such that a valid set exists.

    # Use bitmask for covered differences (bits 0..29 for d=1..30)
    # and a set for all differences used (to enforce Sidon property).

    best_result = None
    best_M = 10**6 + 1

    # Lower bound on M: with n elements, need n*(n-1)/2 distinct differences in [1, M],
    # so M >= n*(n-1)/2. For n=9, M >= 36. But we also need specific differences 1..30.
    # Actually the minimum possible M for n=9 is at least 36 (pigeonhole on differences).
    # Let's start searching from small M upward.

    def check_time():
        return time.time() - start_time < DEADLINE

    # For a given maximum M, try to find a Sidon set of size 9 (or more if needed)
    # containing 0, with all differences 1..30 covered.
    # Use DFS: pick elements in increasing order.

    def solve_for_M(M, max_size=32):
        """Find a Sidon set in [0, M] containing 0 covering diffs 1..30."""
        # State: current elements (sorted), set of all positive differences, bitmask of 1..30 covered
        # We'll use iterative deepening on the number of elements, starting from 9.

        for n in range(9, max_size + 1):
            if not check_time():
                return None

            result = _dfs(0, n, M, [0], set(), 0)
            if result is not None:
                return result
        return None

    def _dfs(pos, target_size, M, current, diff_set, covered_mask):
        """
        pos: index of next element to choose
        target_size: total number of elements needed
        M: maximum value
        current: list of chosen elements (sorted, current[0]=0)
        diff_set: set of all positive differences so far
        covered_mask: bitmask of which of 1..30 are covered
        """
        if not check_time():
            return None

        # If we have enough elements and cover all 30 differences, check validity
        if len(current) == target_size:
            if covered_mask == (1 << 30) - 1:
                return list(current)
            return None

        # Pruning: even if we add all remaining elements, can we cover enough?
        remaining_slots = target_size - len(current)
        # Each new element paired with each existing element gives differences
        # Max additional differences: sum over remaining slots of (len(current) + slot_index)
        # Actually: adding one element x to a set of size k gives k new differences.
        # So adding r elements gives at most k + (k+1) + ... + (k+r-1) new differences.
        # But we only care about covering 1..30, not total count.

        # Pruning: how many more differences from 1..30 do we still need?
        missing = 0
        for d in range(1, 31):
            if not (covered_mask >> (d-1)) & 1:
                missing += 1

        if missing == 0:
            # All covered, just fill remaining slots with valid elements
            # Actually we need exactly target_size elements
            if remaining_slots == 0:
                return list(current)
            # Try to add elements without creating duplicate differences
            return _fill_remaining(pos, target_size, M, current, diff_set, covered_mask)

        # Lower bound: with remaining_slots elements to add, starting from current size k,
        # max new differences involving 1..30 is bounded.
        # Simple prune: if remaining_slots * len(current) < missing, impossible
        # (each new element can create at most len(current) new differences with existing elements,
        #  but some may be outside 1..30 or duplicates)
        if remaining_slots * len(current) < missing:
            return None

        # Determine range for next element
        # Must be > current[-1] and <= M
        last = current[-1]
        # Also, the next element must be at least last+1
        # Upper bound: M
        # But we can tighten: the element must create at least one new difference in 1..30
        # that isn't already covered, OR it's the last element and we already have all covered.

        # Candidate range: from last+1 to M
        # Optimization: try candidates that are likely to cover missing differences
        # For each missing d, a candidate x must satisfy x - a_i = d for some a_i in current,
        # or a_i - x = d (but x > a_i since sorted, so only x - a_i = d matters if x > a_i)
        # Actually x - a_i = d means x = a_i + d. So candidates that cover a missing d
        # are of the form a_i + d for a_i in current, d in missing differences.

        # Build candidate set: all a_i + d for a_i in current, d in 1..30 where d is missing
        # Plus a few extra candidates for flexibility
        candidates = set()
        for a_i in current:
            for d in range(1, 31):
                if not (covered_mask >> (d-1)) & 1:
                    x = a_i + d
                    if last < x <= M:
                        candidates.add(x)
        # Also add some nearby values for flexibility
        for delta in range(1, 15):
            x = last + delta
            if x <= M:
                candidates.add(x)
        # Sort candidates
        candidates = sorted(candidates)

        # Limit candidates to avoid explosion
        # Prioritize candidates that cover more missing differences
        def candidate_score(x):
            """Score = number of new missing differences this candidate would cover."""
            score = 0
            for a_i in current:
                d = x - a_i
                if 1 <= d <= 30 and not (covered_mask >> (d-1)) & 1:
                    score += 1
            return -score  # Negate for min-heap behavior (we want max score first)

        candidates.sort(key=candidate_score)

        # Limit to top candidates to control search space
        max_candidates = 50
        candidates = candidates[:max_candidates]

        for x in candidates:
            if not check_time():
                return None

            # Check Sidon property: no duplicate differences
            valid = True
            new_diffs = []
            for a_i in current:
                d = x - a_i
                if d in diff_set:
                    valid = False
                    break
                new_diffs.append(d)
            if not valid:
                continue

            # Update state
            new_diff_set = diff_set | set(new_diffs)
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))

            result = _dfs(pos + 1, target_size, M, current + [x], new_diff_set, new_mask)
            if result is not None:
                return result

        return None

    def _fill_remaining(pos, target_size, M, current, diff_set, covered_mask):
        """Fill remaining slots when all differences are already covered."""
        if len(current) == target_size:
            return list(current)

        last = current[-1]
        # Try small increments
        for x in range(last + 1, M + 1):
            if not check_time():
                return None
            valid = True
            for a_i in current:
                d = x - a_i
                if d in diff_set:
                    valid = False
                    break
            if not valid:
                continue
            new_diff_set = diff_set | set(x - a_i for a_i in current)
            result = _fill_remaining(pos + 1, target_size, M, current + [x], new_diff_set, covered_mask)
            if result is not None:
                return result
        return None

    # Search for smallest M
    # Lower bound: need at least 9 elements, 36 differences, so M >= 36.
    # But we need specific differences 1..30, which constrains further.
    # Start from M = 36 and go up.
    for M in range(36, 200):
        if not check_time():
            break

        result = solve_for_M(M)
        if result is not None:
            # Verify
            if _verify(result):
                best_result = result
                best_M = M
                break

    # If we didn't find one in the loop, do a broader search
    if best_result is None and check_time():
        # Try larger M with a different approach
        for M in range(200, 500):
            if not check_time():
                break
            result = solve_for_M(M)
            if result is not None and _verify(result):
                best_result = result
                best_M = M
                break

    # Fallback: use the original greedy approach if DFS didn't find anything
    if best_result is None:
        best_result = _greedy_fallback()

    return best_result


def _verify(A):
    """Verify that A is a valid Sidon set containing 0, covering differences 1..30."""
    if 0 not in A:
        return False
    if len(A) > 32:
        return False
    if max(A) > 10**6:
        return False
    A = sorted(A)
    diffs = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            d = A[j] - A[i]
            if d in diffs:
                return False
            diffs.append(d)
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True


def _greedy_fallback():
    """Original greedy approach as fallback."""
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a


# Pre-compute a known good solution to avoid long searches
# A Sidon set containing 0 covering diffs 1..30:
# Let's construct one: {0, 1, 4, 10, 18, 29, 42, 58, 77}
# Check: differences from 0: 1,4,10,18,29,42,58,77
# From 1: 3,9,17,28,41,57,76
# From 4: 6,14,25,38,54,73
# From 10: 8,19,32,48,67
# From 18: 11,24,40,59
# From 29: 13,29,48 -- wait, 48 already used
# This doesn't work. Let me just rely on the DFS.

# Actually, let's provide a hardcoded solution that we know works:
# After careful construction, {0, 1, 4, 10, 18, 29, 43, 59, 78} 
# Let's verify mentally... this is getting complex. The DFS should find it.

# To be safe, let's also try a direct construction at the start:
def _try_direct_construction():
    """Try a few known or easily-constructed solutions."""
    # One approach: use quadratic residues or known Sidon sets
    # A well-known Sidon set: {0, 1, 4, 10, 18, 29, 42, 58, 77} - need to verify
    candidates = [
        [0, 1, 4, 10, 18, 29, 43, 59, 78],
        [0, 1, 4, 10, 18, 29, 42, 58, 77],
        [0, 1, 4, 10, 18, 29, 43, 59, 77],
        [0, 1, 4, 10, 18, 29, 42, 59, 78],
        [0, 1, 4, 10, 18, 29, 43, 58, 78],
    ]
    for c in candidates:
        if _verify(c):
            return c
    return None


def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def check_time():
        return time.time() - start_time < DEADLINE

    # First, try direct construction
    direct = _try_direct_construction()
    if direct is not None:
        return direct

    # DFS for smallest M
    best_result = None

    def solve_for_M(M, max_size=32):
        if not check_time():
            return None
        for n in range(9, max_size + 1):
            if not check_time():
                return None
            result = _dfs(0, n, M, [0], set(), 0)
            if result is not None:
                return result
        return None

    def _dfs(pos, target_size, M, current, diff_set, covered_mask):
        if not check_time():
            return None

        if len(current) == target_size:
            if covered_mask == (1 << 30) - 1:
                return list(current)
            return None

        missing = 0
        for d in range(1, 31):
            if not (covered_mask >> (d-1)) & 1:
                missing += 1

        remaining_slots = target_size - len(current)
        if missing == 0:
            if remaining_slots == 0:
                return list(current)
            return _fill_remaining(target_size, M, current, diff_set, covered_mask)

        if remaining_slots * len(current) < missing:
            return None

        last = current[-1]
        candidates = set()
        for a_i in current:
            for d in range(1, 31):
                if not (covered_mask >> (d-1)) & 1:
                    x = a_i + d
                    if last < x <= M:
                        candidates.add(x)
        for delta in range(1, 10):
            x = last + delta
            if x <= M:
                candidates.add(x)

        candidates = sorted(candidates, key=lambda x: -sum(
            1 for a_i in current
            if 1 <= x - a_i <= 30 and not (covered_mask >> (x - a_i - 1)) & 1
        ))

        candidates = candidates[:40]

        for x in candidates:
            if not check_time():
                return None
            valid = True
            new_diffs = []
            for a_i in current:
                d = x - a_i
                if d in diff_set:
                    valid = False
                    break
                new_diffs.append(d)
            if not valid:
                continue

            new_diff_set = diff_set | set(new_diffs)
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))

            result = _dfs(pos + 1, target_size, M, current + [x], new_diff_set, new_mask)
            if result is not None:
                return result
        return None

    def _fill_remaining(target_size, M, current, diff_set, covered_mask):
        if len(current) == target_size:
            return list(current)
        if not check_time():
            return None
        last = current[-1]
        for x in range(last + 1, min(last + 20, M + 1)):
            valid = True
            for a_i in current:
                d = x - a_i
                if d in diff_set:
                    valid = False
                    break
            if not valid:
                continue
            new_diff_set = diff_set | set(x - a_i for a_i in current)
            result = _fill_remaining(target_size, M, current + [x], new_diff_set, covered_mask)
            if result is not None:
                return result
        return None

    for M in range(36, 150):
        if not check_time():
            break
        result = solve_for_M(M)
        if result is not None and _verify(result):
            best_result = result
            break

    if best_result is None and check_time():
        for M in range(150, 400):
            if not check_time():
                break
            result = solve_for_M(M)
            if result is not None and _verify(result):
                best_result = result
                break

    if best_result is None:
        best_result = _greedy_fallback()

    return best_result
