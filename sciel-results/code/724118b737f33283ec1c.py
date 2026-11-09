"""Distinct subset sums with a small largest element."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find 12 distinct positive integers <= 1048576 such that all 4096 subset sums
    are distinct, minimizing the largest integer.
    """
    start_time = time.time()
    deadline = 82.0  # 8s safety margin before 90s kill

    def check_valid(elements):
        """Check all subset sums distinct using bitset operations.
        Returns True if valid, False otherwise."""
        bits = 1  # bit 0 set = sum 0 exists
        total = 0
        for e in elements:
            total += e
            if (bits << e) & bits:
                return False
            bits |= (bits << e)
        return True

    def check_valid_fast(elements):
        """Same as check_valid but with early total-sum check."""
        total = sum(elements)
        if total >= (1 << 30):
            return False
        bits = 1
        for e in elements:
            shifted = bits << e
            if shifted & bits:
                return False
            bits |= shifted
        return True

    # Phase 1: Find a valid set with a reasonably low maximum via branch-and-bound.
    # We search for elements a1 < a2 < ... < a12 with a12 <= max_val.
    # Key constraint: a1 must be 1 (to form sum 1).
    # For the remaining elements, we use a guided search.

    def find_valid_set(max_val, n=12):
        """Find a valid set of n distinct positive integers with max <= max_val.
        Uses backtracking with pruning."""
        # a1 = 1 is necessary
        elements = [1]
        current_sums = {0, 1}
        current_total = 1

        def backtrack(idx, min_val):
            nonlocal elements, current_sums, current_total
            if time.time() > deadline:
                return False
            if idx == n:
                return True

            # Try candidates from min_val upward
            # Pruning: candidate must be > 0 and distinct
            # Also: candidate + any existing sum must not equal any existing sum
            # i.e., candidate must not be in {s1 - s2 : s1, s2 in current_sums, s1 != s2}
            # More precisely: for all s in current_sums, s + candidate not in current_sums
            # i.e., candidate not in {s1 - s2 : s1, s2 in current_sums, s1 > s2}

            forbidden = set()
            for s1 in current_sums:
                for s2 in current_sums:
                    if s1 > s2:
                        forbidden.add(s1 - s2)

            for candidate in range(min_val, max_val + 1):
                if candidate in forbidden:
                    continue
                if candidate <= 0:
                    continue
                # Check: candidate + s for all s in current_sums must not be in current_sums
                valid = True
                for s in current_sums:
                    if (s + candidate) in current_sums:
                        valid = False
                        break
                if not valid:
                    continue

                # Try this candidate
                elements.append(candidate)
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + candidate)
                current_sums = new_sums
                current_total += candidate

                if backtrack(idx + 1, candidate + 1):
                    return True

                # Backtrack
                elements.pop()
                current_sums = new_sums - {s + candidate for s in current_sums}
                # Actually, let me redo this more carefully
                current_sums = set()
                # Recompute from elements
                current_sums = {0}
                for e in elements:
                    current_sums = current_sums | {s + e for s in current_sums}
                current_total = sum(elements)

            return False

        return backtrack(1, 2)

    # Phase 2: Greedy minimization - for each element, try to reduce it while maintaining validity.
    def minimize_elements(elements):
        """Try to reduce each element to its minimum valid value."""
        elements = list(elements)
        n = len(elements)

        for i in range(n):
            if time.time() > deadline:
                break
            # Try to reduce elements[i]
            old_val = elements[i]
            # The minimum valid value for position i
            # Must be > elements[i-1] (if i > 0) and must maintain distinct subset sums
            min_candidate = elements[i - 1] + 1 if i > 0 else 1

            for candidate in range(min_candidate, old_val):
                if time.time() > deadline:
                    break
                test = elements.copy()
                test[i] = candidate
                # Check distinctness
                if len(set(test)) != n:
                    continue
                # Check all subset sums distinct
                if check_valid(test):
                    elements[i] = candidate
                    break

        return elements

    # Phase 3: Local search - swap elements to find better configurations
    def local_search(elements, max_iters=200):
        """Try local moves to reduce the maximum element."""
        elements = list(elements)
        n = len(elements)
        best_max = max(elements)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            # Try reducing each element
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = elements[i]
                min_cand = elements[i - 1] + 1 if i > 0 else 1

                for candidate in range(min_cand, old_val):
                    test = elements.copy()
                    test[i] = candidate
                    if len(set(test)) != n:
                        continue
                    if check_valid(test):
                        elements = test
                        best_max = max(elements)
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return elements

    # Strategy:
    # 1. Start with a known valid set (powers of 2)
    # 2. Try to minimize each element greedily
    # 3. Try local search

    # Start: powers of 2
    elements = [1 << i for i in range(12)]
    # Verify
    assert check_valid(elements), "Powers of 2 should be valid"

    # Greedy minimization
    elements = minimize_elements(elements)

    # Local search
    elements = local_search(elements)

    # Try multiple random restarts with different initial sets
    best = elements
    best_max = max(elements)

    for restart in range(50):
        if time.time() > deadline:
            break

        # Create a perturbed valid set
        # Start from best, randomly increase one element, then minimize
        perturbed = list(best)
        idx = random.randint(0, 11)
        perturbed[idx] += random.randint(1, 50)
        if len(set(perturbed)) != 12:
            continue
        if not check_valid(perturbed):
            continue

        # Minimize
        minimized = minimize_elements(perturbed)
        if max(minimized) < best_max:
            best = minimized
            best_max = max(best)

    # Final verification
    if not check_valid(best):
        # Fallback
        best = [1 << i for i in range(12)]

    if max(best) > 1048576:
        best = [1 << i for i in range(12)]

    return best

# EVOLVE-BLOCK-END
