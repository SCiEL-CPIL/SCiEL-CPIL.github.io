"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 150s limit

    target_diffs = set(range(1, 31))
    best_solution = None
    best_max_val = float('inf')

    def is_valid_sidon(elements):
        """Check if a set of elements forms a Sidon set."""
        diffs = set()
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                d = elements[j] - elements[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(elements):
        """Get all positive differences in a sorted list."""
        diffs = []
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(elements[j] - elements[i])
        return diffs

    def covers_all_diffs(elements):
        """Check if all differences 1..30 are covered."""
        diffs = set(get_diffs(elements))
        return target_diffs.issubset(diffs)

    def objective(elements):
        """Return max element if valid, inf otherwise."""
        if not is_valid_sidon(elements):
            return float('inf')
        if not covers_all_diffs(elements):
            return float('inf')
        return max(elements)

    def backtrack_build(elements, used_diffs, upper_bound, depth=0):
        """
        Backtracking search to build a Sidon set that covers all diffs 1-30.
        elements: sorted list of current elements (must contain 0)
        used_diffs: set of all positive differences so far
        upper_bound: max allowed value for new elements (best_max_val - 1)
        """
        nonlocal best_solution, best_max_val

        if time.time() > deadline:
            return False

        # Check if we've covered all target diffs
        if target_diffs.issubset(used_diffs):
            mx = elements[-1]
            if mx < best_max_val:
                best_max_val = mx
                best_solution = list(elements)
                return True

        # Pruning: if we can't add more elements within upper_bound, fail
        if upper_bound <= elements[-1]:
            return False

        # Lower bound on elements needed: each new element adds at most len(elements) new diffs
        remaining = len(target_diffs - used_diffs)
        if remaining == 0:
            return False  # Should have been caught above

        # We need at least ceil(remaining / len(elements)) more elements
        min_more = -(-remaining // len(elements))  # ceiling division
        # Each new element must be > elements[-1] and <= upper_bound
        # So we need at least min_more positions available
        if min_more > upper_bound - elements[-1]:
            return False

        # MRV: find the most constrained uncovered difference
        # A difference d is "constrained" if it has few valid placements
        uncovered = target_diffs - used_diffs
        best_d = None
        best_options = None
        best_count = float('inf')

        for d in sorted(uncovered):
            options = set()
            for y in elements:
                x = y + d
                if x <= elements[-1] or x > upper_bound:
                    continue
                # Check if adding x would create duplicate differences
                valid = True
                for y2 in elements:
                    diff = x - y2
                    if diff in used_diffs:
                        valid = False
                        break
                if valid:
                    # Also check internal consistency of new diffs
                    new_diffs = [x - y2 for y2 in elements]
                    if len(set(new_diffs)) != len(new_diffs):
                        valid = False
                if valid:
                    options.add(x)
            if len(options) == 0:
                return False  # Dead end: this difference can't be covered
            if len(options) < best_count:
                best_count = len(options)
                best_d = d
                best_options = options
                if best_count == 1:
                    break

        if best_d is None:
            return False

        # Try each option for the most constrained difference
        for x in sorted(best_options):
            if time.time() > deadline:
                return False

            # Validate x thoroughly
            new_diffs = [x - y for y in elements]
            if any(nd in used_diffs for nd in new_diffs):
                continue
            if len(set(new_diffs)) != len(new_diffs):
                continue

            # Add x and recurse
            elements.append(x)
            added_diffs = set(new_diffs)
            for nd in added_diffs:
                used_diffs.add(nd)

            if backtrack_build(elements, used_diffs, upper_bound, depth + 1):
                elements.pop()
                for nd in added_diffs:
                    used_diffs.discard(nd)
                return True

            elements.pop()
            for nd in added_diffs:
                used_diffs.discard(nd)

        return False

    def local_search_minimize(elements, max_iters=20000):
        """Local search to minimize the maximum element while maintaining validity."""
        nonlocal best_solution, best_max_val

        S = sorted(elements)
        cur_obj = max(S)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            n = len(S)

            # Try moving each element to a smaller position
            for i in range(1, n):  # Skip 0
                orig = S[i]
                # Try positions from just above previous element up to orig-1
                for new_val in range(S[i-1] + 1, orig):
                    if new_val in S:
                        continue
                    trial = S[:i] + [new_val] + S[i+1:]
                    trial = sorted(trial)
                    # Check validity
                    if is_valid_sidon(trial) and covers_all_diffs(trial):
                        if trial[-1] < cur_obj:
                            S = trial
                            cur_obj = trial[-1]
                            improved = True
                            break
                if improved:
                    break

            if not improved:
                # Try removing an element (if set is larger than minimum)
                if n > 9:  # Minimum needed: 9 elements give 36 diffs, enough for 30
                    for i in range(1, n):
                        trial = S[:i] + S[i+1:]
                        if is_valid_sidon(trial) and covers_all_diffs(trial):
                            if trial[-1] < cur_obj:
                                S = trial
                                cur_obj = trial[-1]
                                improved = True
                                break
                    if improved:
                        continue

                # Try swapping: remove one, add another smaller
                for i in range(1, n):
                    orig = S[i]
                    # Try adding a new element and removing orig
                    for new_val in range(1, orig):
                        if new_val in S:
                            continue
                        trial = S[:i] + [new_val] + S[i+1:]
                        trial = sorted(trial)
                        if is_valid_sidon(trial) and covers_all_diffs(trial):
                            if trial[-1] < cur_obj:
                                S = trial
                                cur_obj = trial[-1]
                                improved = True
                                break
                    if improved:
                        break

            if not improved and random.random() < 0.1:
                # Random perturbation to escape local minima
                i = random.randint(1, n - 1)
                delta = random.randint(-3, 3)
                new_val = S[i] + delta
                if new_val > S[i-1] and (i == n-1 or new_val < S[i+1]) and new_val not in S and new_val > 0:
                    trial = sorted(S[:i] + [new_val] + S[i+1:])
                    if is_valid_sidon(trial) and covers_all_diffs(trial):
                        S = trial
                        cur_obj = trial[-1]

        if S[-1] < best_max_val:
            best_max_val = S[-1]
            best_solution = list(S)
        return S

    # Phase 1: Build a valid solution via backtracking
    # Start with just [0] and build up
    elements = [0]
    used_diffs = set()
    initial_upper = 200  # Start with a generous upper bound

    found = backtrack_build(elements, used_diffs, initial_upper)

    if not found and best_solution is None:
        # Try with larger upper bound
        elements = [0]
        used_diffs = set()
        found = backtrack_build(elements, used_diffs, 500)

    if best_solution is None:
        # Emergency: greedy construction
        S = [0]
        used = set()
        for target in range(1, 31):
            if target in used:
                continue
            placed = False
            for x in range(max(S) + 1, max(S) + 300):
                nds = [x - y for y in S]
                if all(nd not in used for nd in nds) and len(set(nds)) == len(nds):
                    S.append(x)
                    for y in S[:-1]:
                        used.add(x - y)
                    placed = True
                    break
            if not placed:
                x = max(S) + 50
                while True:
                    nds = [x - y for y in S]
                    if all(nd not in used for nd in nds) and len(set(nds)) == len(nds):
                        break
                    x += 1
                S.append(x)
                for y in S[:-1]:
                    used.add(x - y)
        if is_valid_sidon(S) and covers_all_diffs(S):
            best_solution = S
            best_max_val = max(S)

    if best_solution is None:
        # Last resort: return a large valid set
        # Use a known construction: start with [0,1,4,10,12,17,20,26,29,36,...]
        # or just build greedily with large gaps
        S = [0]
        used = set()
        for i in range(1, 31):
            if i in used:
                continue
            x = max(S) + i * 10
            while True:
                nds = [x - y for y in S]
                if all(nd not in used for nd in nds) and len(set(nds)) == len(nds):
                    break
                x += 1
            S.append(x)
            for y in S[:-1]:
                used.add(x - y)
        return S

    # Phase 2: Local search to minimize
    S_ls = local_search_minimize(best_solution, max_iters=30000)

    # Phase 3: Try backtracking from the local search result with tighter bound
    if time.time() < deadline - 10:
        elements = sorted(S_ls)
        used_diffs = set(get_diffs(elements))
        upper = best_max_val - 1
        if upper > elements[-1]:
            backtrack_build(elements, used_diffs, upper)

    return best_solution if best_solution is not None else S_ls
# EVOLVE-BLOCK-END
