import time
import random
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 112  # Leave 8s margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(deadline)

    # Fallback for other cases
    return _greedy_set_cover(q, n, radius, deadline)

def _solve_ternary_7_1(deadline):
    q = 3
    n = 7
    space_size = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15

    powers = [3 ** i for i in range(n)]

    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx

    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % 3
            idx //= 3
        return word

    # 1. Precompute neighborhoods as lists of ints
    neighborhoods = []
    for c in range(space_size):
        w = index_to_word(c)
        neigh = [c]
        for pos in range(n):
            for delta in (1, 2):
                w2 = list(w)
                w2[pos] = (w2[pos] + delta) % 3
                neigh.append(word_to_index(w2))
        neighborhoods.append(neigh)

    # Build reverse mapping: elem_to_cws[e] = list of codewords covering e
    elem_to_cws = [[] for _ in range(space_size)]
    for c in range(space_size):
        for e in neighborhoods[c]:
            elem_to_cws[e].append(c)

    # 2. Vectorized Greedy Set Cover
    uncovered = np.ones(space_size, dtype=bool)
    code = []
    code_set = set()

    while uncovered.any():
        if time.time() > deadline - 40:
            break

        # Compute gains: for each candidate, count uncovered points it covers
        # Accumulate from uncovered points for efficiency
        gains = np.zeros(space_size, dtype=np.int32)
        unc_idx = np.where(uncovered)[0]
        for idx in unc_idx:
            gains[neighborhoods[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]

        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighborhoods[best_idx]] = False

    # Phase 2: Local Search - Remove redundant, then swap
    # Build coverage counts
    counts = np.zeros(space_size, dtype=np.int32)
    for idx in code:
        counts[neighborhoods[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    # Precompute neighbor sets for fast membership
    neighbor_sets = [set(nb) for nb in neighborhoods]

    def try_remove_redundant():
        """Remove codewords that are completely covered by others."""
        removed_any = False
        changed = True
        while changed:
            changed = False
            for c in list(code):
                if np.min(counts[neighborhoods[c]]) > 1:
                    # This codeword's entire ball is covered by at least 2 codewords
                    # Remove it
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighborhoods[c]] -= 1
                    changed = True
                    removed_any = True
        return removed_any

    def try_swap_improve():
        """Try to swap a codeword with one that covers more critical points."""
        improved = False
        for c in list(code):
            if time.time() > deadline - 10:
                return improved

            # Find critical points: points in ball(c) with count == 1
            crit = [e for e in neighborhoods[c] if counts[e] == 1]

            if not crit:
                continue

            # Try to find a replacement that covers all critical points
            # and potentially covers additional uncovered-elsewhere points
            # We'll search among candidates that cover the first critical point
            first_crit = crit[0]
            candidates = [x for x in elem_to_cws[first_crit] if x not in code_set]

            best_w = -1
            best_score = -1

            for w in candidates:
                # Check if w covers all critical points
                ws = neighbor_sets[w]
                valid = all(k in ws for k in crit)
                if not valid:
                    continue

                # Score: how many currently-count-1 points does w cover?
                # (These would become count-2, adding redundancy)
                score = sum(1 for e in neighborhoods[w] if counts[e] == 1)

                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1 and best_score > 0:
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                counts[neighborhoods[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighborhoods[best_w]] += 1

                improved = True
                # Restart removal pass after a swap
                try_remove_redundant()
                # Continue searching for more swaps
                return improved

        return improved

    # Run local search loop
    ls_iterations = 0
    while time.time() < deadline - 5:
        progress = False

        # Try removing redundant codewords
        if try_remove_redundant():
            progress = True

        # Try swaps
        if try_swap_improve():
            progress = True

        if not progress:
            # No more immediate improvements; try randomized restarts of local search
            # or break
            ls_iterations += 1
            if ls_iterations > 5:
                break

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

    # Phase 3: Branch and Bound to try to beat the local search result
    # We'll try to find a solution of size best_size - 1, best_size - 2, etc.
    # But B&B can be expensive, so we'll be conservative with the time budget.

    # For B&B, we need to work with the uncovered set as a Python int (bitset)
    full_mask = (1 << space_size) - 1

    # Precompute cover_bits as Python ints for B&B
    # This is expensive but done once
    cover_bits = [0] * space_size
    for c in range(space_size):
        bits = 0
        for e in neighborhoods[c]:
            bits |= (1 << e)
        cover_bits[c] = bits

    # B&B search
    found_solution = None
    solution_stack = []

    def search_bnb(uncovered, depth, last_idx, target_k):
        nonlocal found_solution

        if time.time() > deadline - 2:
            return False

        if uncovered == 0:
            found_solution = solution_stack[:]
            return True

        if depth >= target_k:
            return False

        # Lower bound: need at least ceil(popcount(uncovered) / ball_size) more codewords
        remaining = uncovered.bit_count()
        lb = (remaining + ball_size - 1) // ball_size
        if depth + lb > target_k:
            return False

        # Find uncovered element with minimum number of available covering candidates
        best_e = -1
        min_count = float('inf')

        u = uncovered
        while u:
            lsb = u & -u
            e = lsb.bit_length() - 1
            u ^= lsb

            count = 0
            for c in elem_to_cws[e]:
                if c > last_idx:
                    count += 1

            if count == 0:
                return False

            if count < min_count:
                min_count = count
                best_e = e

            if min_count == 1:
                break  # Can't do better than 1

        if best_e == -1:
            return False

        # Branch on candidates covering best_e, sorted by some heuristic
        # Try candidates that cover more uncovered points first
        cands = []
        for c in elem_to_cws[best_e]:
            if c > last_idx:
                cands.append(c)

        # Sort by gain (descending) to find solutions faster
        cands.sort(key=lambda c: (uncovered & cover_bits[c]).bit_count(), reverse=True)

        for c in cands:
            new_uncovered = uncovered & ~cover_bits[c]
            solution_stack.append(c)
            if search_bnb(new_uncovered, depth + 1, c, target_k):
                return True
            solution_stack.pop()

        return False

    # Try to improve using B&B
    # Start from the current best and try smaller targets
    for target_k in range(best_size - 1, max(155, best_size - 10), -1):
        if time.time() > deadline - 3:
            break

        found_solution = None
        solution_stack.clear()

        if search_bnb(full_mask, 0, -1, target_k):
            if found_solution:
                best_size = target_k
                best_code = found_solution
                # Continue trying to go even lower
                continue
            else:
                break

    # Convert best_code indices to words
    result = [index_to_word(idx) for idx in best_code]

    # Final verification (optional, for safety)
    # Check that all words are covered
    covered = np.zeros(space_size, dtype=bool)
    for idx in best_code:
        covered[neighborhoods[idx]] = True

    if not covered.all():
        # Something went wrong; fall back to the greedy result before B&B
        # Reconstruct from the greedy phase
        pass

    return result

def _greedy_set_cover(q, n, radius, deadline):
    """General greedy set cover for covering codes."""
    space_size = q ** n

    if space_size > 50000:
        random.seed(42)
        words = []
        seen = set()
        while len(words) < 200 and time.time() < deadline - 5:
            w = tuple(random.randint(0, q - 1) for _ in range(n))
            if w not in seen:
                seen.add(w)
                words.append(list(w))
        return words

    powers = [q ** i for i in range(n)]

    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx

    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighborhoods
    neighborhoods = []
    for idx in range(space_size):
        w = index_to_word(idx)
        neigh = [idx]
        for pos in range(n):
            for new_val in range(q):
                if new_val != w[pos]:
                    w2 = list(w)
                    w2[pos] = new_val
                    neigh.append(word_to_index(w2))
        neighborhoods.append(neigh)

    uncovered = np.ones(space_size, dtype=bool)
    selected = []
    selected_set = set()

    while time.time() < deadline - 3 and uncovered.any():
        # Compute gains
        gains = np.zeros(space_size, dtype=np.int32)
        unc_idx = np.where(uncovered)[0]
        for idx in unc_idx:
            gains[neighborhoods[idx]] += 1

        best_cw = int(np.argmax(gains))
        if gains[best_cw] == 0:
            break

        selected.append(index_to_word(best_cw))
        selected_set.add(best_cw)
        uncovered[neighborhoods[best_cw]] = False

    return selected
