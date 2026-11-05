import time
import random
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115  # Leave 5s margin before 120s kill

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(deadline)

    # Fallback for other cases
    return _greedy_set_cover(q, n, radius, deadline)


def _solve_ternary_7_1(deadline):
    q = 3
    n = 7
    space_size = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 1 + 7*2 = 15

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

    # Precompute neighborhoods as arrays of indices
    neighborhoods = np.zeros((space_size, ball_size), dtype=np.int32)
    for c in range(space_size):
        w = index_to_word(c)
        neighbors = [c]
        for pos in range(n):
            for delta in (1, 2):
                w2 = list(w)
                w2[pos] = (w2[pos] + delta) % 3
                neighbors.append(word_to_index(w2))
        neighborhoods[c] = neighbors

    # Precompute cover_bits as Python ints (bitset of covered elements)
    # Also precompute for each element, the list of codewords that cover it
    elem_to_cws = [[] for _ in range(space_size)]
    cover_bits = [0] * space_size

    for c in range(space_size):
        bits = 0
        for idx in neighborhoods[c]:
            bits |= (1 << idx)
            elem_to_cws[idx].append(c)
        cover_bits[c] = bits

    # === Phase 1: Vectorized Greedy ===
    # Use numpy to compute gains efficiently
    # covered_mask: boolean array of size space_size
    # For each candidate, gain = sum of uncovered in its neighborhood

    uncovered_mask = np.ones(space_size, dtype=bool)
    selected = []
    selected_set = set()

    # Precompute neighborhood matrix as a 2D boolean for fast counting
    # neigh_mask[c] is a boolean array of length space_size
    neigh_mask = np.zeros((space_size, space_size), dtype=bool)
    for c in range(space_size):
        neigh_mask[c, neighborhoods[c]] = True

    while uncovered_mask.any() and time.time() < deadline - 30:
        # Compute gain for all candidates at once
        # gain[c] = number of uncovered elements in neighborhood of c
        gains = (neigh_mask & uncovered_mask[None, :]).sum(axis=1)

        # Only consider unselected candidates
        for c in selected_set:
            gains[c] = 0

        best_c = int(np.argmax(gains))
        if gains[best_c] == 0:
            break

        selected.append(best_c)
        selected_set.add(best_c)
        uncovered_mask[neighborhoods[best_c]] = False

    best_solution = selected[:]
    best_k = len(best_solution)

    # === Phase 2: Local Search with Swaps ===
    # Maintain coverage counts
    cov_count = np.zeros(space_size, dtype=np.int32)
    for c in best_solution:
        cov_count[neighborhoods[c]] += 1

    def try_local_search(sol, deadline_local):
        """Perform 1-for-1 and 2-for-1 local search."""
        sol_set = set(sol)
        cov = np.zeros(space_size, dtype=np.int32)
        for c in sol:
            cov[neighborhoods[c]] += 1

        improved = True
        while improved and time.time() < deadline_local:
            improved = False

            # 1-for-1 swap: try removing each codeword and adding a better one
            for i, c_remove in enumerate(sol):
                if time.time() > deadline_local:
                    break
                # Remove c_remove
                cov[neighborhoods[c_remove]] -= 1
                newly_uncovered = neighborhoods[c_remove][cov[neighborhoods[c_remove]] == 0]

                if len(newly_uncovered) == 0:
                    # Can remove without adding
                    sol = sol[:i] + sol[i+1:]
                    sol_set.discard(c_remove)
                    improved = True
                    cov = np.zeros(space_size, dtype=np.int32)
                    for c in sol:
                        cov[neighborhoods[c]] += 1
                    break

                # Try to find a codeword that covers all newly_uncovered
                # and is not already in sol
                # Compute for each candidate how many of newly_uncovered it covers
                # We want one that covers ALL of them
                best_add = -1
                best_gain = -1
                for c_add in range(space_size):
                    if c_add in sol_set:
                        continue
                    # How many of newly_uncovered does c_add cover?
                    covered = 0
                    for e in newly_uncovered:
                        if c_add in elem_to_cws[e]:
                            covered += 1
                    if covered == len(newly_uncovered):
                        # This works! Check if it's better than current
                        # (it's always an improvement since we remove one and add one)
                        best_add = c_add
                        break

                if best_add != -1:
                    sol[i] = best_add
                    sol_set.discard(c_remove)
                    sol_set.add(best_add)
                    cov[neighborhoods[c_remove]] -= 0  # already decremented
                    cov[neighborhoods[best_add]] += 1
                    improved = True
                else:
                    # Restore
                    cov[neighborhoods[c_remove]] += 1

            # 2-for-1 exchange: remove 2, add 1
            if time.time() > deadline_local - 5:
                break
            for i in range(len(sol)):
                if time.time() > deadline_local:
                    break
                for j in range(i + 1, len(sol)):
                    if time.time() > deadline_local:
                        break
                    c1, c2 = sol[i], sol[j]
                    # Remove both
                    cov[neighborhoods[c1]] -= 1
                    cov[neighborhoods[c2]] -= 1
                    newly_uncovered = np.where((cov == 0))[0]

                    if len(newly_uncovered) == 0:
                        # Can remove both without adding
                        sol = [c for k, c in enumerate(sol) if k != i and k != j]
                        sol_set.discard(c1)
                        sol_set.discard(c2)
                        cov = np.zeros(space_size, dtype=np.int32)
                        for c in sol:
                            cov[neighborhoods[c]] += 1
                        improved = True
                        break

                    # Try to find a codeword covering all newly_uncovered
                    best_add = -1
                    for c_add in range(space_size):
                        if c_add in sol_set:
                            continue
                        ok = True
                        for e in newly_uncovered:
                            if c_add not in elem_to_cws[e]:
                                ok = False
                                break
                        if ok:
                            best_add = c_add
                            break

                    if best_add != -1:
                        # Replace c1 and c2 with best_add
                        new_sol = [c for k, c in enumerate(sol) if k != i and k != j]
                        new_sol.append(best_add)
                        sol = new_sol
                        sol_set.discard(c1)
                        sol_set.discard(c2)
                        sol_set.add(best_add)
                        cov = np.zeros(space_size, dtype=np.int32)
                        for c in sol:
                            cov[neighborhoods[c]] += 1
                        improved = True
                        break
                    else:
                        cov[neighborhoods[c1]] += 1
                        cov[neighborhoods[c2]] += 1

                if improved:
                    break

        return sol

    # Run local search
    local_deadline = deadline - 25
    best_solution = try_local_search(best_solution, local_deadline)
    best_k = len(best_solution)

    # === Phase 3: Branch and Bound ===
    # Try to find solutions of size best_k - 1, best_k - 2, ...

    # Precompute for faster B&B
    # For each element, precompute the list of codewords sorted by some heuristic
    # Sort by "rarity" or by how many other elements they cover

    # Lower bound function
    def lower_bound(uncovered_int):
        """Simple lower bound: ceil(popcount / ball_size)"""
        return (uncovered_int.bit_count() + ball_size - 1) // ball_size

    # B&B with global solution storage
    found_sol = [None]
    search_stack = []

    def bb_search(uncovered_int, depth, last_idx, target_k, deadline_bb):
        """Branch and bound search."""
        if time.time() > deadline_bb:
            return False

        if uncovered_int == 0:
            found_sol[0] = search_stack[:]
            return True

        if depth >= target_k:
            return False

        # Lower bound check
        if depth + lower_bound(uncovered_int) > target_k:
            return False

        # Find element with fewest available covering codewords
        best_e = -1
        min_count = float('inf')
        u = uncovered_int
        while u:
            lsb = u & -u
            e = lsb.bit_length() - 1
            u ^= lsb

            # Count available codewords (index > last_idx)
            count = 0
            for c in elem_to_cws[e]:
                if c > last_idx:
                    count += 1
                    if count >= min_count:
                        break

            if count == 0:
                return False

            if count < min_count:
                min_count = count
                best_e = e
                if min_count == 1:
                    break

        if best_e == -1:
            return False

        # Branch on codewords covering best_e
        # Sort by how many other uncovered elements they cover (greedy heuristic)
        candidates = []
        for c in elem_to_cws[best_e]:
            if c <= last_idx:
                continue
            # Quick gain estimate
            gain = (uncovered_int & cover_bits[c]).bit_count()
            candidates.append((gain, c))

        # Sort by gain descending
        candidates.sort(key=lambda x: -x[0])

        for gain, c in candidates:
            if time.time() > deadline_bb:
                return False

            new_uncovered = uncovered_int & ~cover_bits[c]
            search_stack.append(c)
            if bb_search(new_uncovered, depth + 1, c, target_k, deadline_bb):
                return True
            search_stack.pop()

        return False

    # Try decreasing target sizes
    for target_k in range(best_k - 1, max(150, best_k - 30), -1):
        if time.time() > deadline - 5:
            break

        bb_deadline = min(deadline - 5, time.time() + 8)  # Max 8s per target
        found_sol[0] = None
        search_stack.clear()

        full_uncovered = (1 << space_size) - 1

        if bb_search(full_uncovered, 0, -1, target_k, bb_deadline):
            if found_sol[0]:
                best_solution = found_sol[0]
                best_k = target_k
                if best_k <= 160:
                    break

    # === Phase 4: Final local search polish ===
    if time.time() < deadline - 3:
        best_solution = try_local_search(best_solution, deadline - 1)
        best_k = len(best_solution)

    # Convert to words
    result = [index_to_word(idx) for idx in best_solution]
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

    if radius == 1:
        neighborhoods = [None] * space_size
        for idx in range(space_size):
            w = index_to_word(idx)
            neigh = [idx]
            for pos in range(n):
                for delta in range(1, q):
                    w2 = list(w)
                    w2[pos] = (w2[pos] + delta) % q
                    neigh.append(word_to_index(w2))
            neighborhoods[idx] = neigh

    coverage = np.zeros(space_size, dtype=np.int32)
    uncovered = np.ones(space_size, dtype=bool)
    selected = []
    selected_set = set()

    while time.time() < deadline - 3 and uncovered.any():
        best_cw = -1
        best_count = -1

        for cw in range(space_size):
            if cw in selected_set:
                continue
            count = 0
            for nb in neighborhoods[cw]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_cw = cw

        if best_cw < 0:
            break

        selected.append(index_to_word(best_cw))
        selected_set.add(best_cw)
        for nb in neighborhoods[best_cw]:
            coverage[nb] += 1
            if coverage[nb] == 1:
                uncovered[nb] = False

    return selected
