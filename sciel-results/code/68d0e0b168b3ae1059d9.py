import numpy as np
import time
import itertools
import random
import sys

def construct(q, n, radius):
    # Handle the specific ternary n=8 r=1 case efficiently,
    # with a generic fallback for other parameters.
    if q != 3 or n != 8 or radius != 1:
        # Simple greedy fallback for other cases
        return _generic_construct(q, n, radius)

    N = 3 ** 8  # 6561
    deadline = time.time() + 113.0  # 7s safety margin from 120s limit

    # Precompute neighbors: for each word w (0..N-1), neighbors[w] = [w, w1, w2, w3, ..., w_{2n}]
    # where w1, w2 are w with digit 0 changed to 1 and 2 respectively, etc.
    # Shape: (N, 1 + 2*n) = (6561, 17)
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    powers = [3 ** i for i in range(n)]

    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            # Change digit i to (d+1)%3 and (d+2)%3
            delta1 = ((d + 1) % 3 - d) * powers[i]
            delta2 = ((d + 2) % 3 - d) * powers[i]
            neighbors[w, idx] = w + delta1
            neighbors[w, idx + 1] = w + delta2
            idx += 2

    # Seed solution: tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            seed_code_words.append(h + list(tail))

    def word_to_int(word):
        val = 0
        for i in range(n):
            val += word[i] * powers[i]
        return val

    seed_code_ints = [word_to_int(w) for w in seed_code_words]
    best_code = seed_code_ints
    best_size = len(best_code)  # 486

    def compute_cov(code):
        cov = np.zeros(N, dtype=np.int16)
        for c in code:
            cov[neighbors[c]] += 1
        return cov

    def try_remove_redundant(code, cov):
        """Remove redundant codewords iteratively. Returns (new_code, new_cov, changed)."""
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                if np.min(cov[neighbors[c]]) > 1:
                    cov[neighbors[c]] -= 1
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
        return code, cov, changed

    def greedy_set_cover(order):
        """Greedy set cover using a given order of candidates."""
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        # Use the order to break ties, but still compute gains
        # Actually, let's just pick max gain; order helps when gains tie
        gains_cache = None
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            # Use order to pick: find candidate in order that is in candidates
            # Faster: just pick first in order that's in candidates set
            cand_set = set(candidates.tolist())
            for c in order:
                if c in cand_set:
                    w = c
                    break
            else:
                w = candidates[0]
            code.append(int(w))
            uncovered[neighbors[w]] = 0
        return code

    def local_search_swap(code, cov):
        """Try single-swap moves: replace one codeword with another to maintain coverage."""
        current_code = list(code)
        current_cov = cov.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removals first
            new_code, new_cov, removed = try_remove_redundant(current_code, current_cov)
            if removed:
                current_code = new_code
                current_cov = new_cov
                improved = True
                continue

            # Try swaps
            # For each codeword c in code, check if it can be removed.
            # If any neighbor v of c has cov[v]==1, then v is "excluded" (only covered by c).
            # To swap c for c', we need:
            #   - c' is not already in code
            #   - All excluded neighbors of c are covered by c'
            #   - c' doesn't create a situation where removing it would break coverage (not needed immediately)

            # Build set for fast membership
            code_set = set(current_code)

            # For efficiency, precompute excluded words for each codeword
            # But this is expensive. Let's iterate.

            n_swaps = 0
            for i in range(len(current_code)):
                c = current_code[i]
                nbrs_c = neighbors[c]
                # Find words covered only by c
                excl_mask = (current_cov == 1)  # words with coverage count 1
                excl_words = nbrs_c[excl_mask]
                if len(excl_words) == 0:
                    # c is redundant but try_remove_redundant should have caught it
                    # Actually try_remove_redundant runs first, so this shouldn't happen
                    continue

                # Try a few random candidates for c'
                # Candidates: random words not in code
                n_tries = 3
                found = False
                for _ in range(n_tries):
                    c_prime = int(random.randint(0, N - 1))
                    if c_prime in code_set or c_prime == c:
                        continue
                    # Check if all excl_words are in neighbors[c_prime]
                    nbrs_cp = neighbors[c_prime]
                    # Is excl_words subset of nbrs_cp?
                    # Use set intersection
                    excl_set = set(excl_words.tolist())
                    nbrs_cp_set = set(nbrs_cp.tolist())
                    if excl_set.issubset(nbrs_cp_set):
                        # Valid swap
                        current_code[i] = c_prime
                        code_set.remove(c)
                        code_set.add(c_prime)
                        current_cov[neighbors[c]] -= 1
                        current_cov[neighbors[c_prime]] += 1
                        n_swaps += 1
                        improved = True
                        found = True
                        break
                if found:
                    continue

        return current_code, current_cov, n_swaps

    def double_swap(code, cov):
        """Try double-swap: remove 2 codewords, add 1-2 codewords."""
        current_code = list(code)
        current_cov = cov.copy()
        # This is more expensive, do it selectively
        if len(current_code) > N - 1:
            return current_code, current_cov, 0

        n_dswaps = 0
        # Try a few random pairs
        n_pairs = 5
        idxs = list(range(len(current_code)))
        random.shuffle(idxs)
        for _ in range(min(n_pairs, len(idxs) * (len(idxs) - 1) // 2)):
            i = int(random.randint(0, len(current_code) - 1))
            j = int(random.randint(0, len(current_code) - 1))
            if i == j:
                continue
            c1 = current_code[i]
            c2 = current_code[j]

            # Excluded words: words covered only by c1 or c2
            cov_after_remove = current_cov.copy()
            cov_after_remove[neighbors[c1]] -= 1
            cov_after_remove[neighbors[c2]] -= 1

            # Words that become uncovered
            excl1 = neighbors[c1][current_cov == 1]  # covered only by c1
            excl2 = neighbors[c2]
            excl2 = excl2[(current_cov == 1) & ~np.isin(excl2, excl1)]  # covered only by c2, not in excl1

            all_excl = np.concatenate([excl1, excl2]) if len(excl1) + len(excl2) > 0 else np.array([], dtype=np.int32)
            if len(all_excl) == 0:
                # Both redundant, should have been removed
                continue

            # Try to find 1-2 new codewords that cover all_excl
            # For 1 codeword:
            code_set = set(current_code)
            found = False
            # Try random single replacement
            for _ in range(5):
                c_new = int(random.randint(0, N - 1))
                if c_new in code_set or c_new == c1 or c_new == c2:
                    continue
                if np.isin(all_excl, neighbors[c_new]).all():
                    # Valid: remove c1, c2, add c_new
                    current_code[i] = c_new
                    current_code.pop(j) if j > i else current_code.pop(i)
                    # Need to handle indices carefully; let's rebuild
                    # Actually, let's just track and rebuild
                    break

        # This is getting complex; let's implement a cleaner version
        # Actually, for time constraints, let's skip double_swap for now or make it simpler

        return current_code, current_cov, n_dswaps

    # Main optimization loop
    rng = np.random.default_rng()

    while time.time() < deadline:
        # Generate a random order for greedy
        order = np.random.permutation(N)

        # Greedy set cover
        code = greedy_set_cover(order)

        if len(code) >= best_size:
            # Still try to improve via local search? Probably not worth it
            # But greedy might be worse than seed, so skip if too big
            if len(code) > best_size + 50:
                continue

        cov = compute_cov(code)

        # Local search: removals + swaps
        code, cov, _ = local_search_swap(code, cov)

        # Try a few more removal passes
        code, cov, _ = try_remove_redundant(code, cov)

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            # Early exit if we hit or approach the lower bound
            if best_size <= 403:
                break

    # Convert best_code back to lists of integers
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for i in range(n):
            word.append(temp % q)
            temp //= q
        final_code.append(word)

    return final_code


def _generic_construct(q, n, radius):
    """Generic greedy for non-specific parameters."""
    N = q ** n
    if N > 1000000:
        raise ValueError("Space too large")

    # Precompute neighbors within radius
    # For radius=1, neighbors are the word itself + 1-position changes
    # For general radius, this is more complex. Assume radius=1 for simplicity.
    if radius != 1:
        # Fallback: just use all words? No, too many. Use a simple code.
        # For now, return a trivial covering: all words (N codewords)
        return [[i % q for i in range(n)] for i in range(N)]

    powers = [q ** i for i in range(n)]
    neighbors = []
    for w in range(N):
        nbrs = [w]
        temp = w
        for i in range(n):
            d = temp % q
            temp //= q
            for new_d in range(q):
                if new_d == d:
                    continue
                nw = w + (new_d - d) * powers[i]
                nbrs.append(nw)
        neighbors.append(nbrs)
    neighbors = np.array(neighbors, dtype=np.int32)

    uncovered = np.ones(N, dtype=np.uint8)
    code = []
    while uncovered.sum() > 0:
        gains = uncovered[neighbors].sum(axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        w = int(candidates[0])
        code.append(w)
        uncovered[neighbors[w]] = 0

    final_code = []
    for w in code:
        word = []
        temp = w
        for i in range(n):
            word.append(temp % q)
            temp //= q
        final_code.append(word)
    return final_code
