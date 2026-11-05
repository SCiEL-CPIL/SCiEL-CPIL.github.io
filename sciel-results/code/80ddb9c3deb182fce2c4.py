import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s kill

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors for radius=1: self + all 1-change neighbors
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # Precompute reverse neighbors: for each word, which codewords cover it
    # This is the same as neighbors transposed conceptually, but we'll build it lazily
    # Actually, for a codeword c, neighbors[c] gives all words it covers.
    # For a word v, the codewords that cover v are those c where v in neighbors[c].
    # We can precompute cover_map: word -> list of codewords that cover it
    # But that's space_size x space_size which is 2187^2 ~ 4.8M, acceptable.
    # Actually, we can compute it on the fly or precompute as a list of lists.
    # Let's precompute cover_map as a list of lists for speed.
    cover_map = [[] for _ in range(space_size)]
    for c in range(space_size):
        for v in neighbors[c]:
            cover_map[v].append(c)
    # Convert to numpy arrays for speed
    cover_map_arrays = [np.array(lst, dtype=np.int32) for lst in cover_map]

    def build_cov(code_set):
        """Build coverage count array from a set of codewords."""
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_set:
            cov[neighbors[c]] += 1
        return cov

    def is_valid(cov):
        return np.all(cov >= 1)

    # Phase 1: Initial solution via greedy set cover
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 30.0:
            # Find codeword covering most uncovered words
            gains = np.sum(uncovered[neighbors], axis=1)
            # Add small random perturbation for diversity
            perturb = np.array([rng.random() * 0.001 for _ in range(space_size)])
            gains_pert = gains + perturb
            best = int(np.argmax(gains_pert))
            if gains[best] == 0:
                break
            code.append(best)
            uncovered[neighbors[best]] = False
        return code

    # Phase 2: ILS with fixed-size pool
    # We maintain a candidate pool of size ~200-250 and search for subsets of size ~186
    # Objective: minimize the number of words with coverage < 1 (i.e., maximize min coverage)
    # Actually, we want a valid covering (all cov >= 1) with minimum size.
    # Strategy: Start with a valid covering of size ~190-200, then try to reduce size
    # by swapping codewords and removing redundant ones.

    def local_search_reduce(code, deadline_time):
        """Try to reduce the size of a valid covering by removing redundant codewords."""
        cur_code = code[:]
        cur_cov = build_cov(cur_code)
        changed = True
        while changed and (time.time() - t_start) < deadline_time:
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    # Can remove c
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
        return cur_code, cur_cov

    def ils_search(initial_code, deadline_time):
        """Iterated Local Search on a valid covering."""
        best_code = initial_code[:]
        best_size = len(initial_code)
        cur_code = initial_code[:]
        cur_cov = build_cov(cur_code)
        rng = random.Random(42)

        while (time.time() - t_start) < deadline_time:
            # Try to improve cur_code
            improved = False
            for _ in range(100):
                if (time.time() - t_start) > deadline_time:
                    break
                if len(cur_code) <= 1:
                    break
                # Random swap: remove one codeword, add another
                i = rng.randint(0, len(cur_code) - 1)
                j = rng.randint(0, space_size - 1)
                if j in cur_code:
                    continue
                ci = cur_code[i]
                # Check if swapping ci with j maintains validity
                new_cov = cur_cov.copy()
                new_cov[neighbors[ci]] -= 1
                new_cov[neighbors[j]] += 1
                if np.all(new_cov >= 1):
                    cur_code[i] = j
                    cur_cov = new_cov
                    improved = True
                    # Try to reduce after swap
                    cur_code, cur_cov = local_search_reduce(cur_code, deadline_time)
                    if len(cur_code) < best_size:
                        best_code = cur_code[:]
                        best_size = len(cur_code)
                    break

            if not improved:
                # Kick: random perturbation
                # Remove 2-3 random codewords and add 2-3 random codewords
                k = rng.randint(2, 3)
                for _ in range(k):
                    if len(cur_code) > 1:
                        i = rng.randint(0, len(cur_code) - 1)
                        cur_code.pop(i)
                # Rebuild coverage
                cur_cov = build_cov(cur_code)
                # Repair: add codewords to cover uncovered words
                uncovered_mask = cur_cov == 0
                while np.any(uncovered_mask) and (time.time() - t_start) < deadline_time:
                    gains = np.sum(uncovered_mask[neighbors], axis=1)
                    best = int(np.argmax(gains))
                    if gains[best] == 0:
                        break
                    cur_code.append(best)
                    cur_cov[neighbors[best]] += 1
                    uncovered_mask = cur_cov == 0
                # Reduce
                cur_code, cur_cov = local_search_reduce(cur_code, deadline_time)

        return best_code

    # Run multiple ILS instances with different initial solutions
    best_code = None
    best_size = space_size + 1

    for seed in range(8):
        if (time.time() - t_start) > deadline - 20.0:
            break
        code = greedy_cover(seed)
        if not code:
            continue
        cov = build_cov(code)
        if not np.all(cov >= 1):
            # Repair
            uncovered_mask = cov == 0
            while np.any(uncovered_mask) and (time.time() - t_start) < deadline - 20.0:
                gains = np.sum(uncovered_mask[neighbors], axis=1)
                best = int(np.argmax(gains))
                if gains[best] == 0:
                    break
                code.append(best)
                cov[neighbors[best]] += 1
                uncovered_mask = cov == 0
        # Reduce
        code, cov = local_search_reduce(code, deadline - 20.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)
        # ILS on this solution
        if (time.time() - t_start) < deadline - 15.0:
            ils_result = ils_search(code, deadline - 15.0)
            if len(ils_result) < best_size:
                best_code = ils_result[:]
                best_size = len(ils_result)

    # Final fallback
    if best_code is None:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov >= 1):
        # Emergency repair
        uncovered_mask = final_cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < deadline - 5.0:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            best_code.append(best)
            final_cov[neighbors[best]] += 1
            uncovered_mask = final_cov == 0

    return [idx_to_word(c) for c in best_code]
