import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 142.0

    total_space = q ** n
    cover_count = 1 + n * (q - 1)  # 1 + 7*2 = 15 for q=3, n=7, radius=1

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Build neighbors array: neighbors_arr[i] = list of indices covered by word i
    # For radius=1: the word itself plus all words at Hamming distance 1
    neighbors_arr = np.zeros((total_space, cover_count), dtype=np.int32)
    for i in range(total_space):
        digits = [(i // powers[pos]) % q for pos in range(n)]
        neighbors_arr[i, 0] = i
        idx = 1
        for pos in range(n):
            for new_val in range(q):
                if new_val == digits[pos]:
                    continue
                diff = new_val - digits[pos]
                neighbors_arr[i, idx] = i + diff * powers[pos]
                idx += 1

    # Precompute reverse mapping: for each word, which codewords cover it
    # Since the graph is symmetric, reverse neighbors = forward neighbors
    # But we need it for efficient local search
    # reverse_cover[w] = set of codeword indices c such that w is in neighbors_arr[c]
    # Due to symmetry, this is the same as neighbors_arr[w]
    # So we can just use neighbors_arr for both

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def run_greedy(randomized=False, seed=None):
        """Vectorized greedy set cover."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        uncovered = np.ones(total_space, dtype=np.bool_)
        code = []

        while uncovered.any() and time.time() < deadline - 5:
            # Vectorized gain computation
            uncovered_expanded = uncovered[neighbors_arr]
            gains = uncovered_expanded.sum(axis=1)

            max_gain = gains.max()
            if max_gain == 0:
                break

            best_candidates = np.where(gains == max_gain)[0]
            if randomized and len(best_candidates) > 1:
                chosen = rng.choice(best_candidates.tolist())
            else:
                chosen = best_candidates[0]

            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False

        return code

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively using numpy."""
        selected = list(selected)
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            counts[neighbors_arr[w, :]] += 1

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]

                # Check if any word is uniquely covered by w
                counts[neighbors_arr[w, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[w, :]] += 1
                else:
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """Try to reduce the set by swapping or removing codewords."""
        selected = list(selected)
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            counts[neighbors_arr[w, :]] += 1

        for _ in range(100):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]

                # Try removing w
                counts[neighbors_arr[w, :]] -= 1
                if not np.any(counts == 0):
                    # Can remove
                    selected.pop(idx)
                    improved = True
                    break
                else:
                    counts[neighbors_arr[w, :]] += 1
                    # Find uniquely covered words
                    unique_mask = (counts == 1) & (np.isin(np.arange(total_space), neighbors_arr[w, :]))
                    # Simpler: check which neighbors of w have count 1
                    nb = neighbors_arr[w, :]
                    unique_words = nb[counts[nb] == 1]

                    if len(unique_words) == 0:
                        continue

                    # Find replacement: a word w2 that covers all unique_words
                    # and is not already in selected (or we can swap)
                    # For efficiency, try a few random candidates
                    for _attempt in range(20):
                        w2 = random.randint(0, total_space - 1)
                        if w2 == w:
                            continue
                        # Check if w2 covers all unique_words
                        w2_neighbors = set(neighbors_arr[w2, :].tolist())
                        if all(uw in w2_neighbors for uw in unique_words):
                            # Compute gain
                            old_nb_set = set(neighbors_arr[w, :].tolist())
                            new_nb_set = w2_neighbors
                            gain = len(new_nb_set - old_nb_set)
                            if gain > 0 or len(selected) > 0:
                                # Swap
                                counts[neighbors_arr[w, :]] -= 1
                                counts[neighbors_arr[w2, :]] += 1
                                selected[idx] = w2
                                improved = True
                                break
                    if improved:
                        break

            if not improved:
                break

        return selected

    def two_phase_greedy(seed=None):
        """Two-phase: first greedy, then a second pass with different randomization."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        # Phase 1: standard greedy
        uncovered = np.ones(total_space, dtype=np.bool_)
        code = []

        while uncovered.any() and time.time() < deadline - 10:
            uncovered_expanded = uncovered[neighbors_arr]
            gains = uncovered_expanded.sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            best_candidates = np.where(gains == max_gain)[0]
            if len(best_candidates) > 1:
                chosen = rng.choice(best_candidates.tolist())
            else:
                chosen = best_candidates[0]
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False

        # Phase 2: try to replace some codewords with better ones
        # Rebuild coverage
        counts = np.zeros(total_space, dtype=np.int32)
        for w in code:
            counts[neighbors_arr[w, :]] += 1

        # Try to find codewords that can be swapped
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                w = code[idx]
                counts[neighbors_arr[w, :]] -= 1
                if not np.any(counts == 0):
                    code.pop(idx)
                    improved = True
                    break
                else:
                    counts[neighbors_arr[w, :]] += 1
                    # Try to find a better replacement
                    nb = neighbors_arr[w, :]
                    unique_words = nb[counts[nb] == 1]
                    if len(unique_words) == 0:
                        continue
                    for _ in range(10):
                        w2 = rng.randint(0, total_space - 1)
                        if w2 == w:
                            continue
                        w2_nb = neighbors_arr[w2, :]
                        w2_set = set(w2_nb.tolist())
                        if all(uw in w2_set for uw in unique_words):
                            old_set = set(nb.tolist())
                            if len(w2_set - old_set) > 0:
                                counts[neighbors_arr[w, :]] -= 1
                                counts[neighbors_arr[w2, :]] += 1
                                code[idx] = w2
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break

        return code

    best_size = total_space
    best_selected = list(range(total_space))

    # Run deterministic greedy first
    if time.time() < deadline - 20:
        sol = run_greedy(randomized=False)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 10)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Run randomized greedy restarts
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() > deadline - 25:
            break

        sol = run_greedy(randomized=True, seed=r)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 10)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Run two-phase greedy
    for r in range(50):
        if time.time() > deadline - 20:
            break
        sol = two_phase_greedy(seed=1000 + r)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 10)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Final polish and local search
    if time.time() < deadline - 5:
        polished = polish(best_selected[:], deadline - 3)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    if time.time() < deadline - 2:
        ls = local_search(best_selected[:], deadline - 1)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Final polish after local search
    if time.time() < deadline - 0.5:
        polished = polish(best_selected[:], deadline - 0.2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
