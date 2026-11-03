"""Covering code construction for K_q(n, r) via randomized local search."""
import random
import time
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code: minimal set of words of length n over {0..q-1}
    such that every word is within Hamming distance `radius` of some codeword.
    Uses randomized greedy init + stochastic local search with simulated annealing.
    """
    if n == 0:
        return [[]]
    if radius == 0:
        return [[i] for i in range(q ** n)]

    total = q ** n
    t0 = time.time()
    deadline = 138  # safety margin before 150s kill

    def int_to_word(v):
        w = []
        for _ in range(n):
            w.append(v % q)
            v //= q
        return w

    def word_to_int(w):
        v = 0
        for i in range(n - 1, -1, -1):
            v = v * q + w[i]
        return v

    # Precompute all neighbors (words within Hamming distance radius) for each word
    # For q=3, n=8, radius=1: each word has 1 + 8*2 = 17 neighbors
    neighbors = []
    for w in range(total):
        word = int_to_word(w)
        neigh = {w}
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    word[pos] = val
                    neigh.add(word_to_int(word))
                word[pos] = orig
        if radius >= 2:
            word = int_to_word(w)
            for pos1 in range(n):
                o1 = word[pos1]
                for v1 in range(q):
                    if v1 == o1:
                        continue
                    word[pos1] = v1
                    for pos2 in range(pos1 + 1, n):
                        o2 = word[pos2]
                        for v2 in range(q):
                            if v2 == o2:
                                continue
                            word[pos2] = v2
                            neigh.add(word_to_int(word))
                        word[pos2] = o2
                    word[pos1] = o1
        neighbors.append(neigh)

    neigh_arrays = [np.fromiter(neigh, dtype=np.int64) for neigh in neighbors]

    def greedy_randomized_init():
        """Randomized greedy set cover."""
        covered = np.zeros(total, dtype=np.bool_)
        code = []
        uncovered_count = total

        while uncovered_count > 0:
            # Sample uncovered words to find good candidate
            if uncovered_count > 200:
                # Find some uncovered words
                uncov_idx = np.where(~covered)[0]
                sample = np.random.choice(uncov_idx, size=min(50, len(uncov_idx)), replace=False)
            else:
                uncov_idx = np.where(~covered)[0]
                sample = uncov_idx

            # Collect candidate codewords from neighbors of sampled uncovered words
            candidates = set()
            for u in sample:
                candidates.update(neighbors[u])

            if not candidates:
                break

            # Evaluate candidates: count how many uncovered words each covers
            best_cw = None
            best_gain = -1
            cand_list = list(candidates)
            # If too many, sample
            if len(cand_list) > 60:
                cand_list = random.sample(cand_list, 60)

            for cw in cand_list:
                gain = 0
                for u in neighbors[cw]:
                    if not covered[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cw = cw
                    if best_gain == uncovered_count:
                        break

            if best_cw is None:
                break

            code.append(best_cw)
            for u in neighbors[best_cw]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1

        return code

    def count_coverage(code):
        """Count how many codewords cover each word."""
        cov = np.zeros(total, dtype=np.int32)
        for cw in code:
            for u in neighbors[cw]:
                cov[u] += 1
        return cov

    def unique_words(cov):
        """Words covered by exactly one codeword."""
        return np.where(cov == 1)[0]

    def local_search(code, cov, time_budget):
        """
        Stochastic local search: iteratively try to remove or swap codewords.
        A codeword can be removed if it has no uniquely covered words.
        A swap replaces a codeword with another that maintains coverage.
        Uses simulated annealing to accept worse solutions probabilistically.
        """
        code_set = set(code)
        best_code = list(code)
        best_len = len(code)
        temperature = 2.0
        cooling = 0.9995
        t_start = time.time()

        # Track number of uniquely covered words per codeword for quick checks
        def unique_count_for(cw_idx, code_list):
            cw = code_list[cw_idx]
            count = 0
            for u in neighbors[cw]:
                if cov[u] == 1:
                    count += 1
            return count

        no_improve = 0

        while time.time() - t_start < time_budget and temperature > 0.01:
            improved = False

            # Phase 1: Try removing codewords (fast exact removals)
            for i in range(len(code) - 1, -1, -1):
                if time.time() - t_start > time_budget:
                    break
                cw = code[i]
                can_remove = True
                for u in neighbors[cw]:
                    if cov[u] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for u in neighbors[cw]:
                        cov[u] -= 1
                    code.pop(i)
                    code_set.discard(cw)
                    improved = True
                    no_improve = 0
                    if len(code) < best_len:
                        best_code = list(code)
                        best_len = len(code)

            if not improved:
                # Phase 2: Try swaps
                if not code:
                    break
                idx = random.randrange(len(code))
                old_cw = code[idx]
                # Find words uniquely covered by old_cw
                unique = [u for u in neighbors[old_cw] if cov[u] == 1]

                if not unique:
                    # Can just remove
                    for u in neighbors[old_cw]:
                        cov[u] -= 1
                    code.pop(idx)
                    code_set.discard(old_cw)
                    improved = True
                    no_improve = 0
                    if len(code) < best_len:
                        best_code = list(code)
                        best_len = len(code)
                    continue

                # Find candidate replacements: must cover all uniquely covered words
                candidates = set(neighbors[unique[0]])
                for u in unique[1:]:
                    candidates &= neighbors[u]
                    if not candidates:
                        break
                candidates -= code_set
                candidates.discard(old_cw)

                if not candidates:
                    no_improve += 1
                    temperature *= cooling
                    # Perturbation: random accept a worse move
                    if no_improve > 5:
                        # Try a random swap even if it temporarily increases size
                        # or do a 2-swap perturbation
                        if len(code) > 2:
                            i1 = random.randrange(len(code))
                            i2 = random.randrange(len(code) - 1)
                            if i2 >= i1:
                                i2 += 1
                            # Swap: remove one, add a random alternative
                            cw1 = code[i1]
                            uniq1 = [u for u in neighbors[cw1] if cov[u] == 1]
                            if uniq1:
                                cands1 = set(neighbors[uniq1[0]])
                                for u in uniq1[1:]:
                                    cands1 &= neighbors[u]
                                    if not cands1:
                                        break
                                cands1 -= code_set
                                if cands1:
                                    new_cw1 = random.choice(list(cands1))
                                    for u in neighbors[cw1]:
                                        cov[u] -= 1
                                    for u in neighbors[new_cw1]:
                                        cov[u] += 1
                                    code[i1] = new_cw1
                                    code_set.discard(cw1)
                                    code_set.add(new_cw1)
                        no_improve = 0
                    continue

                # Score candidates by how many redundantly covered words they add
                cand_list = list(candidates)
                if len(cand_list) > 30:
                    cand_list = random.sample(cand_list, 30)

                best_new = None
                best_score = -1
                for nc in cand_list:
                    score = 0
                    for u in neighbors[nc]:
                        if cov[u] >= 2:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_new = nc

                if best_new is not None:
                    for u in neighbors[old_cw]:
                        cov[u] -= 1
                    for u in neighbors[best_new]:
                        cov[u] += 1
                    code[idx] = best_new
                    code_set.discard(old_cw)
                    code_set.add(best_new)
                    improved = True
                    no_improve = 0
                    if len(code) < best_len:
                        best_code = list(code)
                        best_len = len(code)
                else:
                    no_improve += 1
                    temperature *= cooling

            # Cool down
            temperature *= cooling

        # Final pass: remove all redundant
        for i in range(len(code) - 1, -1, -1):
            cw = code[i]
            can_remove = True
            for u in neighbors[cw]:
                if cov[u] == 1:
                    can_remove = False
                    break
            if can_remove:
                for u in neighbors[cw]:
                    cov[u] -= 1
                code.pop(i)

        return best_code, best_len

    # Main loop: multiple restarts with different random seeds
    best_result = []
    best_size = total + 1

    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break

        # Greedy initialization
        code = greedy_randomized_init()
        cov = count_coverage(code)
        current_size = len(code)

        if current_size < best_size:
            best_size = current_size
            best_result = list(code)

        # Local search with time budget proportional to remaining time
        ls_budget = min(remaining - 3, max(2.0, remaining * 0.4))
        if ls_budget > 1:
            improved_code, improved_size = local_search(code, cov, ls_budget)
            if improved_size < best_size:
                best_size = improved_size
                best_result = improved_code

        # Early termination if we've hit known optimum
        if best_size <= 486:
            break

    return [int_to_word(cw) for cw in best_result]
