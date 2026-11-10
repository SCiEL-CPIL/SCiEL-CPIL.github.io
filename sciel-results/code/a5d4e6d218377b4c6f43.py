import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 118.0  # safety margin before 150s kill

    total_words = q ** n

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all words (radius-1 ball)
    neighbors = np.empty(total_words, dtype=object)
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[i] = new_val
                nbrs.append(word_to_idx(w))
            w[i] = orig
        neighbors[idx] = np.array(nbrs, dtype=np.int64)

    # Incidence matrix: row = word, col = neighbor index (1 if word covers col)
    # This lets us vectorize "new coverage" scans.
    ball_size = int(len(neighbors[0]))
    incidence = np.zeros((total_words, total_words), dtype=np.int8)
    for idx in range(total_words):
        incidence[idx, neighbors[idx]] = 1
    # Precompute list of neighbor arrays for fast iteration
    neighbor_lists = [neighbors[i] for i in range(total_words)]

    # Precompute a coverage bitset per word for fast operations
    # covered_mask: bool array total_words
    # For each word, neighbors[word] gives indices

    def covers(idx):
        return neighbors[idx]

    # --- Structured seed via Z_3 linear code ---
    # A linear code over Z_3 with generator matrix G (k x n) gives 3^k codewords.
    # For covering radius 1, we want a code whose cosets cover the space.
    # Try several random generator matrices and pick the best covering.

    def random_linear_code(k, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        # Random k x n matrix over Z_3, try to make rows independent
        rows = []
        attempts = 0
        while len(rows) < k and attempts < 200:
            attempts += 1
            row = [rng.randrange(q) for _ in range(n)]
            if all(r != 0 for r in row):
                # Check independence (over Z_3)
                independent = True
                for existing in rows:
                    # Check if row is a scalar multiple of existing
                    if existing[0] != 0:
                        factor = (row[0] * pow(existing[0], q - 2, q)) % q
                        if all((row[j] - factor * existing[j]) % q == 0 for j in range(n)):
                            independent = False
                            break
                    else:
                        # existing[0] == 0, check if row is multiple
                        found_pivot = False
                        for j in range(n):
                            if existing[j] != 0:
                                factor = (row[j] * pow(existing[j], q - 2, q)) % q
                                if all((row[a] - factor * existing[a]) % q == 0 for a in range(n)):
                                    independent = False
                                break
                        if not independent:
                            break
                if independent:
                    rows.append(row)
        if len(rows) < k:
            # Fill with standard basis if needed
            while len(rows) < k:
                row = [0] * n
                row[len(rows)] = 1
                rows.append(row)
        return rows

    def linear_code_codewords(G):
        k = len(G)
        codewords = []
        for coeffs in range(q ** k):
            c = coeffs
            cw = [0] * n
            for i in range(k):
                cw[i] = c % q
                c //= q
            word = [0] * n
            for j in range(n):
                s = 0
                for i in range(k):
                    s += cw[i] * G[i][j]
                word[j] = s % q
            codewords.append(word)
        return codewords

    def coverage_size(code_words):
        covered = np.zeros(total_words, dtype=bool)
        for w in code_words:
            idx = word_to_idx(w)
            covered[neighbors[idx]] = True
        return int(np.sum(covered))

    # Try several linear codes to find a good seed
    best_seed = None
    best_seed_coverage = -1
    for k in [3, 4]:
        for trial in range(30):
            if time.time() - start_time > DEADLINE - 60:
                break
            G = random_linear_code(k, seed=trial * 1000 + k)
            cw = linear_code_codewords(G)
            cov = coverage_size(cw)
            if cov > best_seed_coverage:
                best_seed_coverage = cov
                best_seed = cw
            if time.time() - start_time > DEADLINE - 60:
                break
        if best_seed_coverage >= total_words - 50:
            break

    # If no good linear seed, use greedy from scratch
    if best_seed is None:
        best_seed = []

    # --- Greedy set-cover to complete ---
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    for w in best_seed:
        idx = word_to_idx(w)
        code_indices.append(idx)
        covered[neighbors[idx]] = True

    # Greedy fill (vectorized new-coverage scan)
    while not np.all(covered):
        if time.time() - start_time > DEADLINE - 20:
            break
        # For each candidate word c, new coverage = number of its neighbors not yet covered
        # = sum over columns j in ball(c) of (1 - covered[j])
        # = ball_size - sum over columns j in ball(c) of covered[j]
        # Use incidence: incidence[c, :] has 1 at ball(c) positions
        # new_cov[c] = ball_size - (incidence[c] * covered).sum()
        # But covered is bool; convert to int8 for matmul
        covered_int = covered.astype(np.int8)
        # Sum over columns where incidence[c, col] == 1
        # new_cov = ball_size - incidence @ covered_int
        new_cov = ball_size - incidence.dot(covered_int)
        best_idx = int(np.argmax(new_cov))
        best_new = int(new_cov[best_idx])
        if best_new <= 0:
            break
        code_indices.append(best_idx)
        covered[neighbor_lists[best_idx]] = True

    # --- Local search polish ---
    # Try to swap a codeword for one that covers more uncovered, or remove redundant ones
    def is_covered_by(idx, code_list, exclude=None):
        # Check if idx is covered by some codeword in code_list (excluding 'exclude' index)
        for ci in code_list:
            if ci == exclude:
                continue
            nbrs = neighbors[ci]
            if idx in nbrs:
                return True
        return False

    def full_coverage_check(code_list):
        cov = np.zeros(total_words, dtype=bool)
        for ci in code_list:
            cov[neighbors[ci]] = True
        return cov

    # Local search: try removing redundant codewords, then swapping
    # Use coverage-count array for efficient redundancy detection
    def build_coverage_count(code_list):
        """Return an int array where count[j] = number of codewords whose ball contains j."""
        cnt = np.zeros(total_words, dtype=np.int32)
        for ci in code_list:
            cnt[neighbor_lists[ci]] += 1
        return cnt

    for iteration in range(200):
        if time.time() - start_time > DEADLINE - 5:
            break

        # Try to remove redundant codewords using coverage counts
        removed_any = True
        while removed_any:
            removed_any = False
            cnt = build_coverage_count(code_indices)
            for i in range(len(code_indices)):
                ci = code_indices[i]
                # ci is redundant if every neighbor j of ci has cnt[j] >= 2
                # (i.e., even after removing ci, cnt[j] >= 1)
                if np.all(cnt[neighbor_lists[ci]] >= 2):
                    code_indices.pop(i)
                    removed_any = True
                    break

        # Try swaps: for each codeword, see if removing it leaves some words uncovered.
        # If so, try to find a replacement that covers all those words.
        improved = True
        while improved:
            improved = False
            cnt = build_coverage_count(code_indices)
            for i in range(len(code_indices)):
                ci = code_indices[i]
                # After removing ci, words with cnt[j]==1 in ball(ci) become uncovered
                # Actually: words j where cnt[j] == 1 and j is in ball(ci) become uncovered
                # But we need to check ALL words, not just ball(ci).
                # A word j is uncovered after removing ci iff cnt[j] == 1 and ci covers j.
                # So uncovered set = { j : cnt[j] == 1 and j in ball(ci) }
                # This is equivalent to: for j in ball(ci), if cnt[j] == 1, j is lost.
                nbrs_ci = neighbor_lists[ci]
                lost = nbrs_ci[cnt[nbrs_ci] == 1]
                if len(lost) == 0:
                    # ci is redundant
                    code_indices.pop(i)
                    improved = True
                    break
                # Try to find a replacement c such that ball(c) covers all of lost
                # i.e., for all j in lost, j in ball(c)
                # Using incidence: incidence[c, lost] must all be 1
                # Check: np.all(incidence[lost, lost]) is wrong.
                # We need: for each j in lost, incidence[c, j] == 1
                # That's: np.all(incidence[c, lost])
                # Vectorize over all candidates c:
                # candidate_ok[c] = all(incidence[c, lost] == 1)
                # = (incidence[:, lost] == 1).all(axis=1)
                if len(lost) > 0:
                    # Only consider candidates not already in code_indices
                    in_code = np.zeros(total_words, dtype=bool)
                    in_code[code_indices] = True
                    # Check which candidates cover all lost words
                    covers_all = (incidence[:, lost] == 1).all(axis=1)
                    # Exclude current codewords
                    covers_all[in_code] = False
                    if np.any(covers_all):
                        # Pick the candidate that maximizes additional coverage
                        # (to potentially enable further removals)
                        # Additional coverage = number of words in ball(c) that are currently cnt==1
                        # (those would go from cnt=1 to cnt=2, no loss) plus words with cnt==0
                        # Actually we just need to maintain coverage; pick one with max new coverage
                        # for future flexibility
                        cands = np.where(covers_all)[0]
                        if len(cands) > 0:
                            # Score each candidate by how many currently-uncovered-or-single-covered
                            # words it adds coverage to (i.e., ball(c) intersect {j: cnt[j] <= 1})
                            single_or_zero = cnt <= 1
                            scores = incidence[cands].dot(single_or_zero.astype(np.int32))
                            best_c = int(cands[np.argmax(scores)])
                            code_indices[i] = best_c
                            improved = True
                            break
            if improved:
                continue

    # Final coverage verification
    final_cov = full_coverage_check(code_indices)
    if not np.all(final_cov):
        # Emergency fill (vectorized)
        covered = final_cov
        while not np.all(covered):
            if time.time() - start_time > DEADLINE - 1:
                break
            covered_int = covered.astype(np.int8)
            new_cov = ball_size - incidence.dot(covered_int)
            best_idx = int(np.argmax(new_cov))
            best_new = int(new_cov[best_idx])
            if best_new <= 0:
                break
            code_indices.append(best_idx)
            covered[neighbor_lists[best_idx]] = True

    # Deduplicate
    seen = set()
    unique_code = []
    for idx in code_indices:
        if idx not in seen:
            seen.add(idx)
            unique_code.append(idx)

    return [idx_to_word(idx) for idx in unique_code]
