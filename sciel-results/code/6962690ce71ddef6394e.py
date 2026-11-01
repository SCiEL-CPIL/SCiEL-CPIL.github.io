"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    total_words = q ** n

    def word_to_idx(word, q, n):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx, q, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors for radius=1 (ball of size 1 + n*(q-1))
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx, q, n)
        neighbors[idx].append(idx)
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        neighbors[idx].append(word_to_idx(w, q, n))
                w[pos] = orig

    ball_size = 1 + n * (q - 1)
    neighbor_matrix = np.zeros((total_words, ball_size), dtype=np.int32)
    for idx in range(total_words):
        neighbor_matrix[idx] = neighbors[idx]

    def greedy_cover(restart_seed=0, initial_codewords=None, temperature=1.0,
                     use_symmetry_tie_break=False):
        """Greedy set cover with a smooth surrogate (softmax) over per-candidate
        uncovered-coverage counts. Lower temperature -> more greedy; higher ->
        more exploration among near-best candidates."""
        if time.time() > deadline:
            return None

        rng = random.Random(restart_seed)
        codewords = []
        uncovered_mask = np.ones(total_words, dtype=bool)

        if initial_codewords:
            for cw in initial_codewords:
                if cw not in codewords:
                    codewords.append(cw)
                    uncovered_mask[neighbor_matrix[cw]] = False

        # Adaptive temperature: start broad (explore), anneal toward greedy
        T = temperature
        T_min = 0.5
        T_decay = 0.985

        while uncovered_mask.any() and time.time() < deadline:
            # Vectorized coverage counts per candidate
            counts = uncovered_mask[neighbor_matrix].sum(axis=1)  # shape (total_words,)

            best_count = int(counts.max())
            if best_count == 0:
                break

            # Smooth surrogate: softmax over exp((counts - best_count)/T)
            # This is a soft-max of the max-coverage objective, giving a
            # probability distribution that favors high-count candidates but
            # still samples near-best ones (avoids brittle hard-argmax ties).
            shifted = (counts.astype(np.float64) - best_count) / T
            # exp of shifted is <= 1; sum for normalization
            exp_vals = np.exp(shifted)
            total = exp_vals.sum()
            if total <= 0 or not np.isfinite(total):
                # Fallback to hard argmax if numerical issues
                best_cw = int(np.argmax(counts))
            else:
                probs = exp_vals / total
                # Sample from the distribution (smooth surrogate selection)
                r = rng.random()
                cdf = np.cumsum(probs)
                best_cw = int(np.searchsorted(cdf, r, side='right'))
                best_cw = min(best_cw, total_words - 1)

            # Symmetry tie-break refinement: among candidates within 1 of best,
            # prefer the one whose ball overlaps most with other top candidates
            # (a smooth auxiliary to break symmetries). Only when counts are tied.
            if use_symmetry_tie_break:
                near = np.where(counts >= best_count - 1)[0]
                if len(near) > 1:
                    near_set = set(int(x) for x in near)
                    scores = np.array([
                        len(set(neighbor_matrix[int(c)]) & near_set)
                        for c in near
                    ])
                    # Blend: pick from near with probability proportional to scores
                    # (smooth combination of coverage and overlap)
                    if scores.sum() > 0:
                        sp = scores / scores.sum()
                        r2 = rng.random()
                        best_cw = int(near[int(np.searchsorted(np.cumsum(sp), r2, side='right'))])
                        best_cw = min(best_cw, total_words - 1)

            codewords.append(best_cw)
            uncovered_mask[neighbor_matrix[best_cw]] = False

            # Anneal temperature
            T = max(T_min, T * T_decay)

        return codewords

    def remove_redundant(codewords):
        """Iteratively remove codewords whose entire ball is covered by >=2
        other codewords (epigraph-style: minimize count subject to coverage)."""
        if time.time() > deadline:
            return codewords

        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage[neighbor_matrix[cw]] += 1

        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw = codewords[i]
                if coverage[neighbor_matrix[cw]].min() >= 2:
                    coverage[neighbor_matrix[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        return codewords

    def polish(codewords):
        """High-precision final pass: try removing each codeword and re-covering
        the newly-uncovered words with a minimal greedy set; keep if it shrinks
        the code. This is a tighter-tolerance local solve on top of the greedy."""
        if time.time() > deadline:
            return codewords

        codewords = list(codewords)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw = codewords[i]
                # Tentatively remove
                remaining = [c for j, c in enumerate(codewords) if j != i]
                # Compute which words become uncovered
                coverage = np.zeros(total_words, dtype=np.int32)
                for c in remaining:
                    coverage[neighbor_matrix[c]] += 1
                newly_uncovered = (coverage[neighbor_matrix[cw]] == 0)
                n_uncovered = int(newly_uncovered.sum())
                if n_uncovered == 0:
                    # Purely redundant
                    codewords = remaining
                    improved = True
                    break
                # Greedily re-cover the newly-uncovered words
                # A codeword covers a word if the word is in its ball.
                # We need to cover n_uncovered words; pick codewords covering
                # the most of them.
                uncv_mask = np.zeros(total_words, dtype=bool)
                uncv_mask[neighbor_matrix[cw]] = newly_uncovered
                added = []
                temp = uncv_mask.copy()
                while temp.any() and len(added) < n_uncovered:
                    counts = temp[neighbor_matrix].sum(axis=1)
                    bc = int(counts.max())
                    if bc == 0:
                        break
                    best_cw = int(np.argmax(counts))
                    added.append(best_cw)
                    temp[neighbor_matrix[best_cw]] = False
                # Keep only if total size shrinks
                if len(remaining) + len(added) < len(codewords):
                    codewords = remaining + added
                    improved = True
                    break
        return codewords

    # Tetracode seed (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                initial_codewords.append(word_to_idx(word, q, n))

    best_result = None
    best_count = float('inf')

    # Seed with tetracode + redundancy removal
    reduced = remove_redundant(initial_codewords.copy())
    if len(reduced) < best_count:
        best_count = len(reduced)
        best_result = reduced

    # Multiple greedy restarts with varied temperature + smooth surrogate
    for seed in range(80):
        if time.time() > deadline:
            break
        # Vary temperature: lower = more greedy, higher = more exploration
        temp = 0.8 + (seed % 5) * 0.4  # 0.8, 1.2, 1.6, 2.0, 2.4
        use_sym = (seed % 4 == 0)
        result = greedy_cover(restart_seed=seed, initial_codewords=None,
                              temperature=temp, use_symmetry_tie_break=use_sym)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result

    # Hybrid: partial tetracode + greedy
    for seed in range(40):
        if time.time() > deadline:
            break
        rng = random.Random(seed + 1000)
        k = rng.randint(20, 60)
        initial = rng.sample(initial_codewords, k=min(len(initial_codewords), k))
        temp = 0.8 + (seed % 5) * 0.4
        use_sym = (seed % 4 == 0)
        result = greedy_cover(restart_seed=seed, initial_codewords=initial,
                              temperature=temp, use_symmetry_tie_break=use_sym)
        if result is not None:
            result = remove_redundant(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result

    # Final high-precision polish on the best result
    if best_result is not None:
        best_result = polish(best_result)
        best_result = remove_redundant(best_result)

    if best_result is None:
        return [idx_to_word(idx, q, n) for idx in initial_codewords]

    return [idx_to_word(idx, q, n) for idx in best_result]
# EVOLVE-BLOCK-END
