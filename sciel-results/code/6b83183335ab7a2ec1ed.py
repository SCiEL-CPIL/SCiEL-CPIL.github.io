"""Model-and-optimize approach: structured parameterization + local search refinement."""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110  # safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))

    pos_ap_indices = []
    pos_ap_offsets = []
    for i in range(n):
        if pos_aps[i]:
            indices, offsets = zip(*pos_aps[i])
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_ap_offsets.append(np.array(offsets, dtype=np.int8))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_offsets.append(np.array([], dtype=np.int8))

    def count_mono_4ap(color):
        c = color[aps_arr]
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))

    def delta_flip(color, pos):
        """Vectorized delta for flipping position `pos`."""
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0
        sub = color[aps_arr[ap_idx]]  # shape (m, 4)
        off = pos_ap_offsets[pos]     # shape (m,)
        
        # Current mono count
        curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
        curr_count = int(np.sum(curr_mono))
        
        # After flipping position `off[j]` in row j, check if all 4 become equal.
        # For each row, the flipped value is 1 - sub[j, off[j]].
        # We need: all 4 positions equal to the flipped value.
        # Equivalently: the 3 unflipped positions must all equal 1 - sub[j, off[j]].
        # Which means: the 3 unflipped positions must all be equal to each other
        # AND equal to 1 - sub[j, off[j]].
        
        # Build a matrix where we zero out the flipped column:
        # unflipped[j, k] = sub[j, k] if k != off[j], else 0
        # Then check: all unflipped values in row j are equal AND equal to 1 - sub[j, off[j]]
        
        # Vectorized approach:
        # For each row, count how many of the other 3 positions match 1 - sub[j, off[j]]
        # If all 3 match, then flipping creates a mono AP.
        
        # Let's compute: for each row j, the target value is t_j = 1 - sub[j, off[j]]
        # We need sub[j, k] == t_j for all k != off[j]
        # i.e., sum over k != off[j] of (sub[j,k] == t_j) == 3
        
        # Create a boolean matrix: match[j, k] = (sub[j, k] == 1 - sub[j, off[j]])
        # Then for each row, subtract the entry at column off[j] (which is always True since sub[j,off[j]] != 1-sub[j,off[j]]... wait no)
        # Actually sub[j, off[j]] == 1 - sub[j, off[j]] is always False (since values are 0 or 1)
        # So match[j, off[j]] = False always.
        # Thus sum of match[j, :] over k gives the count of unflipped positions matching target.
        
        target = 1 - sub[np.arange(len(ap_idx)), off]  # shape (m,)
        match = (sub == target[:, None])  # shape (m, 4)
        # match[j, off[j]] is always False (since sub[j,off[j]] != 1-sub[j,off[j]])
        count_matches = match.sum(axis=1)  # shape (m,), values in {0,1,2,3}
        
        # Flipping creates mono iff count_matches[j] == 3 (all 3 unflipped match target)
        creates_mono = (count_matches == 3)
        
        # Flipping destroys mono iff curr_mono[j] is True
        # (flipping any position in a mono AP breaks it)
        
        delta = int(np.sum(creates_mono)) - int(np.sum(curr_mono))
        return delta

    def local_search(color, score, time_budget=5.0):
        """Fast greedy local search using vectorized delta_flip."""
        improved = True
        start = time.time()
        rng = np.random.default_rng(42)
        while improved and time.time() - start < time_budget:
            improved = False
            order = rng.permutation(n)
            for pos in order:
                if time.time() - start > time_budget:
                    break
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        return color, score

    def simulated_annealing(color, score, time_budget=10.0):
        """Simulated annealing using fast vectorized delta_flip for single flips."""
        start = time.time()
        T_start = 3.0
        T_end = 0.01
        sa_best_score = score
        sa_best_color = color.copy()
        rng = np.random.default_rng(99)
        
        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.5

            # Single flip only (much faster than multi-flip)
            pos = int(rng.integers(0, n))
            delta = delta_flip(color, pos)
            if delta <= 0:
                color[pos] = 1 - color[pos]
                score += delta
            else:
                if rng.random() < np.exp(-delta / max(T, 0.001)):
                    color[pos] = 1 - color[pos]
                    score += delta

            # Track best
            if score < sa_best_score:
                sa_best_score = score
                sa_best_color = color.copy()

        return sa_best_color, sa_best_score

    def generate_structured_colorings():
        """Generate a set of structured candidate colorings (model-based)."""
        candidates = []

        # Thue-Morse
        tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(tm.copy())

        # Bit-reversal Thue-Morse
        br = np.array([bin(i)[::-1].count('1') % 2 for i in range(n)], dtype=np.int8)
        candidates.append(br.copy())

        # Modular patterns: color[i] = (i mod p) < q for various p, q
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            for q in range(1, p):
                c = np.array([(i % p) < q for i in range(n)], dtype=np.int8)
                candidates.append(c)
            # Also flipped
            for q in range(1, p):
                c = np.array([(i % p) >= q for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Walsh functions: color[i] = bit k of i
        for k in range(8):
            c = np.array([(i >> k) & 1 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # XOR combinations of bits (Walsh-Hadamard basis) - select diverse masks
        for mask in range(1, 256, 7):  # Sample every 7th mask for diversity
            c = np.array([bin(i & mask).count('1') % 2 for i in range(n)], dtype=np.int8)
            candidates.append(c)

        # Linear phase: color[i] = (a*i + b) mod 2
        for a in range(1, 8):
            for b in range(2):
                c = np.array([(a * i + b) % 2 for i in range(n)], dtype=np.int8)
                candidates.append(c)

        # Quadratic: color[i] = (a*i*i + b*i + c) mod 2
        for a in range(1, 4):
            for b in range(2):
                for c in range(2):
                    col = np.array([(a * i * i + b * i + c) % 2 for i in range(n)], dtype=np.int8)
                    candidates.append(col)

        # Sine-wave quantization
        for freq in [1, 2, 3, 4, 5, 7, 8, 11]:
            for phase in range(4):
                vals = np.sin(2 * np.pi * freq * np.arange(n) / n + phase * np.pi / 2)
                c = (vals > 0).astype(np.int8)
                candidates.append(c)

        # Rademacher-like: random sign patterns with structure
        rng = np.random.default_rng(12345)
        for _ in range(30):
            c = rng.integers(0, 2, n).astype(np.int8)
            candidates.append(c)
        
        # Block patterns: alternating blocks of varying sizes
        for block_size in [4, 8, 16, 32, 64]:
            pattern = np.zeros(n, dtype=np.int8)
            for i in range(n):
                pattern[i] = (i // block_size) % 2
            candidates.append(pattern)
            candidates.append(1 - pattern)

        return candidates

    best_color = None
    best_score = float('inf')

    # Evaluate all structured candidates
    candidates = generate_structured_colorings()

    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))

    scored.sort(key=lambda x: x[0])

    # Multi-restart strategy: use fast vectorized delta to do many restarts
    # Phase 1: Local search on top candidates
    top_k = min(40, len(scored))
    for rank in range(top_k):
        if time.time() > deadline - 20:
            break
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()

        ls_budget = min(2.0, (deadline - time.time() - 20) / max(1, top_k - rank))
        if ls_budget > 0.1:
            color_ls, score_ls = local_search(color.copy(), score, time_budget=ls_budget)
            if score_ls < best_score:
                best_score = score_ls
                best_color = color_ls.copy()

    # Phase 2: SA restarts from diverse starting points (perturbed best + random)
    rng_restart = np.random.default_rng(777)
    sa_rounds = 0
    while time.time() < deadline - 15:
        sa_rounds += 1
        # Choose starting point: 60% perturbed best, 40% random
        if best_color is not None and rng_restart.random() < 0.6:
            start_color = best_color.copy()
            # Perturb: flip 5-15 random positions
            n_perturb = int(rng_restart.integers(5, 16))
            perturb_positions = rng_restart.choice(n, size=n_perturb, replace=False)
            start_color[perturb_positions] = 1 - start_color[perturb_positions]
            start_score = count_mono_4ap(start_color)
        else:
            start_color = rng_restart.integers(0, 2, n).astype(np.int8)
            start_score = count_mono_4ap(start_color)

        sa_budget = min(8.0, (deadline - time.time() - 15) * 0.4)
        if sa_budget < 1.0:
            break
        color_sa, score_sa = simulated_annealing(start_color, start_score, time_budget=sa_budget)
        if score_sa < best_score:
            best_score = score_sa
            best_color = color_sa.copy()

        # Quick LS after SA
        if time.time() < deadline - 10:
            ls_budget = min(2.0, (deadline - time.time() - 10) * 0.3)
            if ls_budget > 0.5:
                color_final, score_final = local_search(color_sa.copy(), score_sa, time_budget=ls_budget)
                if score_final < best_score:
                    best_score = score_final
                    best_color = color_final.copy()

    # Phase 3: Final intensive local search on the best
    if best_color is not None and time.time() < deadline - 5:
        ls_budget = deadline - time.time() - 5
        color_final, score_final = local_search(best_color.copy(), best_score, time_budget=ls_budget)
        if score_final < best_score:
            best_score = score_final
            best_color = color_final.copy()

    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]
