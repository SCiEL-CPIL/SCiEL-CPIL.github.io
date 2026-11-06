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
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0
        sub = color[aps_arr[ap_idx]]
        off = pos_ap_offsets[pos]
        curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
        curr_count = int(np.sum(curr_mono))
        
        # Vectorized: after flipping position 'off' in each AP, check if all 4 become equal
        # For each AP, the new values are: flip sub[row, off[j]]
        # We need to check if all 4 positions become equal after the flip
        # Equivalent: count positions equal to sub[row, 0] before flip
        # If flipping makes all equal, then before flip, exactly 3 were equal to the new value
        # Let's compute: for each AP, what's the majority?
        
        # After flip at position o:
        # new_vals = sub with sub[:, o] flipped
        # mono_new = all 4 new_vals are equal
        # This happens iff:
        #   - Before: 3 of the 4 were equal to (1 - sub[:, o]) and sub[:, o] was the odd one out
        #   - OR: Before: all 4 were equal (mono) and we're breaking it (but that's -1)
        
        # Simpler vectorized approach:
        # For each AP, count how many positions have value 0
        # After flip at position o, the new count of 0s depends on sub[:, o]
        # mono_new iff new_count_0 == 0 or new_count_0 == 4
        
        zeros = (sub == 0)
        zero_counts = zeros.sum(axis=1)  # count of 0s in each AP
        
        # After flipping position o:
        # If sub[:, o] == 0, it becomes 1, so zero_count decreases by 1
        # If sub[:, o] == 1, it becomes 0, so zero_count increases by 1
        flipped_vals = sub[:, off]  # value at the flipped position
        new_zero_counts = zero_counts - flipped_vals + (1 - flipped_vals)
        # Wait: if flipped_vals[j] == 0 (was 0, now 1), zero_count decreases by 1
        # if flipped_vals[j] == 1 (was 1, now 0), zero_count increases by 1
        # new_zero_counts = zero_counts - flipped_vals + (1 - flipped_vals)
        # = zero_counts + 1 - 2*flipped_vals
        
        new_zero_counts = zero_counts + 1 - 2 * flipped_vals
        new_mono = (new_zero_counts == 0) | (new_zero_counts == 4)
        new_count = int(np.sum(new_mono))
        return new_count - curr_count

    def local_search(color, score, time_budget=5.0):
        """Greedy local search: keep flipping if it improves. Uses random order for diversity."""
        improved = True
        start = time.time()
        rng = np.random.default_rng(42)
        while improved and time.time() - start < time_budget:
            improved = False
            order = rng.permutation(n)
            for pos in order:
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        return color, score

    def simulated_annealing(color, score, time_budget=10.0):
        """Simulated annealing with single flip moves, optimized for speed."""
        start = time.time()
        T_start = 3.0
        T_end = 0.002
        sa_best_score = score
        sa_best_color = color.copy()
        rng = np.random.default_rng(99)
        
        while time.time() - start < time_budget:
            elapsed = time.time() - start
            remaining_frac = max(0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.2

            # Single flip
            pos = rng.integers(0, n)
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

        # Return best seen during SA
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

    # Evaluate all structured candidates, keep the best few for refinement
    candidates = generate_structured_colorings()

    # Score all candidates
    scored = []
    for c in candidates:
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))

    scored.sort(key=lambda x: x[0])

    # Also add some random starts and mutated Thue-Morse for diversity
    rng_starts = np.random.default_rng(777)
    tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    for _ in range(20):
        # Mutated Thue-Morse: flip 5-15 random positions
        mut = tm.copy()
        num_flips = rng_starts.integers(5, 16)
        flip_positions = rng_starts.choice(n, size=num_flips, replace=False)
        mut[flip_positions] = 1 - mut[flip_positions]
        s = count_mono_4ap(mut)
        scored.append((s, mut.copy()))
    
    # Pure random starts
    for _ in range(10):
        c = rng_starts.integers(0, 2, n).astype(np.int8)
        s = count_mono_4ap(c)
        scored.append((s, c.copy()))
    
    scored.sort(key=lambda x: x[0])

    # Take top candidates for refinement
    top_k = min(40, len(scored))
    remaining_time = deadline - time.time()
    
    for rank in range(top_k):
        if time.time() > deadline - 10:
            break
        score, color = scored[rank]
        if score < best_score:
            best_score = score
            best_color = color.copy()

        # Local search refinement
        time_remaining = deadline - time.time() - 10
        if time_remaining < 1:
            break
        ls_budget = min(2.0, time_remaining * 0.15)
        color_ls, score_ls = local_search(color.copy(), score, time_budget=ls_budget)
        if score_ls < best_score:
            best_score = score_ls
            best_color = color_ls.copy()

        # Simulated annealing from best local search result
        time_remaining = deadline - time.time() - 10
        if time_remaining > 2:
            sa_budget = min(5.0, time_remaining * 0.40)
            color_sa, score_sa = simulated_annealing(color_ls.copy(), score_ls, time_budget=sa_budget)
            if score_sa < best_score:
                best_score = score_sa
                best_color = color_sa.copy()

            # Final local search after SA
            time_remaining = deadline - time.time() - 10
            if time_remaining > 1:
                ls2_budget = min(2.0, time_remaining * 0.20)
                color_final, score_final = local_search(color_sa.copy(), score_sa, time_budget=ls2_budget)
                if score_final < best_score:
                    best_score = score_final
                    best_color = color_final.copy()

    # Final polish on the overall best: multiple rounds of LS + SA
    if best_color is not None and time.time() < deadline - 5:
        # Round 1: Local search
        if time.time() < deadline - 4:
            ls_budget = min(2.0, (deadline - time.time() - 4) * 0.3)
            color_final, score_final = local_search(best_color.copy(), best_score, time_budget=ls_budget)
            if score_final < best_score:
                best_score = score_final
                best_color = color_final.copy()

        # Round 2: SA
        if time.time() < deadline - 3:
            sa_budget = min(5.0, deadline - time.time() - 3)
            color_sa_f, score_sa_f = simulated_annealing(best_color.copy(), best_score, time_budget=sa_budget)
            if score_sa_f < best_score:
                best_score = score_sa_f
                best_color = color_sa_f.copy()

        # Round 3: Final local search
        if time.time() < deadline - 2:
            ls_budget = deadline - time.time() - 2
            color_final2, score_final2 = local_search(best_color.copy(), best_score, time_budget=ls_budget)
            if score_final2 < best_score:
                best_score = score_final2
                best_color = color_final2.copy()

    if best_color is None:
        best_color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    return [int(c) for c in best_color]
